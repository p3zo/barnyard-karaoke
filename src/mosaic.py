"""Analysis and frame-selection logic for the audio mosaicing pipeline.

The notebooks handle orchestration and plotting; the logic that decides what a
frame is, how it is described, and which source frame replaces it lives here so
it can be tested (see tests/test_mosaic.py).
"""

import numpy as np
import essentia
import essentia.standard as estd

SAMPLE_RATE = 44100

# Melodia's defaults. The pitch contour is sampled every MELODIA_HOP_SIZE
# samples (2.9 ms), which is the time resolution available to the segmenter.
MELODIA_FRAME_SIZE = 2048
MELODIA_HOP_SIZE = 128

# MFCCs are averaged over fixed-size windows rather than computed in one shot
# over a whole note. A single FFT over a whole note makes the coefficients a
# function of note length: the same 440 Hz tone yields mfcc_1 = 200 over a
# 2048-sample frame but 81 over an 88200-sample one, so frames of different
# lengths are not comparable. Averaging fixed windows holds it to within 1%.
MFCC_FRAME_SIZE = 2048
MFCC_HOP_SIZE = 1024
N_MFCC = 13

MFCC_FEATURES = [f"mfcc_{i}" for i in range(N_MFCC)]
FEATURE_COLUMNS = ["mean_pitch", "loudness"] + MFCC_FEATURES


def load_audio(path):
    return estd.MonoLoader(filename=path, sampleRate=SAMPLE_RATE)()


def _mean_mfcc(frame):
    """Mean MFCC vector over fixed-size windows of `frame`."""
    windowing = estd.Windowing(type="hann")
    spectrum = estd.Spectrum()
    mfcc = estd.MFCC(inputSize=MFCC_FRAME_SIZE // 2 + 1, sampleRate=SAMPLE_RATE)

    # FrameGenerator yields nothing for input shorter than one window, so pad
    # rather than let a short note silently drop out of the collection.
    if len(frame) < MFCC_FRAME_SIZE:
        frame = np.pad(frame, (0, MFCC_FRAME_SIZE - len(frame)))

    coefficients = [
        mfcc(spectrum(windowing(window)))[1]
        for window in estd.FrameGenerator(
            essentia.array(frame),
            frameSize=MFCC_FRAME_SIZE,
            hopSize=MFCC_HOP_SIZE,
            startFromZero=True,
        )
    ]
    return np.mean(coefficients, axis=0)


def _loudness(frame):
    """Length-invariant loudness.

    Essentia's Loudness is Stevens' law, energy**0.67, so it grows with frame
    length; dividing by len(frame)**0.67 (rather than len(frame)) makes it the
    0.67 power of mean energy and comparable across notes of different length.
    """
    return estd.Loudness()(essentia.array(frame)) / len(frame) ** 0.67


def segment_notes(audio, min_duration, pitch_distance_threshold=30, rms_threshold=-4):
    """Estimate the predominant melody and segment it into notes.

    Returns (onsets, durations, midi_pitches, pitch_values), all in seconds
    except the MIDI pitches and the raw contour.
    """
    pitch_values, _ = estd.PredominantPitchMelodia(
        frameSize=MELODIA_FRAME_SIZE, hopSize=MELODIA_HOP_SIZE
    )(audio)

    onsets, durations, midi_pitches = estd.PitchContourSegmentation(
        hopSize=MELODIA_HOP_SIZE,
        minDuration=min_duration,
        pitchDistanceThreshold=pitch_distance_threshold,
        rmsThreshold=rms_threshold,
    )(pitch_values, audio)

    return onsets, durations, midi_pitches, pitch_values


def analyze_sound(audio_path, min_duration, audio_id=None, **segmentation_kwargs):
    """Describe every note in a sound as a row of features.

    A note spans onset -> onset + duration. Using the next onset as the end
    instead would fold the rest that follows a note into the note itself and
    drop the final note of every sound.
    """
    audio = load_audio(audio_path)
    onsets, durations, midi_pitches, _ = segment_notes(
        audio, min_duration, **segmentation_kwargs
    )

    rows = []
    for onset, duration, midi_pitch in zip(onsets, durations, midi_pitches):
        start = int(round(onset * SAMPLE_RATE))
        end = min(int(round((onset + duration) * SAMPLE_RATE)), len(audio))
        frame = audio[start:end]
        if len(frame) < 2:
            continue

        row = {
            "freesound_id": audio_id,
            "path": audio_path,
            "start_sample": start,
            "end_sample": end,
            "mean_pitch": midi_pitch,
            "loudness": _loudness(frame),
        }
        row.update(dict(zip(MFCC_FEATURES, _mean_mfcc(frame))))
        rows.append(row)

    return rows


def analyze_collection(df, min_duration, **segmentation_kwargs):
    """Analyze every sound in a collection DataFrame.

    Returns (rows, skipped_ids). Sounds with no detectable melodic contour
    yield no frames; they are returned rather than silently dropped because
    they are a large fraction of a collection of animal sounds.
    """
    rows, skipped_ids = [], []
    for position in range(len(df)):
        sound = df.iloc[position]
        print(
            "Analyzing sound with id {0} [{1}/{2}]".format(
                sound["freesound_id"], position + 1, len(df)
            )
        )
        sound_rows = analyze_sound(
            sound["path"],
            min_duration,
            audio_id=sound["freesound_id"],
            **segmentation_kwargs,
        )
        if not sound_rows:
            skipped_ids.append(sound["freesound_id"])
        rows += sound_rows

    print(
        "Extracted {0} frames from {1}/{2} sounds; {3} yielded no melodic "
        "contour and contribute nothing to the collection.".format(
            len(rows), len(df) - len(skipped_ids), len(df), len(skipped_ids)
        )
    )
    return rows, skipped_ids


def standardize(df_source, df_target, features):
    """Put features on a common scale, fitted on the source collection.

    Without this the nearest-neighbour distance is meaningless: mfcc_0 spans
    ~700 units, mean_pitch ~50 and loudness ~0.006, so on raw columns mfcc_0
    accounts for over half the squared distance and loudness for none of it.
    """
    values = df_source[features].to_numpy(dtype=float)
    mean = values.mean(axis=0)
    std = values.std(axis=0)

    # A feature that never varies carries no information; leaving std at 0
    # would divide by zero rather than say so.
    if np.any(std == 0):
        constant = [f for f, s in zip(features, std) if s == 0]
        raise ValueError(f"Features are constant across the source collection: {constant}")

    return (values - mean) / std, (df_target[features].to_numpy(dtype=float) - mean) / std


def select_source_frame(
    target_index,
    target_scaled,
    source_scaled,
    df_source,
    target_pitch,
    rng,
    n_candidates=10,
    max_pitch_deviation=0,
):
    """Pick a source frame to stand in for one target frame.

    Pitch is settled first: candidates are the frames at the smallest pitch
    deviation the collection can offer, provided that is within
    `max_pitch_deviation`. Only among those does the rest of the feature
    vector rank them, and one of the closest `n_candidates` is picked at
    random.

    Pitch has to win outright rather than compete on distance. It is one
    feature against thirteen MFCCs, so ranking the whole pool at once picks a
    timbrally closer frame a semitone flat even where an exact match exists.
    Ranking on pitch alone and drawing uniformly from the ten nearest is no
    better: for a target of MIDI 60 the ten nearest frames in the violin
    collection are 58-62 with no exact match at all.

    Returns (row, pitch_deviation).
    """
    deviations = np.abs(df_source["mean_pitch"].to_numpy(dtype=float) - target_pitch)
    best = deviations.min()
    if best > max_pitch_deviation:
        raise ValueError(
            f"No source frame within {max_pitch_deviation} semitones of MIDI "
            f"{target_pitch} (closest is {best:.0f} away). Widen "
            f"max_pitch_deviation or add source material in that register."
        )
    eligible = np.flatnonzero(deviations == best)

    distances = np.linalg.norm(source_scaled[eligible] - target_scaled[target_index], axis=1)
    closest = eligible[np.argsort(distances)[:n_candidates]]
    chosen = int(rng.choice(closest))
    return df_source.iloc[chosen], float(deviations[chosen])


def render_frame(source_audio, source_row, n_samples, fade_samples=220):
    """Cut `n_samples` from a source frame, never reading past the frame's end.

    Slicing the target's length out of the source ignores how long the source
    note actually is; half the eligible picks in the violin collection are
    shorter than the target frame, so the segment would run past the analysed
    note into whatever follows it in the file.
    """
    start = int(source_row["start_sample"])
    end = int(source_row["end_sample"])
    segment = np.array(source_audio[start : min(end, start + n_samples)], dtype=np.float64)

    # Butt-joined segments click at the splice; a short fade costs 5 ms of the
    # note and removes the discontinuity at both ends.
    fade = min(fade_samples, len(segment) // 2)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade)
        segment[:fade] *= ramp
        segment[-fade:] *= ramp[::-1]

    return segment


def reconstruct(
    df_target,
    df_source,
    features,
    target_length,
    seed,
    n_candidates=10,
    max_pitch_deviation=0,
    fade_samples=220,
):
    """Rebuild the target from source frames.

    Returns (audio, report). `report` records what was actually placed so the
    result can be described rather than just listened to.
    """
    source_scaled, target_scaled = standardize(df_source, df_target, features)
    rng = np.random.default_rng(seed)

    generated = np.zeros(target_length)
    loaded = {}
    placements = []

    for position in range(len(df_target)):
        target_row = df_target.iloc[position]
        source_row, deviation = select_source_frame(
            position,
            target_scaled,
            source_scaled,
            df_source,
            float(target_row["mean_pitch"]),
            rng,
            n_candidates=n_candidates,
            max_pitch_deviation=max_pitch_deviation,
        )

        path = source_row["path"]
        if path not in loaded:
            loaded[path] = load_audio(path)

        start = int(target_row["start_sample"])
        wanted = int(target_row["end_sample"]) - start
        segment = render_frame(loaded[path], source_row, wanted, fade_samples=fade_samples)
        generated[start : start + len(segment)] = segment

        placements.append(
            {
                "target_frame": position,
                "freesound_id": source_row["freesound_id"],
                "target_pitch": float(target_row["mean_pitch"]),
                "source_pitch": float(source_row["mean_pitch"]),
                "pitch_deviation": deviation,
                "requested_samples": wanted,
                "placed_samples": len(segment),
            }
        )

    filled = sum(p["placed_samples"] for p in placements)
    report = {
        "placements": placements,
        "frames_placed": len(placements),
        "mean_abs_pitch_deviation": float(
            np.mean([p["pitch_deviation"] for p in placements])
        ),
        "max_abs_pitch_deviation": float(
            np.max([p["pitch_deviation"] for p in placements])
        ),
        "coverage": filled / target_length,
        "truncated_frames": sum(
            1 for p in placements if p["placed_samples"] < p["requested_samples"]
        ),
    }
    return generated, report
