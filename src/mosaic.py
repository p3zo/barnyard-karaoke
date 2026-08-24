"""Analysis and frame-selection logic for the audio mosaicing pipeline.

The notebooks handle orchestration and plotting; the logic that decides what a
frame is, how it is described, and which source frame replaces it lives here so
it can be tested (see tests/test_mosaic.py).
"""

import collections
import os
import subprocess
import tempfile

import numpy as np
import essentia
import essentia.standard as estd

SAMPLE_RATE = 44100

# Melodia's defaults. The pitch contour is sampled every MELODIA_HOP_SIZE
# samples (2.9 ms), which is the time resolution available to the segmenter.
MELODIA_FRAME_SIZE = 2048
MELODIA_HOP_SIZE = 128

# MFCCs are averaged over fixed-size windows so that notes of different lengths
# stay comparable. One FFT across a whole note makes the coefficients a function
# of its length: a 440 Hz tone gives mfcc_1 = 200 over 2048 samples and 81 over
# 88200. Averaging fixed windows holds that to within 1%.
MFCC_FRAME_SIZE = 2048
MFCC_HOP_SIZE = 1024
N_MFCC = 13

# Essentia's Loudness is Stevens' law: energy ** LOUDNESS_EXPONENT.
LOUDNESS_EXPONENT = 0.67

# How a source note that is shorter than the target note gets to fill it.
#   truncate    play it once and leave the remainder silent
#   longest     pick the longest frame at the right pitch, then truncate
#   concatenate run consecutive frames at the right pitch together
#   loop        repeat the one frame until the note is filled
#   stretch     slow the one frame down to the note's length
FILL_STRATEGIES = ("truncate", "longest", "concatenate", "loop", "stretch")

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


def _loudness_to_rms(loudness):
    """Invert _loudness: it stores rms ** (2 * LOUDNESS_EXPONENT)."""
    return np.asarray(loudness, dtype=float) ** (1.0 / (2 * LOUDNESS_EXPONENT))


def _contour_quality(pitch_values, pitch_confidence, onset, duration):
    """How well the pitch tracker actually held on to this note.

    A frame only reads as a musical note if it is voiced throughout and sits at
    a steady pitch. Animal calls often are not: a whinny slides across an
    octave, a bark has no pitch at all.
    """
    first = int(round(onset * SAMPLE_RATE / MELODIA_HOP_SIZE))
    last = int(round((onset + duration) * SAMPLE_RATE / MELODIA_HOP_SIZE))
    contour = np.asarray(pitch_values[first:last], dtype=float)
    confidence = np.asarray(pitch_confidence[first:last], dtype=float)
    if contour.size == 0:
        return {"voiced_fraction": 0.0, "pitch_drift_cents": np.inf, "pitch_confidence": 0.0}

    voiced = contour > 0
    if voiced.sum() < 2:
        drift = np.inf
    else:
        cents = 1200 * np.log2(contour[voiced] / np.median(contour[voiced]))
        drift = float(np.std(cents))

    return {
        "voiced_fraction": float(voiced.mean()),
        "pitch_drift_cents": drift,
        "pitch_confidence": float(np.mean(confidence)) if confidence.size else 0.0,
    }


def _loudness(frame):
    """Length-invariant loudness.

    Essentia's Loudness is Stevens' law, energy**0.67, so it grows with frame
    length; dividing by len(frame)**0.67 (rather than len(frame)) makes it the
    0.67 power of mean energy and comparable across notes of different length.
    """
    return estd.Loudness()(essentia.array(frame)) / len(frame) ** LOUDNESS_EXPONENT


def segment_notes(audio, min_duration, pitch_distance_threshold=30, rms_threshold=-4):
    """Estimate the predominant melody and segment it into notes.

    Returns (onsets, durations, midi_pitches, pitch_values, pitch_confidence),
    all in seconds except the MIDI pitches and the two per-frame contours.
    """
    pitch_values, pitch_confidence = estd.PredominantPitchMelodia(
        frameSize=MELODIA_FRAME_SIZE, hopSize=MELODIA_HOP_SIZE
    )(audio)

    onsets, durations, midi_pitches = estd.PitchContourSegmentation(
        hopSize=MELODIA_HOP_SIZE,
        minDuration=min_duration,
        pitchDistanceThreshold=pitch_distance_threshold,
        rmsThreshold=rms_threshold,
    )(pitch_values, audio)

    return onsets, durations, midi_pitches, pitch_values, pitch_confidence


def analyze_sound(audio_path, min_duration, audio_id=None, **segmentation_kwargs):
    """Describe every note in a sound as a row of features.

    A note spans onset -> onset + duration, so it covers the note itself and
    not the rest that follows it.
    """
    audio = load_audio(audio_path)
    onsets, durations, midi_pitches, pitch_values, pitch_confidence = segment_notes(
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
        row.update(_contour_quality(pitch_values, pitch_confidence, onset, duration))
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


def filter_frames(df, max_pitch_drift, min_seconds):
    """Keep only frames that can stand in for a melody note.

    A frame is usable if it holds one steady pitch: `pitch_drift_cents` is the
    standard deviation of its contour about its own median, so a bleat that
    warbles or a whinny that slides across an octave scores high and is cut.
    Frames too short to carry a note are cut as well.

    Returns (kept, rejected).
    """
    seconds = (df["end_sample"] - df["start_sample"]) / SAMPLE_RATE
    steady = df["pitch_drift_cents"] <= max_pitch_drift
    long_enough = seconds >= min_seconds
    keep = steady & long_enough

    rejected = {
        "unsteady_pitch": int((~steady).sum()),
        "too_short": int((steady & ~long_enough).sum()),
    }
    return df[keep].reset_index(drop=True), rejected


def standardize(df_source, df_target, features):
    """Put features on a common scale, fitted on the source collection.

    The raw columns differ by orders of magnitude -- mfcc_0 spans ~700 units,
    mean_pitch ~50, loudness ~0.006 -- so an unscaled distance is almost
    entirely mfcc_0 and not at all loudness.
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
    target_row,
    rng,
    n_candidates=10,
    max_pitch_deviation=0,
    max_gain=10.0,
    prefer="similar",
    avoid_sounds=(),
    octave_folding=True,
):
    """Pick a source frame to stand in for one target frame.

    Pitch is settled first: candidates are the frames at the smallest pitch
    deviation the collection can offer, provided that is within
    `max_pitch_deviation`. Only among those does the rest of the feature
    vector rank them, and one of the closest `n_candidates` is picked at
    random.

    Pitch has to win outright rather than compete on distance. It is one
    feature against thirteen MFCCs, so ranking the whole pool at once buys a
    closer timbre at the cost of a semitone even where an exact match exists.

    Candidates that would need more than `max_gain` to reach the target note's
    level are then dropped. Freesound recordings span a ~350x range in level,
    and a frame recorded 40 dB down cannot be raised to sit with the others
    without dragging its noise floor up with it. Every note in the barnyard
    collection has a same-pitch frame needing at most 1.5x, so this costs
    very little choice.

    With `octave_folding`, a frame counts as matching when its pitch class
    matches, whatever octave it sits in, and frames nearer the melody's own
    octave rank higher. A collection of animal calls covers few pitches steadily
    but many pitch classes, and displacing a note by an octave keeps it
    consonant where settling for a semitone would not.

    `avoid_sounds` holds recordings to pass over -- ones just used, which would
    otherwise be heard as the same animal twice in a row. It is a preference,
    not a rule: if nothing else is available the pool is used as it stands.

    Returns (row, pitch_deviation, octave_shift, level_limited), where
    `level_limited` says no candidate at that pitch was loud enough and the
    loudest was taken.
    """
    target_pitch = float(target_row["mean_pitch"])
    signed = df_source["mean_pitch"].to_numpy(dtype=float) - target_pitch

    if octave_folding:
        deviations = np.abs(((signed + 6) % 12) - 6)
        octave_shifts = np.round(signed / 12)
    else:
        deviations = np.abs(signed)
        octave_shifts = np.zeros_like(signed)


    best = deviations.min()
    if not np.isfinite(best) or best > max_pitch_deviation:
        raise ValueError(
            f"No source frame within {max_pitch_deviation} semitones of MIDI "
            f"{target_pitch} (closest is {best:.0f} away). Widen "
            f"max_pitch_deviation or add source material in that register."
        )
    eligible = np.flatnonzero(deviations == best)

    # Skip recordings just used, so long as that leaves something to choose from.
    if len(avoid_sounds):
        ids = df_source["freesound_id"].to_numpy()
        fresh = eligible[~np.isin(ids[eligible], list(avoid_sounds))]
        if fresh.size:
            eligible = fresh

    target_rms = _loudness_to_rms(target_row["loudness"])
    source_rms = _loudness_to_rms(df_source["loudness"].to_numpy(dtype=float)[eligible])
    with np.errstate(divide="ignore", invalid="ignore"):
        needed_gain = np.where(source_rms > 0, target_rms / source_rms, np.inf)

    level_limited = False
    loud_enough = eligible[needed_gain <= max_gain]
    if loud_enough.size:
        eligible = loud_enough
    else:
        level_limited = True
        eligible = eligible[[int(np.argmin(needed_gain))]]

    if prefer == "longest":
        lengths = (df_source["end_sample"].to_numpy() - df_source["start_sample"].to_numpy())
        chosen = int(eligible[np.argmax(lengths[eligible])])
    else:
        # Register is a preference rather than a filter: restricting to the single
        # nearest octave leaves some notes with one candidate and no choice at all,
        # so the same frame lands under every occurrence of that pitch.
        distances = np.linalg.norm(
            source_scaled[eligible] - target_scaled[target_index], axis=1
        )
        order = np.lexsort((distances, np.abs(octave_shifts[eligible])))
        closest = eligible[order[:n_candidates]]
        chosen = int(rng.choice(closest))

    return (df_source.iloc[chosen], float(deviations[chosen]),
            int(octave_shifts[chosen]), level_limited)


def render_frame(source_audio, source_row, n_samples, fade_samples=220):
    """Cut up to `n_samples` from a source frame, stopping at the frame's end.

    Source notes are often shorter than the target note they fill, and reading
    on past the end would pull in whatever follows in the source recording.
    """
    start = int(source_row["start_sample"])
    end = int(source_row["end_sample"])
    segment = np.array(source_audio[start : min(end, start + n_samples)], dtype=np.float64)
    return apply_fades(segment, fade_samples)


def apply_fades(segment, fade_samples):
    """Ramp both ends to zero so a splice does not click."""
    segment = np.array(segment, dtype=np.float64)
    fade = min(fade_samples, len(segment) // 2)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade)
        segment[:fade] *= ramp
        segment[-fade:] *= ramp[::-1]
    return segment


def _atempo_chain(rate):
    """Decompose a tempo change into factors ffmpeg's atempo accepts (0.5-2.0)."""
    factors = []
    while rate < 0.5:
        factors.append(0.5)
        rate /= 0.5
    while rate > 2.0:
        factors.append(2.0)
        rate /= 2.0
    factors.append(rate)
    return factors


def time_stretch(segment, n_samples):
    """Stretch `segment` to `n_samples` without moving its pitch.

    Uses ffmpeg's atempo, a WSOLA implementation, rather than a hand-rolled
    one, so the comparison against the other fill strategies is not skewed by
    the quality of the stretcher.
    """
    if len(segment) == 0 or n_samples <= 0 or len(segment) == n_samples:
        return segment[:n_samples]

    rate = len(segment) / n_samples
    chain = ",".join(f"atempo={f:.6f}" for f in _atempo_chain(rate))

    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "in.wav")
        out = os.path.join(tmp, "out.wav")
        estd.MonoWriter(filename=raw, format="wav", sampleRate=SAMPLE_RATE)(
            essentia.array(segment.astype(np.float32))
        )
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-filter:a", chain, out],
            check=True,
        )
        stretched = np.array(load_audio(out), dtype=np.float64)

    # atempo lands within a few samples of the requested length.
    if len(stretched) < n_samples:
        return np.pad(stretched, (0, n_samples - len(stretched)))
    return stretched[:n_samples]


def loop_to_length(segment, n_samples, fade_samples):
    """Repeat `segment` until it fills `n_samples`, fading each repeat."""
    if len(segment) == 0:
        return segment
    faded = apply_fades(segment, fade_samples)
    repeats = int(np.ceil(n_samples / len(faded)))
    return np.tile(faded, repeats)[:n_samples]


def _rms(samples):
    return float(np.sqrt(np.mean(np.square(samples)))) if len(samples) else 0.0


def match_loudness(segment, target_note, max_gain=10.0):
    """Scale `segment` to sit at the same RMS as the target note it replaces.

    Freesound recordings arrive at wildly different levels, and a close-mic'd
    bark would otherwise sit ten times louder than a distant moo. Matching the
    target note also carries the original melody's dynamics across.

    The gain is capped, since a near-silent segment would otherwise be
    amplified into whatever noise it contains, and further limited so the
    result cannot clip.
    """
    source_rms = _rms(segment)
    target_rms = _rms(target_note)
    if source_rms == 0.0 or target_rms == 0.0:
        return segment, 1.0

    gain = min(target_rms / source_rms, max_gain)

    peak = float(np.abs(segment).max())
    if peak * gain > 1.0:
        gain = 1.0 / peak

    return segment * gain, gain


def reconstruct(
    df_target,
    df_source,
    features,
    target_audio,
    seed,
    n_candidates=10,
    max_pitch_deviation=0,
    fade_samples=220,
    normalize_loudness=True,
    max_gain=10.0,
    fill="truncate",
    octave_folding=True,
):
    """Rebuild the target from source frames.

    `fill` decides what happens when the chosen source note is shorter than the
    target note; see FILL_STRATEGIES.

    Returns (audio, report). `report` records what was actually placed so the
    result can be described rather than just listened to.
    """
    if fill not in FILL_STRATEGIES:
        raise ValueError(f"Unknown fill strategy {fill!r}; expected one of {FILL_STRATEGIES}")
    source_scaled, target_scaled = standardize(df_source, df_target, features)
    rng = np.random.default_rng(seed)

    target_audio = np.asarray(target_audio, dtype=np.float64)
    generated = np.zeros(len(target_audio))
    loaded = {}
    placements = []
    all_used_ids = set()
    # Enough history that a recording is not heard again while it is still fresh.
    recent_sounds = collections.deque(maxlen=3)

    for position in range(len(df_target)):
        target_row = df_target.iloc[position]
        start = int(target_row["start_sample"])
        wanted = int(target_row["end_sample"]) - start

        def pick(also_avoid=()):
            return select_source_frame(
                position,
                target_scaled,
                source_scaled,
                df_source,
                target_row,
                rng,
                n_candidates=n_candidates,
                max_pitch_deviation=max_pitch_deviation,
                max_gain=max_gain,
                prefer="longest" if fill == "longest" else "similar",
                avoid_sounds=tuple(recent_sounds) + tuple(also_avoid),
                octave_folding=octave_folding,
            )

        def cut(row, n):
            if row["path"] not in loaded:
                loaded[row["path"]] = load_audio(row["path"])
            return render_frame(loaded[row["path"]], row, n, fade_samples=fade_samples)

        source_row, deviation, octave_shift, level_limited = pick()
        used_ids = [source_row["freesound_id"]]

        if fill == "concatenate":
            # Keep appending different frames at this pitch until the note is covered.
            # Picking at random rather than longest-first is what makes the run of
            # animals differ from note to note; always taking the longest would put the
            # same one under every occurrence of a pitch.
            segment = cut(source_row, wanted)
            # Avoid by recording, not by frame: one recording of three barks would
            # otherwise supply all three pieces and sound like a single stutter.
            in_this_note = [source_row["freesound_id"]]
            while len(segment) < wanted:
                try:
                    nxt, _, _, _ = pick(also_avoid=tuple(in_this_note))
                except ValueError:
                    break
                in_this_note.append(nxt["freesound_id"])
                piece = cut(nxt, wanted - len(segment))
                if len(piece) == 0:
                    break
                segment = np.concatenate([segment, piece])
                used_ids.append(nxt["freesound_id"])
        elif fill == "loop":
            segment = loop_to_length(cut(source_row, wanted), wanted, fade_samples)
        elif fill == "stretch":
            segment = apply_fades(
                time_stretch(cut(source_row, wanted), wanted), fade_samples
            )
        else:
            segment = cut(source_row, wanted)

        gain = 1.0
        if normalize_loudness:
            segment, gain = match_loudness(
                segment, target_audio[start : start + len(segment)], max_gain=max_gain
            )

        generated[start : start + len(segment)] = segment

        placements.append(
            {
                "target_frame": position,
                "freesound_id": source_row["freesound_id"],
                "sounds_used": len(used_ids),
                "target_pitch": float(target_row["mean_pitch"]),
                "source_pitch": float(source_row["mean_pitch"]),
                "pitch_deviation": deviation,
                "octave_shift": octave_shift,
                "requested_samples": wanted,
                "placed_samples": len(segment),
                "gain": gain,
                "rms": _rms(segment),
                "level_limited": level_limited,
            }
        )
        all_used_ids.update(used_ids)
        recent_sounds.extend(used_ids)

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
        "coverage": filled / len(target_audio),
        # The share of the target's *note* time that got filled. The excerpt is part
        # rests, so this is the number that says how sparse the result is.
        "note_coverage": filled / sum(p["requested_samples"] for p in placements),
        "truncated_frames": sum(
            1 for p in placements if p["placed_samples"] < p["requested_samples"]
        ),
        # How far apart the placed segments sit in level.
        "rms_spread": (
            max(p["rms"] for p in placements) / min(p["rms"] for p in placements)
            if placements and min(p["rms"] for p in placements) > 0
            else float("inf")
        ),
        "gain_applied": [p["gain"] for p in placements],
        "level_limited_frames": sum(1 for p in placements if p["level_limited"]),
        "octave_shifted_frames": sum(1 for p in placements if p["octave_shift"] != 0),
        "fill": fill,
        "freesound_ids_used": sorted(all_used_ids),
    }
    return generated, report
