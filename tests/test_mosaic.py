"""Checks for the mosaicing logic in src/mosaic.py.

Run with `python tests/test_mosaic.py`. Everything is synthesised at runtime,
so no Freesound key, no downloads and no committed audio are needed.

The fixture is a melody of tones separated by rests. Rests are the point: the
bug this guards against framed a note as onset -> next onset, which swallowed
the rest that followed it.
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import essentia
import essentia.standard as estd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mosaic

FS = mosaic.SAMPLE_RATE
NOTE_SECONDS = 0.9
REST_SECONDS = 0.25

failures = []


def check(name, condition, detail=""):
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
    if not condition:
        failures.append(name)


def tone(midi, seconds, rng, level=1.0):
    """A vibrato'd harmonic tone. Melodia ignores pure sine waves.

    `level` stands in for how unevenly Freesound recordings are mastered.
    """
    n = int(seconds * FS)
    t = np.arange(n) / FS
    freq = 440 * 2 ** ((midi - 69) / 12) * (1 + 0.006 * np.sin(2 * np.pi * 5.5 * t))
    phase = 2 * np.pi * np.cumsum(freq) / FS
    envelope = np.minimum(1, np.minimum(t / 0.03, (seconds - t) / 0.08))
    partials = sum((0.7 ** k) * np.sin((k + 1) * phase) for k in range(6))
    return level * (0.4 * envelope * partials + 0.002 * rng.standard_normal(n))


def write(path, samples):
    estd.MonoWriter(filename=path, format="wav", sampleRate=FS)(
        essentia.array(np.asarray(samples, dtype=np.float32))
    )


def write_melody(path, midi_notes, rng):
    rest = np.zeros(int(REST_SECONDS * FS))
    parts = []
    for midi in midi_notes:
        parts += [tone(midi, NOTE_SECONDS, rng), rest]
    write(path, np.concatenate(parts))


def test_loudness_and_mfcc_are_length_invariant():
    print("\nFeatures are comparable across notes of different length")
    rng = np.random.default_rng(0)
    short = tone(69, 0.5, rng)
    long = tone(69, 4.0, rng)

    ratio = mosaic._loudness(long) / mosaic._loudness(short)
    check("loudness within 10% across an 8x length change", abs(ratio - 1) < 0.10,
          f"ratio={ratio:.3f}")

    a, b = mosaic._mean_mfcc(short)[1:], mosaic._mean_mfcc(long)[1:]
    similarity = float(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))
    check("MFCC shape cosine similarity > 0.99", similarity > 0.99, f"cos={similarity:.4f}")


def test_frames_are_notes_not_note_plus_rest(tmpdir):
    print("\nFrames span the note only, and the last note survives")
    path = os.path.join(tmpdir, "melody.wav")
    write_melody(path, [60, 62, 64, 65, 67], np.random.default_rng(1))

    audio = mosaic.load_audio(path)
    onsets, durations, _, _ = mosaic.segment_notes(audio, min_duration=0.2)
    rows = mosaic.analyze_sound(path, min_duration=0.2)

    check("a frame per segmented note (none dropped)", len(rows) == len(onsets),
          f"{len(rows)} frames vs {len(onsets)} notes")

    lengths = np.array([r["end_sample"] - r["start_sample"] for r in rows]) / FS
    check("frame lengths match the reported durations",
          np.allclose(lengths, durations[: len(rows)], atol=0.01),
          f"frames={lengths.round(3).tolist()}")

    check("no frame is longer than a note plus its rest",
          bool(np.all(lengths < NOTE_SECONDS + REST_SECONDS)),
          f"longest={lengths.max():.3f}s, note+rest={NOTE_SECONDS + REST_SECONDS}s")

    # The old framing would have run each frame into the following silence.
    quietest = min(float(np.sqrt(np.mean(audio[r["start_sample"]:r["end_sample"]] ** 2)))
                   for r in rows)
    check("every frame is voiced throughout (no rest folded in)", quietest > 0.05,
          f"quietest frame RMS={quietest:.3f}")

    return rows


def test_scaling_rebalances_the_real_collection():
    print("\nStandardising restores pitch to the distance (frozen pre-fix violin data)")
    fixtures = os.path.join(os.path.dirname(__file__), "fixtures")
    df_source = pd.read_csv(os.path.join(fixtures, "prefix-analysis_violin_source.csv"),
                            index_col=0)
    df_target = pd.read_csv(os.path.join(fixtures, "prefix-analysis_over_the_rainbow_target.csv"),
                            index_col=0)
    features = mosaic.FEATURE_COLUMNS

    raw = (df_source[features].to_numpy(float) - df_target[features].to_numpy(float)[0]) ** 2
    raw_share = raw.sum(0) / raw.sum()

    source_scaled, target_scaled = mosaic.standardize(df_source, df_target, features)
    scaled = (source_scaled - target_scaled[0]) ** 2
    scaled_share = scaled.sum(0) / scaled.sum()

    pitch, loud = features.index("mean_pitch"), features.index("loudness")
    print(f"      pitch    share of squared distance: {raw_share[pitch]:.3%} -> {scaled_share[pitch]:.3%}")
    print(f"      loudness share of squared distance: {raw_share[loud]:.3%} -> {scaled_share[loud]:.3%}")
    print(f"      mfcc_0   share of squared distance: {raw_share[0 + features.index('mfcc_0')]:.3%} -> {scaled_share[features.index('mfcc_0')]:.3%}")

    check("pitch was negligible on raw columns", raw_share[pitch] < 0.01)
    check("loudness was entirely absent on raw columns", raw_share[loud] < 1e-6)
    check("no feature dominates after scaling", scaled_share.max() < 0.35,
          f"max share={scaled_share.max():.1%} ({features[int(scaled_share.argmax())]})")
    check("pitch is now a meaningful share", scaled_share[pitch] > 0.02,
          f"{scaled_share[pitch]:.1%}")


def test_reconstruction(tmpdir):
    print("\nReconstruction places in-tune, non-overrunning, reproducible frames")
    rng = np.random.default_rng(2)

    target_path = os.path.join(tmpdir, "target.wav")
    write_melody(target_path, [60, 62, 64, 65, 67], rng)
    target_rows = mosaic.analyze_sound(target_path, min_duration=0.2)
    df_target = pd.DataFrame(target_rows)
    target_audio = mosaic.load_audio(target_path)
    target_length = len(target_audio)

    # Build the source collection at the pitches actually detected in the
    # target, so an exact match provably exists for every target frame.
    detected = sorted({int(p) for p in df_target["mean_pitch"]})
    print(f"      target frames: {len(df_target)} at MIDI {detected}")
    source_rows = []
    for index, midi in enumerate(detected * 3):
        path = os.path.join(tmpdir, f"source_{index}.wav")
        # Deliberately shorter than most target frames, to exercise clamping, and at
        # levels an order of magnitude apart, to exercise loudness matching.
        write(path, tone(midi, 0.45, rng, level=[1.0, 0.12, 0.4][index % 3]))
        source_rows += mosaic.analyze_sound(path, min_duration=0.2, audio_id=index)
    df_source = pd.DataFrame(source_rows).reset_index(drop=True)
    check("source collection analysed", len(df_source) > 0, f"{len(df_source)} frames")

    features = mosaic.FEATURE_COLUMNS
    audio, report = mosaic.reconstruct(df_target, df_source, features, target_audio, seed=7)

    check("every target frame was placed", report["frames_placed"] == len(df_target),
          f"{report['frames_placed']}/{len(df_target)}")
    check("every placement is exactly in tune", report["max_abs_pitch_deviation"] == 0,
          f"max deviation={report['max_abs_pitch_deviation']} semitones")

    # Raising the tolerance must not cost tuning where an exact match exists: pitch is
    # settled before the other features get to rank anything.
    tolerant, tolerant_report = mosaic.reconstruct(
        df_target, df_source, features, target_audio, seed=7, max_pitch_deviation=3
    )
    check("a wider tolerance is not spent where exact matches exist",
          tolerant_report["max_abs_pitch_deviation"] == 0,
          f"max deviation={tolerant_report['max_abs_pitch_deviation']} semitones at tolerance 3")
    check("output is the target's length", len(audio) == target_length)
    check("output is not silent", float(np.abs(audio).max()) > 0.01)

    # Clamping: a placement may be shorter than requested, never longer, and
    # never longer than the source frame it came from.
    overruns = []
    for placement in report["placements"]:
        source = df_source[df_source["freesound_id"] == placement["freesound_id"]]
        available = int((source["end_sample"] - source["start_sample"]).max())
        if placement["placed_samples"] > min(placement["requested_samples"], available):
            overruns.append(placement)
    check("no placement reads past its source frame", not overruns, f"{len(overruns)} overruns")
    check("clamping actually exercised", report["truncated_frames"] > 0,
          f"{report['truncated_frames']} frames truncated to the source note's length")

    # No clicks: with the fade, every placement starts and ends at silence, so
    # the step across a splice is far smaller than the tone's own slope. The
    # comparison against fade_samples=0 shows the fade is what achieves it.
    def worst_boundary_step(rendered):
        steps = []
        for placement in report["placements"]:
            start = int(df_target.iloc[placement["target_frame"]]["start_sample"])
            end = start + placement["placed_samples"]
            steps += [abs(float(rendered[start])), abs(float(rendered[end - 1]))]
        return max(steps)

    faded = worst_boundary_step(audio)
    butt_joined, _ = mosaic.reconstruct(
        df_target, df_source, features, target_audio, seed=7, fade_samples=0
    )
    unfaded = worst_boundary_step(butt_joined)
    check("fades remove the splice discontinuity", faded < 0.01,
          f"boundary step {unfaded:.4f} butt-joined -> {faded:.6f} faded")

    again, _ = mosaic.reconstruct(df_target, df_source, features, target_audio, seed=7)
    check("same seed reproduces the audio exactly", np.array_equal(audio, again))
    differ, _ = mosaic.reconstruct(df_target, df_source, features, target_audio, seed=8)
    check("a different seed varies the result", not np.array_equal(audio, differ))

    # Loudness matching: source recordings arrive at very different levels, and each
    # placed segment should end up at the level of the note it replaces.
    _, unmatched = mosaic.reconstruct(df_target, df_source, features, target_audio, seed=7,
                                      normalize_loudness=False)
    raw_spread = unmatched["rms_spread"]
    matched_spread = report["rms_spread"]
    check("loudness matching tightens the level spread", matched_spread < raw_spread,
          f"{raw_spread:.1f}x unmatched -> {matched_spread:.1f}x matched")
    check("matched segments sit near the target note's level", matched_spread < 1.5,
          f"spread={matched_spread:.2f}x")
    check("no frame was too quiet to be usable", report["level_limited_frames"] == 0,
          f"{report['level_limited_frames']} level-limited")
    check("nothing clips", float(np.abs(audio).max()) <= 1.0,
          f"peak={np.abs(audio).max():.3f}")

    print(f"      coverage={report['coverage']:.1%}  "
          f"mean|deviation|={report['mean_abs_pitch_deviation']} semitones")
    return df_target, df_source


def test_failures_are_loud(df_target, df_source):
    print("\nImpossible requests raise instead of producing silence")
    features = mosaic.FEATURE_COLUMNS

    shifted = df_target.copy()
    shifted["mean_pitch"] = shifted["mean_pitch"] + 40  # nothing in the source is this high
    try:
        mosaic.reconstruct(shifted, df_source, features, np.zeros(44100), seed=0)
        check("unmatchable pitch raises", False, "returned silently")
    except ValueError as error:
        check("unmatchable pitch raises", "No source frame within" in str(error))

    constant = df_source.copy()
    constant["loudness"] = 1.0
    try:
        mosaic.standardize(constant, df_target, ["loudness", "mean_pitch"])
        check("constant feature raises", False, "returned silently")
    except ValueError as error:
        check("constant feature raises", "constant" in str(error))


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        test_loudness_and_mfcc_are_length_invariant()
        test_frames_are_notes_not_note_plus_rest(tmpdir)
        test_scaling_rebalances_the_real_collection()
        df_target, df_source = test_reconstruction(tmpdir)
        test_failures_are_loud(df_target, df_source)

    print()
    if failures:
        print(f"{len(failures)} FAILED: {failures}")
        sys.exit(1)
    print("All checks passed.")
