"""Extract one row of features per note, for a source collection and/or a target file.

    python src/analyze.py --collection barnyard
    python src/analyze.py --target over_the_rainbow

The analysis itself lives in mosaic.py; this script configures it, saves the results and
writes plots. Run `python tests/test_mosaic.py` to check the analysis behaves as described.
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mosaic
import paths

# The shortest note we want to resolve, in seconds. Derived from the target's tempo and the
# shortest note value in its melody: at 90 bpm an eighth note is 60/90 * 1/2 = 0.33 s. Set
# just below that to absorb tempo variation. Over-detecting onsets is preferable to
# under-detecting: a spare onset splits one melody note across two source frames, while a
# missing onset leaves silence.
DEFAULT_MIN_DURATION = 0.3

# A source frame is only usable as a melody note if it holds one steady pitch. The default
# admits a frame whose contour stays within about a quarter tone of its own median.
DEFAULT_MAX_PITCH_DRIFT = 50.0
DEFAULT_MIN_FRAME_SECONDS = 0.2


def analyze_collection(collection, min_duration, max_pitch_drift, min_frame_seconds):
    out_path = paths.collection_frames(collection)
    df = pd.read_csv(paths.collection_csv(collection), index_col=0)
    rows, skipped_ids = mosaic.analyze_collection(df, min_duration=min_duration)

    df_source, rejected = mosaic.filter_frames(
        pd.DataFrame(rows), max_pitch_drift, min_frame_seconds
    )
    print(f"\nKept {len(df_source)}/{len(rows)} frames from "
          f"{df_source['freesound_id'].nunique()} sounds; dropped "
          f"{rejected['unsteady_pitch']} for unsteady pitch and "
          f"{rejected['too_short']} for being too short.")

    df_source.to_csv(out_path)
    print(f"Saved source DataFrame with {len(df_source)} entries! {out_path}")

    if skipped_ids:
        print(f"\n{len(skipped_ids)} sounds yielded no melodic contour at all.")

    durations = (df_source["end_sample"] - df_source["start_sample"]) / mosaic.SAMPLE_RATE
    print(
        f"\nframe duration: min {durations.min():.2f}s  "
        f"median {durations.median():.2f}s  max {durations.max():.2f}s"
    )
    pitches = df_source["mean_pitch"]
    print(f"pitch coverage: MIDI {pitches.min():.0f}-{pitches.max():.0f}, "
          f"{pitches.nunique()} distinct pitches")

    plt.figure(figsize=(15, 3))
    plt.hist(pitches, bins=np.arange(pitches.min(), pitches.max() + 2) - 0.5)
    plt.xlabel("MIDI pitch")
    plt.ylabel("source frames")
    plt.title(f'Pitch coverage of the "{collection}" collection')
    plot_path = paths.ensure_parent(paths.plot(f"{collection}_pitch_coverage"))
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Wrote {plot_path}")


def analyze_target(name, min_duration):
    out_path = paths.target_notes(name)
    path = paths.target_audio(name)
    audio = mosaic.load_audio(path)
    onsets, _, _, pitch_values, _ = mosaic.segment_notes(audio, min_duration=min_duration)

    rows = mosaic.analyze_sound(path, min_duration=min_duration)
    df_target = pd.DataFrame(rows)
    df_target.to_csv(out_path)
    print(f"Saved target DataFrame with {len(df_target)} entries! {out_path}")

    covered = (df_target["end_sample"] - df_target["start_sample"]).sum() / len(audio)
    print(f"{len(df_target)} notes detected, covering {covered:.0%} of the excerpt")
    print(f"pitches: MIDI {sorted(int(p) for p in df_target['mean_pitch'])}")

    # The contour plotted here is the one the saved frames came from, so the figure and the
    # data describe the same analysis.
    pitch_times = np.arange(len(pitch_values)) * mosaic.MELODIA_HOP_SIZE / mosaic.SAMPLE_RATE
    envelope = np.abs(audio[:: mosaic.MELODIA_HOP_SIZE])
    n = min(len(pitch_times), len(envelope))

    f, axarr = plt.subplots(2, sharex=True, figsize=(15, 5))
    axarr[0].plot(pitch_times[:n], pitch_values[:n])
    axarr[0].set_title("estimated pitch [Hz]")
    axarr[1].plot(pitch_times[:n], envelope[:n])
    axarr[1].set_title("waveform envelope")
    axarr[1].set_xlabel("time [s]")
    plot_path = paths.ensure_parent(paths.plot(f"{name}_pitch_contour"))
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Wrote {plot_path}")

    # Shade each detected note. The gaps between the shaded spans are the rests: a note ends
    # at onset + duration, not at the following onset.
    plt.figure(figsize=(15, 5))
    plt.plot(np.arange(len(audio)) / mosaic.SAMPLE_RATE, audio, linewidth=0.5)
    for _, row in df_target.iterrows():
        plt.axvspan(
            row["start_sample"] / mosaic.SAMPLE_RATE,
            row["end_sample"] / mosaic.SAMPLE_RATE,
            color="red",
            alpha=0.2,
        )
    plt.axis([0, len(audio) / mosaic.SAMPLE_RATE, -1, 1])
    plt.xlabel("time [s]")
    plt.title(f"{name}: detected notes shaded ({len(onsets)} onsets)")
    plot_path = paths.ensure_parent(paths.plot(f"{name}_notes"))
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Wrote {plot_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collection", help="name of a downloaded collection to analyze")
    parser.add_argument("--target", help="name of a target prepared by prepare_target.sh")
    parser.add_argument("--min-duration", type=float, default=DEFAULT_MIN_DURATION,
                        help=f"shortest note to resolve, seconds (default {DEFAULT_MIN_DURATION})")
    parser.add_argument("--max-pitch-drift", type=float, default=DEFAULT_MAX_PITCH_DRIFT,
                        help="drop source frames whose pitch wanders more than this many "
                             f"cents (default {DEFAULT_MAX_PITCH_DRIFT})")
    parser.add_argument("--min-frame-seconds", type=float, default=DEFAULT_MIN_FRAME_SECONDS,
                        help=f"drop source frames shorter than this "
                             f"(default {DEFAULT_MIN_FRAME_SECONDS})")
    args = parser.parse_args()

    if not args.collection and not args.target:
        parser.error("give --collection, --target, or both")

    if args.collection:
        analyze_collection(args.collection, args.min_duration,
                           args.max_pitch_drift, args.min_frame_seconds)
    if args.target:
        analyze_target(args.target, args.min_duration)


if __name__ == "__main__":
    main()
