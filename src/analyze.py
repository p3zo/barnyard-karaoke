"""Extract one row of features per note, for a source collection and/or a target file.

    python src/analyze.py --collection barnyard
    python src/analyze.py --target over_the_rainbow --target-path targets/short_V1bFr2SWP1I.wav

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

# The shortest note we want to resolve, in seconds. Derived from the target's tempo and the
# shortest note value in its melody: at 90 bpm an eighth note is 60/90 * 1/2 = 0.33 s. Set
# just below that to absorb tempo variation. Over-detecting onsets is preferable to
# under-detecting: a spare onset splits one melody note across two source frames, while a
# missing onset leaves silence.
DEFAULT_MIN_DURATION = 0.3


def analyze_collection(collection, min_duration):
    dataframe_path = f"dataframe_{collection}.csv"
    out_path = f"dataframe_{collection}_source.csv"

    df = pd.read_csv(dataframe_path, index_col=0)
    rows, skipped_ids = mosaic.analyze_collection(df, min_duration=min_duration)

    df_source = pd.DataFrame(rows)
    df_source.to_csv(out_path)
    print(f"Saved source DataFrame with {len(df_source)} entries! {out_path}")

    if skipped_ids:
        print("\nSounds that yielded no frames:")
        print(df[df["freesound_id"].isin(skipped_ids)][["name", "tags"]].to_string())

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
    plot_path = f"plot_{collection}_pitch_coverage.png"
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Wrote {plot_path}")


def analyze_target(name, path, min_duration):
    out_path = f"dataframe_{name}_target.csv"

    audio = mosaic.load_audio(path)
    onsets, _, _, pitch_values = mosaic.segment_notes(audio, min_duration=min_duration)

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
    plot_path = f"plot_{name}_pitch_contour.png"
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
    plot_path = f"plot_{name}_notes.png"
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Wrote {plot_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collection", help="name of a downloaded collection to analyze")
    parser.add_argument("--target", help="name to save the target analysis under")
    parser.add_argument("--target-path", help="audio file to analyze as the target")
    parser.add_argument("--min-duration", type=float, default=DEFAULT_MIN_DURATION,
                        help=f"shortest note to resolve, seconds (default {DEFAULT_MIN_DURATION})")
    args = parser.parse_args()

    if not args.collection and not args.target:
        parser.error("give --collection, --target, or both")
    if bool(args.target) != bool(args.target_path):
        parser.error("--target and --target-path go together")

    if args.collection:
        analyze_collection(args.collection, args.min_duration)
    if args.target:
        analyze_target(args.target, args.target_path, args.min_duration)


if __name__ == "__main__":
    main()
