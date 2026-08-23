"""Rebuild a target from source collection frames and write a demo mix.

    python src/reconstruct.py --collection barnyard --target over_the_rainbow

Reports what was actually placed: how much of the target got covered, how far each chosen
frame was from the target note's pitch, and how many segments had to be cut short. The
selection and rendering logic lives in mosaic.py.
"""

import argparse
import os
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import essentia
import essentia.standard as estd

import mosaic

# The three feature sets compared in the paper. They are standardised before the distance is
# computed (see mosaic.standardize), without which mfcc_0 alone accounts for over half of it
# and loudness for none.
FEATURE_SETS = {
    "pitch": ["mean_pitch"],
    "pitch-loudness": ["mean_pitch", "loudness"],
    "pitch-loudness-mfcc": ["mean_pitch", "loudness"] + mosaic.MFCC_FEATURES,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--features", choices=sorted(FEATURE_SETS),
                        default="pitch-loudness-mfcc")
    parser.add_argument("--seed", type=int, default=0,
                        help="seeded so a demo can be regenerated (default 0)")
    parser.add_argument("--max-pitch-deviation", type=float, default=0,
                        help="how far, in semitones, to settle for when the collection has "
                             "nothing at the target note's pitch; 0 means refuse rather than "
                             "go out of tune. An exact match is always preferred when one exists")
    parser.add_argument("--n-candidates", type=int, default=10,
                        help="frames at the best available pitch to pick from at random")
    parser.add_argument("--no-normalize-loudness", action="store_true",
                        help="place segments at their recorded level; without this each "
                             "segment is scaled to the RMS of the note it replaces, since "
                             "Freesound recordings arrive at wildly different levels")
    parser.add_argument("--target-gain", type=float, default=0.15,
                        help="how much of the original to leave under the demo mix")
    args = parser.parse_args()

    df = pd.read_csv(f"dataframe_{args.collection}.csv", index_col=0)
    df_source = pd.read_csv(f"dataframe_{args.collection}_source.csv", index_col=0)
    df_target = pd.read_csv(f"dataframe_{args.target}_target.csv", index_col=0)

    target_path = df_target.iloc[0]["path"]
    target_audio = mosaic.load_audio(target_path)

    generated_audio, report = mosaic.reconstruct(
        df_target,
        df_source,
        FEATURE_SETS[args.features],
        target_audio,
        seed=args.seed,
        n_candidates=args.n_candidates,
        max_pitch_deviation=args.max_pitch_deviation,
        normalize_loudness=not args.no_normalize_loudness,
    )

    print(f"features:                 {args.features}")
    print(f"frames placed:            {report['frames_placed']}/{len(df_target)}")
    print(f"coverage of the target:   {report['coverage']:.1%}")
    print(f"mean |pitch deviation|:   {report['mean_abs_pitch_deviation']:.2f} semitones")
    print(f"max  |pitch deviation|:   {report['max_abs_pitch_deviation']:.2f} semitones")
    print(f"frames cut short:         {report['truncated_frames']} "
          f"(source note shorter than the target note)")
    print(f"loudest/quietest segment: {report['rms_spread']:.1f}x")
    print(f"too quiet to reach level:  {report['level_limited_frames']} "
          f"(no frame at that pitch loud enough; used the loudest)")
    print()
    print(pd.DataFrame(report["placements"]).to_string())

    reconstructed_path = f"{target_path}.{args.features}.reconstructed.wav"
    estd.MonoWriter(filename=reconstructed_path, format="wav", sampleRate=mosaic.SAMPLE_RATE)(
        essentia.array(generated_audio.astype(np.float32))
    )
    print(f"\nSaved {reconstructed_path}")

    # Credit the sounds actually used. Every CC-BY sound requires attribution, and BY-NC
    # sounds cannot be used commercially.
    used_ids = {p["freesound_id"] for p in report["placements"]}
    used = df[df["freesound_id"].isin(used_ids)]
    if len(used) != len(used_ids):
        raise ValueError(
            f"Collection metadata is missing {len(used_ids) - len(used)} of the "
            f"{len(used_ids)} sounds used, so the result cannot be attributed. Check that "
            f"dataframe_{args.collection}.csv matches the analysed collection."
        )
    print(f"\nBuilt from {len(used)} sounds:")
    print(used[["name", "username", "license", "freesound_id"]].to_string())

    f, axarr = plt.subplots(2, sharex=True, sharey=True, figsize=(15, 6))
    axarr[0].plot(np.arange(len(target_audio)) / mosaic.SAMPLE_RATE, target_audio, linewidth=0.5)
    axarr[0].set_title("Target audio")
    axarr[1].plot(np.arange(len(generated_audio)) / mosaic.SAMPLE_RATE, generated_audio,
                  linewidth=0.5)
    axarr[1].set_title("Reconstructed")
    axarr[1].set_xlabel("time [s]")
    axarr[1].set_ylim(-1, 1)
    plot_path = f"plot_{args.target}_{args.collection}_{args.features}_reconstruction.png"
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nWrote {plot_path}")

    # Demo mix, mono: the reconstruction over a quiet copy of the target. Hard-panning the
    # reconstruction one way and the target the other misrepresents how the result sounds.
    mix = (generated_audio * 0.7 + target_audio * args.target_gain).astype(np.float32)
    peak = np.abs(mix).max()
    if peak > 1.0:
        mix = mix / peak

    # Encoded with ffmpeg rather than Essentia's AudioWriter, which writes an unplayable
    # file and then segfaults on macOS/arm64 with essentia 2.1b6.dev1389.
    stem = f"{args.target}_{args.collection}_{args.features}"
    demo_wav = f"{stem}_mix.wav"
    demo_mp3 = os.path.join("..", "demo", f"{stem}_mix.mp3")
    estd.MonoWriter(filename=demo_wav, format="wav", sampleRate=mosaic.SAMPLE_RATE)(
        essentia.array(mix)
    )
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", demo_wav, "-b:a", "192k", demo_mp3],
        check=True,
    )
    os.remove(demo_wav)
    print(f"Saved {demo_mp3}")


if __name__ == "__main__":
    main()
