"""Rebuild a target from source collection frames and write a demo mix.

    python src/reconstruct.py --collection barnyard --target over_the_rainbow

Reports what was actually placed: how much of the target got covered, how far each chosen
frame was from the target note's pitch, how evenly the segments sit in level, and how many
had to be cut short. The selection and rendering logic lives in mosaic.py.
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
import paths

# The three feature sets compared in the paper. They are standardised before the distance is
# computed (see mosaic.standardize), so each contributes on comparable terms.
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
    parser.add_argument("--features", choices=sorted(FEATURE_SETS), default="pitch-loudness")
    parser.add_argument("--fill", choices=mosaic.FILL_STRATEGIES, default="truncate",
                        help="what to do when the source note is shorter than the target "
                             "note (default truncate: leave the remainder silent)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seeded so a demo can be regenerated (default 0)")
    parser.add_argument("--max-pitch-deviation", type=float, default=0,
                        help="how far, in semitones, to settle for when the collection has "
                             "nothing at the target note's pitch; 0 means refuse rather than "
                             "go out of tune. An exact match is always preferred")
    parser.add_argument("--n-candidates", type=int, default=10,
                        help="frames at the best available pitch to pick from at random")
    parser.add_argument("--max-pitch-shift", type=float, default=1.0,
                        help="semitones a frame may be moved to land exactly in tune "
                             "(default 1). Frames already in tune are preferred; shifting "
                             "widens a thin pitch class to everything within reach of it. "
                             "0 disables it")
    parser.add_argument("--no-octave-folding", action="store_true",
                        help="require the chosen frame to be in the target note's own "
                             "octave, instead of accepting any octave of its pitch class")
    parser.add_argument("--no-normalize-loudness", action="store_true",
                        help="place segments at their recorded level, instead of scaling "
                             "each to the level of the note it replaces")
    parser.add_argument("--target-gain", type=float, default=0.15,
                        help="how much of the original to leave under the demo mix")
    parser.add_argument("--demo-axis", default="by-fill-strategy",
                        help="subdirectory of demo/<target>_<collection>/ to file the mix "
                             "under, naming whichever setting is being varied")
    parser.add_argument("--demo-name",
                        help="filename for the mix (defaults to the varied setting)")
    args = parser.parse_args()

    df = pd.read_csv(paths.collection_csv(args.collection), index_col=0)
    df_source = pd.read_csv(paths.collection_frames(args.collection), index_col=0)
    df_target = pd.read_csv(paths.target_notes(args.target), index_col=0)

    target_audio = mosaic.load_audio(paths.target_audio(args.target))

    generated_audio, report = mosaic.reconstruct(
        df_target,
        df_source,
        FEATURE_SETS[args.features],
        target_audio,
        seed=args.seed,
        n_candidates=args.n_candidates,
        max_pitch_deviation=args.max_pitch_deviation,
        normalize_loudness=not args.no_normalize_loudness,
        fill=args.fill,
        octave_folding=not args.no_octave_folding,
        max_pitch_shift=args.max_pitch_shift,
    )

    print(f"features:                 {args.features}")
    print(f"fill:                     {args.fill}")
    print(f"frames placed:            {report['frames_placed']}/{len(df_target)}")
    print(f"coverage of the target:   {report['coverage']:.1%} of its duration, "
          f"{report['note_coverage']:.1%} of its note time")
    print(f"mean |pitch deviation|:   {report['mean_abs_pitch_deviation']:.2f} semitones")
    print(f"max  |pitch deviation|:   {report['max_abs_pitch_deviation']:.2f} semitones")
    print(f"loudest/quietest segment: {report['rms_spread']:.1f}x")
    print(f"frames cut short:         {report['truncated_frames']}")
    print(f"too quiet to reach level: {report['level_limited_frames']}")
    print(f"placed in another octave: {report['octave_shifted_frames']}")
    print(f"pitch-shifted into tune:  {report['pitch_shifted_frames']} "
          f"(largest {report['max_pitch_correction']:.0f} semitones)")
    print(f"sounds used:              {len(report['freesound_ids_used'])}")
    print()
    print(pd.DataFrame(report["placements"]).to_string())

    stem = f"{args.target}_{args.collection}_{args.features}_{args.fill}"
    wav_path = paths.ensure_parent(paths.reconstruction(stem))
    estd.MonoWriter(filename=wav_path, format="wav", sampleRate=mosaic.SAMPLE_RATE)(
        essentia.array(generated_audio.astype(np.float32))
    )
    print(f"\nSaved {wav_path}")

    # Credit the sounds actually used. Every CC-BY sound requires attribution, and BY-NC
    # sounds cannot be used commercially.
    used_ids = set(report["freesound_ids_used"])
    used = df[df["freesound_id"].isin(used_ids)]
    if len(used) != len(used_ids):
        raise ValueError(
            f"Collection metadata is missing {len(used_ids) - len(used)} of the "
            f"{len(used_ids)} sounds used, so the result cannot be attributed. Check that "
            f"{paths.collection_csv(args.collection)} matches the analysed collection."
        )
    print(f"\nBuilt from {len(used)} sounds:")
    print(used[["name", "username", "license", "freesound_id"]].to_string())

    f, axarr = plt.subplots(2, sharex=True, sharey=True, figsize=(15, 6))
    axarr[0].plot(np.arange(len(target_audio)) / mosaic.SAMPLE_RATE, target_audio,
                  linewidth=0.5)
    axarr[0].set_title("Target audio")
    axarr[1].plot(np.arange(len(generated_audio)) / mosaic.SAMPLE_RATE, generated_audio,
                  linewidth=0.5)
    axarr[1].set_title(f"Reconstructed ({args.features}, {args.fill})")
    axarr[1].set_xlabel("time [s]")
    axarr[1].set_ylim(-1, 1)
    plot_path = paths.ensure_parent(paths.plot(stem))
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nWrote {plot_path}")

    # Demo mix, mono: the reconstruction over a quiet copy of the target.
    mix = (generated_audio * 0.7 + target_audio * args.target_gain).astype(np.float32)
    peak = np.abs(mix).max()
    if peak > 1.0:
        mix = mix / peak

    default_name = args.fill if args.demo_axis == "by-fill-strategy" else args.features
    demo_path = paths.ensure_parent(
        paths.demo(args.target, args.collection, args.demo_axis,
                   args.demo_name or default_name)
    )
    demo_wav = paths.ensure_parent(os.path.join(paths.OUT, f"{stem}_mix.wav"))
    estd.MonoWriter(filename=demo_wav, format="wav", sampleRate=mosaic.SAMPLE_RATE)(
        essentia.array(mix)
    )
    # Encoded with ffmpeg; Essentia's mp3 AudioWriter writes an unplayable file and then
    # segfaults on macOS/arm64 with essentia 2.1b6.dev1389.
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", demo_wav, "-b:a", "192k", demo_path],
        check=True,
    )
    os.remove(demo_wav)
    print(f"Saved {demo_path}")


if __name__ == "__main__":
    main()
