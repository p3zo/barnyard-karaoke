# Barnyard Karaoke

Recreating melodies with animal sounds, via audio mosaicing. See [the paper](paper/main.pdf).

## Setup

```sh
pip install -r requirements.txt
cp .env.template .env    # then fill in your Freesound key from https://freesound.org/apiv2/apply/
```

`prepare_target.sh` additionally needs `yt-dlp` and `ffmpeg` on your PATH. `ffmpeg` is also
used to time-stretch segments and to encode the demo mixes.

## Usage

Run everything from the repo root.

```sh
./prepare_target.sh over_the_rainbow V1bFr2SWP1I 00:01:05 10
python src/download_collection.py --collection barnyard
python src/analyze.py --collection barnyard
python src/analyze.py --target over_the_rainbow
python src/reconstruct.py --collection barnyard --target over_the_rainbow
```

The other target used in the paper is
`./prepare_target.sh old_macdonald _6HzoUcx3eo 00:00:15 20`.

## Layout

```
data/collections/<name>/   collection.csv, credits.txt, frames.csv, sounds/
data/targets/<name>/       audio.wav, notes.csv
out/plots/                 figures
out/reconstructions/       rendered wavs
demo/                      published mixes
```

`data/` holds inputs and the analysis derived from them; the audio itself is fetched by the
scripts and not committed. `out/` is working output and is not committed. Each collection and
target lives in its own directory, so one run never overwrites another.

## Options

`reconstruct.py` takes:

- `--features` — `pitch`, `pitch-loudness` or `pitch-loudness-mfcc`, the three sets compared
  in the paper.
- `--fill` — what to do when the chosen source note is shorter than the target note, which is
  the usual case. `truncate` leaves the rest of the note silent; `longest` picks the longest
  frame at the right pitch; `concatenate` runs several frames together; `loop` repeats one
  frame; `stretch` slows one frame down to fit.
- `--max-pitch-deviation` — how far to settle for when the collection has nothing at the
  target's pitch. Pitch is settled before the other features rank anything, so an exact match
  is always preferred where one exists. The default `0` refuses rather than going out of tune,
  and names the note it could not fill.
- `--seed`, `--n-candidates`, `--no-normalize-loudness`, `--target-gain`.

Each placed segment is scaled to the level of the note it replaces, and selection skips frames
too quiet to get there without dragging their noise floor up; Freesound recordings span
roughly a 350x range in level.

`reconstruct.py` prints coverage, pitch deviation, level spread and how many segments were cut
short, then writes the reconstruction, a plot and a demo mix.

## Demos

`demo/<target>_<collection>/` holds mixes filed under the setting they vary — `by-fill-strategy/`
and `by-feature-set/` — each holding the reconstruction over a quiet copy of the original.
[demo/v0/](demo/v0/) holds the first round of demos.

## Tests

```sh
python tests/test_mosaic.py     # the analysis and selection logic
python tests/test_scripts.py    # the scripts end to end
```

Both synthesise their own audio, so they need no API key and no downloads.

## Attribution

Freesound sounds carry per-sound licenses; CC-BY requires crediting the uploader and CC BY-NC
forbids commercial use. `download_collection.py` writes `credits.txt` alongside each
collection, and `reconstruct.py` refuses to finish if any sound it used is missing from the
collection metadata.

## Credits

These scripts were adapted from templates provided in the
[AMP Lab](https://www.upf.edu/web/smc/audio-and-music-processing-lab) course at UPF.
