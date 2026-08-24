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
./prepare_target.sh old_macdonald   _6HzoUcx3eo 00:00:15 20
python src/download_collection.py --collection barnyard
python src/analyze.py --collection barnyard
python src/analyze.py --target over_the_rainbow
python src/reconstruct.py --collection barnyard --target over_the_rainbow
```

`--collection instruments` builds a control collection of sustained single notes from real
instruments. A melody rebuilt from those should come back clean and in tune, which separates
faults in the method from the limits of the animal material.

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

## Source material

Two things decide whether a collection is any use.

The queries ask for sustained, pitched calls -- howls, moos, crows -- and exclude field
recordings, ambiences and loops by tag. Barks are left out: they are short and carry no pitch.

Then `analyze.py` drops frames that cannot stand in for a melody note. `pitch_drift_cents` is
the standard deviation of a frame's pitch contour about its own median, so a bleat that
warbles or a whinny that slides scores high and is cut by `--max-pitch-drift` (default 50,
about a quarter tone). `--min-frame-seconds` drops frames too short to carry a note. On the
animal collection this keeps about a quarter of the frames; on the instrument collection,
about two thirds, which is the difference the filter is there to find.

## Options

`reconstruct.py` takes:

- `--features` — `pitch`, `pitch-loudness` or `pitch-loudness-mfcc`, the three sets compared
  in the paper.
- `--fill` — what to do when the chosen source note is shorter than the target note, which is
  the usual case. `truncate` leaves the rest of the note silent; `longest` picks the longest
  frame at the right pitch; `concatenate` runs several frames together; `loop` repeats one
  frame; `stretch` slows one frame down to fit.
- `--max-pitch-deviation` — how far to settle for when the collection has nothing at the
  target's pitch class. Pitch is settled before the other features rank anything, so an exact
  match is always preferred. The default `0` refuses rather than going out of tune, and names
  the note it could not fill.
- `--no-octave-folding` — by default a frame matches when its pitch class does, whatever
  octave it sits in, and frames nearer the melody's own octave rank higher. A collection of
  animal calls holds few pitches steadily but many pitch classes, and displacing a note by an
  octave keeps it consonant where settling for a semitone would not.
- `--max-pitch-shift` — semitones a frame may be moved to land it exactly in tune (default 1;
  `0` disables). Frames already in tune are preferred, so this only comes into play on pitch
  classes the collection barely covers. It is what stops a repeated note reaching for the same
  recording every time: MIDI 59 has two recordings sitting on it and forty within a semitone.
- `--seed`, `--n-candidates`, `--no-normalize-loudness`, `--target-gain`.

Each placed segment is scaled to the level of the note it replaces, and selection skips frames
too quiet to get there without dragging their noise floor up; Freesound recordings span
roughly a 350x range in level.

Recordings used in the last few notes are passed over while alternatives remain, so the same
animal is not heard twice in a row.

`reconstruct.py` prints coverage, pitch deviation, level spread, how many segments were cut
short and how many were shifted or displaced by an octave, then writes the reconstruction, a
plot and a demo mix.

## Demos

`demo/<target>_<collection>/` holds mixes filed under the setting they vary --
`by-fill-strategy/` and `by-feature-set/` -- each the reconstruction over a quiet copy of the
original. Both targets are built from both collections, so the animal versions can be heard
against the instrument control. [demo/v0/](demo/v0/) holds the first round of demos.

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
