# Barnyard Karaoke

Recreating melodies with animal sounds, via audio mosaicing. See [the paper](paper/main.tex).

## Setup

```sh
pip install -r requirements.txt
cp .env.template .env    # then fill in your Freesound key from https://freesound.org/apiv2/apply/
```

`prepare_target.sh` additionally needs `yt-dlp` and `ffmpeg` on your PATH.

## Usage

Prepare a target excerpt. The two used in the paper:

```sh
./prepare_target.sh V1bFr2SWP1I 00:01:05 10   # Somewhere Over the Rainbow
./prepare_target.sh _6HzoUcx3eo 00:00:15 20   # Old Macdonald Had A Farm
```

Then run the three scripts from inside [src/](src/), in order:

```sh
cd src
python download_collection.py --collection barnyard
python analyze.py --collection barnyard
python analyze.py --target over_the_rainbow --target-path targets/short_V1bFr2SWP1I.wav
python reconstruct.py --collection barnyard --target over_the_rainbow
```

`download_collection.py` takes `--collection barnyard` or `violin`; each writes its own
`dataframe_<collection>.csv`, `files_<collection>/` and `credits_<collection>.txt`, so one
collection never overwrites another.

`reconstruct.py` takes `--features` (`pitch`, `pitch-loudness` or `pitch-loudness-mfcc`, the
three sets compared in the paper), `--seed`, `--n-candidates`, and `--max-pitch-deviation`.

Pitch is settled before the other features rank anything, so an exact match is always
preferred where the collection has one. `--max-pitch-deviation` only says how far to settle
for when it has nothing at the target's pitch; the default `0` refuses rather than going out
of tune, and tells you which note it could not fill.

`reconstruct.py` prints coverage, pitch deviation and how many segments were cut short, and
writes the demo mix and plots.

The analysis and selection logic lives in [src/mosaic.py](src/mosaic.py); the scripts
configure it, report on it and plot the results.

## Tests

```sh
python tests/test_mosaic.py     # the analysis and selection logic
python tests/test_scripts.py    # the scripts end to end
```

Both synthesise their own audio, so they need no API key and no downloads. It checks that features
are comparable across notes of different lengths, that a note excludes the rest that follows
it, that standardising the features rebalances the distance, and that selection stays in
tune, does not read past the end of a source note, and is reproducible from its seed.

## Attribution

Freesound sounds carry per-sound licenses; CC-BY requires crediting the uploader and CC BY-NC
forbids commercial use. Notebook 1 writes `credits_<collection>.txt` for this, and notebook 3
refuses to finish if any sound it used is missing from the collection metadata.

`demo/over_the_rainbow_barnyard_*.mp3` were built with the current code from the collection
recorded in `src/dataframe_barnyard.csv` and `src/credits_barnyard.txt`, one per feature set.

The other mixes in [demo/](demo/) predate this. The violin ones can still be traced to
`src/dataframe_violin.csv` (whose `path` column refers to the old layout, but whose Freesound
IDs and licenses are intact). The barnyard ones cannot: that collection's metadata was
overwritten by the violin run, so the sounds behind them are unrecoverable. Rebuild the
barnyard collection and regenerate those demos before relying on them.

## Credits

These scripts were adapted from templates provided in the
[AMP Lab](https://www.upf.edu/web/smc/audio-and-music-processing-lab) course at UPF.
