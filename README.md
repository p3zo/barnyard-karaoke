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

Then run the notebooks in [src/](src/) in order:

1. [1-create-source-collection.ipynb](src/1-create-source-collection.ipynb) downloads a
   collection of sounds from Freesound. Set `COLLECTION` to `barnyard` or `violin`; each
   writes its own `dataframe_<collection>.csv`, `files_<collection>/` and
   `credits_<collection>.txt`.
2. [2-analyze-source-collection-and-target.ipynb](src/2-analyze-source-collection-and-target.ipynb)
   extracts one row of features per note, for the collection and for the target.
3. [3-reconstruct-target.ipynb](src/3-reconstruct-target.ipynb) rebuilds the target from
   source frames and writes a demo mix.

The analysis and selection logic lives in [src/mosaic.py](src/mosaic.py); the notebooks
configure it and plot the results.

## Tests

```sh
python tests/test_mosaic.py
```

Synthesises its own audio, so it needs no API key and no downloads. It checks that features
are comparable across notes of different lengths, that a note excludes the rest that follows
it, that standardising the features rebalances the distance, and that selection stays in
tune, does not read past the end of a source note, and is reproducible from its seed.

## Attribution

Freesound sounds carry per-sound licenses; CC-BY requires crediting the uploader and CC BY-NC
forbids commercial use. Notebook 1 writes `credits_<collection>.txt` for this, and notebook 3
refuses to finish if any sound it used is missing from the collection metadata.

The demo mixes in [demo/](demo/) predate this. The violin ones can still be traced to
`src/dataframe_violin.csv` (whose `path` column refers to the old layout, but whose Freesound
IDs and licenses are intact). The barnyard ones cannot: that collection's metadata was
overwritten by the violin run, so the sounds behind them are unrecoverable. Rebuild the
barnyard collection and regenerate those demos before relying on them.

## Credits

These scripts were adapted from templates provided in the
[AMP Lab](https://www.upf.edu/web/smc/audio-and-music-processing-lab) course at UPF.
