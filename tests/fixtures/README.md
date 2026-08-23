# Frozen fixtures

`prefix-analysis_*.csv` are the violin analysis as it was produced *before* the fixes in
`src/mosaic.py` — one MFCC over each whole note, loudness divided by frame length, frames
running from one onset to the next. They are kept as a frozen record so
`tests/test_mosaic.py` can demonstrate, on real data, what standardising the features
changes: on these raw columns `mfcc_0` is 55% of the distance between two frames and
`mean_pitch` is 0.3%.

Do not use them as pipeline input. Regenerate analysis dataframes by running the notebooks.
