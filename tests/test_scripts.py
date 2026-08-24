"""Runs analyze.py and reconstruct.py end to end against a synthetic collection.

Run with `python tests/test_scripts.py`. Checks the command line glue (arguments,
filenames, reporting, plotting, encoding) rather than the analysis itself, which
tests/test_mosaic.py covers. Needs ffmpeg on PATH, no API key and no downloads.

download_collection.py is not exercised here because it only talks to Freesound.
"""

import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

FIXTURE = r'''
import os
import numpy as np
import pandas as pd
import essentia
import essentia.standard as estd

FS = 44100

def tone(midi, seconds, rng):
    n = int(seconds * FS)
    t = np.arange(n) / FS
    freq = 440 * 2 ** ((midi - 69) / 12) * (1 + 0.006 * np.sin(2 * np.pi * 5.5 * t))
    phase = 2 * np.pi * np.cumsum(freq) / FS
    env = np.minimum(1, np.minimum(t / 0.03, (seconds - t) / 0.08))
    return 0.4 * env * sum((0.7 ** k) * np.sin((k + 1) * phase) for k in range(6)) \
        + 0.002 * rng.standard_normal(n)

def write(path, samples):
    estd.MonoWriter(filename=path, format="wav", sampleRate=FS)(
        essentia.array(np.asarray(samples, dtype=np.float32)))

rng = np.random.default_rng(0)
SOUNDS = os.path.join("data", "collections", "testing", "sounds")
TARGET = os.path.join("data", "targets", "t")
os.makedirs(SOUNDS, exist_ok=True)
os.makedirs(TARGET, exist_ok=True)

rest = np.zeros(int(0.25 * FS))
write(os.path.join(TARGET, "audio.wav"), np.concatenate(
    sum([[tone(m, 0.9, rng), rest] for m in [60, 62, 64, 65, 67]], [])))

records = []
for i, midi in enumerate([60, 62, 64, 65, 67] * 3):
    path = os.path.join(SOUNDS, "s%d.wav" % i)
    write(path, tone(midi, 1.1, rng))
    records.append(dict(name="tone %d" % midi, username="tester", license="CC0",
                        tags="['tone']", freesound_id=1000 + i, path=path))
# Unpitched, so analyze.py has something to report as yielding no frames.
for j in range(2):
    path = os.path.join(SOUNDS, "n%d.wav" % j)
    write(path, 0.2 * rng.standard_normal(int(0.6 * FS)))
    records.append(dict(name="noise %d" % j, username="tester", license="CC-BY",
                        tags="['noise']", freesound_id=2000 + j, path=path))
pd.DataFrame(records).to_csv(os.path.join("data", "collections", "testing", "collection.csv"))
print("fixture ready")
'''

EXPECTED = [
    "data/collections/testing/frames.csv",
    "data/targets/t/notes.csv",
    "out/reconstructions/t_testing_pitch_concatenate.wav",
    "demo/t_testing/by-fill-strategy/concatenate.mp3",
    "out/plots/testing_pitch_coverage.png",
    "out/plots/t_pitch_contour.png",
    "out/plots/t_notes.png",
    "out/plots/t_testing_pitch_concatenate.png",
]


def main():
    work = tempfile.mkdtemp()
    src = os.path.join(work, "src")
    os.makedirs(src)
    for name in ("mosaic.py", "paths.py", "analyze.py", "reconstruct.py"):
        shutil.copy(os.path.join(ROOT, "src", name), src)
    with open(os.path.join(work, "_fixture.py"), "w") as f:
        f.write(FIXTURE)

    steps = [
        (["_fixture.py"], "build fixture"),
        (["src/analyze.py", "--collection", "testing"], "analyze collection"),
        (["src/analyze.py", "--target", "t"], "analyze target"),
        (["src/reconstruct.py", "--collection", "testing", "--target", "t",
          "--features", "pitch", "--fill", "concatenate"], "reconstruct"),
    ]

    failed = False
    for argv, label in steps:
        result = subprocess.run([sys.executable, "-u"] + argv, cwd=work,
                                capture_output=True, text=True)
        status = "PASS" if result.returncode == 0 else "FAIL"
        print(f"  {status}  {label}")
        if result.returncode != 0:
            failed = True
            print(result.stdout[-2000:])
            print(result.stderr[-2000:])

    for artifact in EXPECTED:
        path = os.path.join(work, artifact)
        exists = os.path.exists(path)
        failed |= not exists
        size = f" ({os.path.getsize(path)} bytes)" if exists else ""
        print(f"  {'PASS' if exists else 'FAIL'}  wrote {artifact}{size}")

    shutil.rmtree(work)
    print()
    if failed:
        print("FAILED")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
