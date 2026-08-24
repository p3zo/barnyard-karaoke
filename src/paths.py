"""Where everything lives. Every path is relative to the repo root, which is where the
scripts are run from. The README sketches the same layout for a reader.
"""

import os

DATA = "data"
OUT = "out"
DEMO = "demo"


def collection_dir(collection):
    return os.path.join(DATA, "collections", collection)


def collection_sounds(collection):
    return os.path.join(collection_dir(collection), "sounds")


def collection_csv(collection):
    return os.path.join(collection_dir(collection), "collection.csv")


def collection_credits(collection):
    return os.path.join(collection_dir(collection), "credits.txt")


def collection_frames(collection):
    return os.path.join(collection_dir(collection), "frames.csv")


def target_dir(target):
    return os.path.join(DATA, "targets", target)


def target_audio(target):
    return os.path.join(target_dir(target), "audio.wav")


def target_notes(target):
    return os.path.join(target_dir(target), "notes.csv")


def plot(name):
    return os.path.join(OUT, "plots", f"{name}.png")


def reconstruction(name):
    return os.path.join(OUT, "reconstructions", f"{name}.wav")


def demo(target, collection, axis, name):
    """A published mix, filed under the axis it varies."""
    return os.path.join(DEMO, f"{target}_{collection}", axis, f"{name}.mp3")


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path
