"""Download a collection of sounds from Freesound to use as mosaicing source material.

    python src/download_collection.py --collection barnyard

Writes collection.csv, credits.txt and sounds/ into data/collections/<collection>/.
Needs a Freesound API key in .env at the repo root.
"""

import argparse
import os

import freesound
import pandas as pd
from dotenv import load_dotenv

import paths

METADATA_FIELDS = ["id", "name", "username", "previews", "license", "tags"]

# Tags that mark a recording as unsuitable: field recordings and ambiences carry
# background noise and several overlapping sources, loops and music are not single calls.
EXCLUDE_TAGS = " ".join(
    f"-tag:{tag}"
    for tag in ("field-recording", "ambience", "ambient", "soundscape", "loop",
                "music", "atmosphere", "multiple")
)

# A call has to last long enough to fill a melody note, and stop before it turns into a
# recording of a farmyard. Sustained, pitched calls are what a melody can be built from,
# so the collection is weighted towards howls, moos and crows rather than barks, which
# are short and carry no pitch at all.
# Counts are set well above what the reconstruction needs. Only about a quarter of
# animal frames survive the steadiness filter, and what matters is having several
# different recordings at every pitch class, not just one.
BARNYARD_QUERIES = [
    ("cat meow", 60),
    ("dog howl", 50),
    ("wolf howl", 50),
    ("cow moo", 40),
    ("rooster crow", 40),
    ("duck quack", 30),
    ("sheep baa", 30),
    ("goat bleat", 25),
    ("horse whinny", 18),
    ("owl hoot", 10),
]

# A control collection: sustained single notes from real instruments. If the pipeline is
# working, a melody rebuilt from these should come back recognisable and in tune, which
# separates faults in the method from the limits of the animal material.
INSTRUMENT_QUERIES = [
    ("flute note", 25),
    ("cello note", 25),
    ("violin note", 25),
    ("trumpet note", 20),
    ("clarinet note", 20),
    ("saxophone note", 20),
    ("piano note", 20),
    ("organ note", 20),
    ("french horn note", 15),
    ("oboe note", 15),
]

COLLECTIONS = {
    "barnyard": [
        {"num_results": n, "query": q, "sort": "rating_desc",
         "filter": f"duration:[0.35 TO 4] {EXCLUDE_TAGS}"}
        for q, n in BARNYARD_QUERIES
    ],
    "instruments": [
        {"num_results": n, "query": q, "sort": "rating_desc",
         "filter": f"duration:[0.5 TO 6] {EXCLUDE_TAGS} -tag:chord -tag:melody -tag:phrase"}
        for q, n in INSTRUMENT_QUERIES
    ],
}


def query_freesound(client, query, filter, sort, num_results):
    # search() returns the first page already loaded, so iterate it directly.
    pager = client.search(
        query=query,
        filter=filter,
        sort=sort,
        fields=",".join(METADATA_FIELDS),
        group_by_pack=1,
        page_size=num_results,
    )
    return list(pager)


def make_record(sound, directory):
    record = {key: sound.as_dict()[key] for key in METADATA_FIELDS}
    del record["previews"]
    record["freesound_id"] = record.pop("id")
    record["path"] = os.path.join(directory, sound.previews.preview_hq_ogg.split("/")[-1])
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", choices=sorted(COLLECTIONS), required=True)
    args = parser.parse_args()

    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    client = freesound.FreesoundClient()
    client.set_token(os.environ["FREESOUND_API_KEY"])

    files_dir = paths.collection_sounds(args.collection)
    dataframe_path = paths.collection_csv(args.collection)
    credits_path = paths.collection_credits(args.collection)
    os.makedirs(files_dir, exist_ok=True)

    sounds = []
    for query in COLLECTIONS[args.collection]:
        results = query_freesound(
            client, query["query"], query["filter"], query["sort"], query["num_results"]
        )
        print(f"{query['query']!r}: {len(results)} results")
        sounds += results

    for count, sound in enumerate(sounds):
        print(f"Downloading sound with id {sound.id} [{count + 1}/{len(sounds)}]")
        freesound.FSRequest.retrieve(
            sound.previews.preview_hq_ogg,
            client,
            os.path.join(files_dir, sound.previews.preview_hq_ogg.split("/")[-1]),
        )

    df = pd.DataFrame([make_record(s, files_dir) for s in sounds])

    # Queries overlap -- a goat answers to both "cow moo" and "bleat" -- and a sound kept
    # twice would get twice the chance of being picked for any note it matches.
    duplicated = df.duplicated("freesound_id").sum()
    if duplicated:
        print(f"Dropping {duplicated} sounds returned by more than one query")
        df = df.drop_duplicates("freesound_id").reset_index(drop=True)

    df.to_csv(dataframe_path)
    print(f"Saved DataFrame with {len(df)} entries! {dataframe_path}")

    # Freesound sounds are individually licensed: CC-BY requires crediting the uploader
    # and BY-NC forbids commercial use. Write the credits next to the collection so a
    # published demo can be attributed to the sounds it was actually built from.
    with open(credits_path, "w") as f:
        for _, row in df.iterrows():
            f.write(
                f"{row['name']} by {row['username']} -- "
                f"https://freesound.org/s/{row['freesound_id']}/ -- {row['license']}\n"
            )
    print(f"Wrote {credits_path}")
    print(df["license"].value_counts().to_string())


if __name__ == "__main__":
    main()
