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

# Durations are capped per query because shorter recordings are more likely to be
# single-shot sounds than field recordings; cows and sheep get longer because they
# take longer to vocalise.
COLLECTIONS = {
    "barnyard": [
        {"num_results": 10, "query": "cat meow", "filter": "duration:[0 TO 5]", "sort": "rating_desc"},
        {"num_results": 20, "query": "dog bark", "filter": "duration:[0 TO 1]", "sort": "rating_desc"},
        {"num_results": 20, "query": "cow moo", "filter": "duration:[0 TO 10]", "sort": "rating_desc"},
        {"num_results": 5, "query": "horse whinny", "filter": "duration:[0 TO 5]", "sort": "rating_desc"},
        {"num_results": 5, "query": "horse neighing", "filter": "duration:[0 TO 5]", "sort": "rating_desc"},
        {"num_results": 20, "query": "bird chirp", "filter": "duration:[0 TO 10]", "sort": "rating_desc"},
        {"num_results": 20, "query": "bleat", "filter": "duration:[0 TO 10]", "sort": "rating_desc"},
    ],
    "violin": [
        {"num_results": 100, "query": "violin", "filter": "ac_single_event:True", "sort": None},
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
