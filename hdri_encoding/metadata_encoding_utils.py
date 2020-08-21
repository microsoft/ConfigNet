# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Common functions used by the metadata encoding scripts"""
import json

def load_metadata_dicts(metadata_files):
    metadata_dicts = []
    for metadata_file in metadata_files:
        with open(metadata_file, "r") as fp:
            metadata_dicts.append(json.load(fp))

    return metadata_dicts

def save_metadata_dicts(metadata_dicts, metadata_files):
    assert(len(metadata_dicts) == len(metadata_files))
    for i in range(len(metadata_dicts)):
        with open(metadata_files[i], "w") as fp:
            json.dump(metadata_dicts[i], fp, indent=4)
