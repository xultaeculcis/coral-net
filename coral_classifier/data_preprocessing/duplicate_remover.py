"""
Taken from https://stackoverflow.com/questions/748675/finding-duplicate-files-and-removing-them
"""

import hashlib
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Union

from tqdm import tqdm


import logging


class DuplicateRemover:
    def __init__(self, paths: Union[List[str], List[Path]]):
        self.paths = paths

    @staticmethod
    def chunk_reader(fobj, chunk_size=1024):
        """Generator that reads a file in chunks of bytes"""
        while True:
            chunk = fobj.read(chunk_size)
            if not chunk:
                return
            yield chunk

    def get_hash(self, filename, first_chunk_only=False, hash_func=hashlib.sha1):
        hashobj = hash_func()
        file_object = open(filename, 'rb')

        if first_chunk_only:
            hashobj.update(file_object.read(1024))
        else:
            for chunk in self.chunk_reader(file_object):
                hashobj.update(chunk)
        hashed = hashobj.digest()

        file_object.close()
        return hashed

    def check_for_duplicates(self, hash_func=hashlib.sha1):
        hashes_by_size = defaultdict(list)  # dict of size_in_bytes: [full_path_to_file1, full_path_to_file2, ]
        hashes_on_1k = defaultdict(list)  # dict of (hash1k, size_in_bytes): [full_path_to_file1, full_path_to_file2, ]
        hashes_full = {}  # dict of full_file_hash: full_path_to_file_string
        files = []

        logging.info("Detecting duplicates.")
        logging.info("Checking file sizes.")
        for path in self.paths:
            for dirpath, _, filenames in os.walk(path):
                for filename in filenames:
                    full_path = os.path.join(dirpath, filename)
                    try:
                        # if the target is a symlink (soft one), this will
                        # dereference it - change the value to the actual target file
                        full_path = os.path.realpath(full_path)
                        file_size = os.path.getsize(full_path)
                    except OSError:
                        # not accessible (permissions, etc) - pass on
                        continue
                    hashes_by_size[file_size].append(full_path)

        # For all files with the same file size, get their hash on the 1st 1024 bytes only
        logging.info("Checking for duplicates based on the hashes computed on first 1k bytes")
        for size_in_bytes, files in tqdm(hashes_by_size.items()):
            if len(files) < 2:
                continue  # this file size is unique, no need to spend CPU cycles on it

            for filename in files:
                try:
                    small_hash = self.get_hash(filename, first_chunk_only=True, hash_func=hash_func)
                    # the key is the hash on the first 1024 bytes plus the size - to
                    # avoid collisions on equal hashes in the first part of the file
                    # credits to @Futal for the optimization
                    hashes_on_1k[(small_hash, size_in_bytes)].append(filename)
                except (OSError,):
                    # the file access might've changed till the exec point got here
                    continue

        # For all files with the hash on the 1st 1024 bytes,
        # get their hash on the full file - collisions will be duplicates
        logging.info("Verifying and removing duplicate candidates.")
        total_removed = 0
        for __, files_list in tqdm(hashes_on_1k.items()):
            if len(files) < 2:
                continue  # this hash of fist 1k file bytes is unique, no need to spend cpy cycles on it

            for filename in files_list:
                try:
                    full_hash = self.get_hash(filename, first_chunk_only=False)
                    duplicate = hashes_full.get(full_hash)
                    if duplicate:
                        os.remove(filename)
                        total_removed += 1
                    else:
                        hashes_full[full_hash] = filename
                except (OSError,):
                    # the file access might've changed till the exec point got here
                    continue

        logging.info(f"Removed total of {total_removed} duplicates")
