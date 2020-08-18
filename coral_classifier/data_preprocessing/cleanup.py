import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import Union, List
from uuid import uuid4

from PIL import Image
from tqdm import tqdm

from coral_classifier.data_preprocessing.duplicate_remover import DuplicateRemover
from coral_classifier.guard import Guard

handler = logging.StreamHandler()
root = logging.getLogger()
root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
root.addHandler(handler)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--root-data-path',
                        metavar='DIR',
                        type=str,
                        default="../../datasets/data-lake",
                        help='Root directory where the raw data resides',
                        dest='root_data_path')
    parser.add_argument('--rename-output-folder',
                        default="../../datasets/intermediate-stage",
                        type=str,
                        metavar='rof',
                        help='If specified the renaming operation will copy original files to the new location first'
                             'and only then perform the renaming. None by default - renaming is performed inplace.',
                        dest='rename_output_folder')
    parser.add_argument('--convert-to',
                        default=".jpg",
                        type=str,
                        metavar='ct',
                        help='The extension to which to convert all of the files.',
                        dest='convert_to')

    return parser.parse_args()


def rename_files(original_data_folder: Union[str, Path],
                 file_names: Union[List[str], List[Path]],
                 new_location: Union[str, Path] = None,
                 class_name: str = None) -> List[str]:
    Guard.against_none(file_names, "file_paths")

    inplace_str = f"The renamed files will be transferred to the new location in: {new_location}" \
        if new_location \
        else "Renaming will be done inplace."

    logging.info(f"Renaming files. {inplace_str}")

    if class_name is not None and new_location is not None:
        new_location = os.path.join(new_location, class_name)

    renamed_files = []
    for current_file_name in tqdm(file_names):
        original_file_path = os.path.join(original_data_folder, current_file_name)
        new_file_name = str(uuid4()).replace('-', '')
        extension = os.path.splitext(current_file_name)[1]

        new_file_path = os.path.join(
            new_location if new_location else original_data_folder,
            new_file_name + extension
        )

        os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
        shutil.copy(original_file_path, new_file_path)
        renamed_files.append(new_file_path)

    return renamed_files


def rename_files_in_folders(class_folders: Union[List[str], List[Path]],
                            root_data_path: Union[str, Path],
                            rename_output_folder: Union[str, Path] = None,
                            class_name: str = None) -> List[str]:
    renamed_file_paths = []
    for folder in class_folders:
        images = os.listdir(os.path.join(root_data_path, folder))
        logging.info(f"Found {len(images)} in folder {folder}.")
        file_paths = rename_files(os.path.join(root_data_path, folder), images, rename_output_folder, class_name)
        renamed_file_paths.extend(file_paths)

    return renamed_file_paths


def convert_files(files: Union[List[str], List[Path]], target_extension: str):
    Guard.against_none(files, "file_paths")
    converted_images = []

    logging.info(f"Converting images to common file format - '{target_extension}'")

    for current_file in tqdm(files):
        root_file_name, current_extension = os.path.splitext(current_file)

        if current_extension == target_extension:
            # Nothing to do...
            converted_images.append(current_file)
            continue

        img = Image.open(current_file)

        if not img.mode == 'RGB':
            img = img.convert('RGB')

        file_with_proper_extension = root_file_name + target_extension
        img.save(file_with_proper_extension, quality=100)
        os.remove(current_file)
        converted_images.append(file_with_proper_extension)


def remove_duplicates(paths: Union[List[str], List[Path]]):
    Guard.against_none(paths, "file_paths")

    logging.info("Removing duplicated files")

    duplicate_remover = DuplicateRemover(paths)
    duplicate_remover.check_for_duplicates()


def main(args):
    classes = sorted(os.listdir(args.root_data_path))
    for class_name, folder in zip(classes, [os.path.join(args.root_data_path, c) for c in classes]):
        class_folders = os.listdir(folder)
        file_paths = rename_files_in_folders(class_folders,
                                             folder,
                                             args.rename_output_folder,
                                             class_name)
        convert_files(file_paths, args.convert_to)

        if args.rename_output_folder is None:
            dirs_to_chek = [os.path.join(args.root_data_path, folder) for folder in args.class_folders]
        else:
            dirs_to_chek = [os.path.join(args.root_data_path, args.rename_output_folder)]

        remove_duplicates(dirs_to_chek)


if __name__ == "__main__":
    arguments = get_args()
    logging.info(f"Raw data processing has started with following arguments:\n {vars(arguments)}")
    main(arguments)
    logging.info("Done")
