import os
from pathlib import Path

from imagededup.methods import CNN


def deduplicate(root_data_path):
    method_object = CNN()
    folders = sorted([os.path.join(root_data_path, folder) for folder in os.listdir(root_data_path)])
    for i, image_folder in enumerate(folders):
        print(f"Processing folder {i+1}/{len(folders)} - {image_folder}")

        to_remove = method_object.find_duplicates_to_remove(
            image_dir=Path(image_folder),
            min_similarity_threshold=0.95
        )

        print(f"Removing {len(to_remove)} duplicated images.")

        for img in [os.path.join(image_folder, image_to_remove) for image_to_remove in to_remove]:
            os.remove(img)

        print("DONE")
        print("="*80)


if __name__ == "__main__":
    data_path = "../../datasets/intermediate-stage"
    deduplicate(data_path)
