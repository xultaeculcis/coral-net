import glob
import os
import uuid

import dask.bag as db
import pandas as pd
from PIL import Image


def dasked_rename(img_path):
    try:
        img = Image.open(img_path)
        dirname = os.path.dirname(img_path)
        new_name = os.path.join(dirname, str(uuid.uuid4()).replace("-", "") + target_extension)
        img = img.convert("RGB")
        img.save(new_name)
        os.remove(img_path)
        return img_path, 1, None
    except Exception as ex:
        return img_path, 0, str(ex)


if __name__ == "__main__":
    p = "../../datasets/labeled"
    target_extension = ".jpg"
    extensions = ["*.jpg", "*.png"]
    files = []
    for extension in extensions:
        pattern = os.path.join(p, "*", extension)
        files.extend(sorted(glob.glob(pattern, recursive=False)))

    results = db.from_sequence(files).map(dasked_rename).compute()

    df = pd.DataFrame(results, columns=["file", "success", "exception"])
    print(df.exception.value_counts())
