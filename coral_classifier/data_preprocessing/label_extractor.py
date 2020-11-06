from pathlib import Path
from typing import Union

import pandas as pd
import numpy as np
import os
import glob
import logging
import re
from tqdm import tqdm
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

data_dir = "../../datasets/scrape/"


class LabelExtractor:
    label_column_name = "Target Label"

    def __init__(
        self,
        rename: bool = True,
        move: bool = True,
        output_path: Union[str, Path] = "../../datasets/csv"
    ) -> None:
        self.rename = rename
        self.move = move
        self.output_path = output_path
        self.target_labels = {
            'acanthastrea': [
                'lord',
                'lords',
                'echinata',
                'acanthastrea',
                'bowerbanki',
                'micromussa',
                'acan',
                'candy cane aussie lord',
                'lordhowensis'
            ],
            'acanthophyllia': ['acanthophyllia'],
            'acropora': ['acro', 'acrpopora', 'acropora', 'stag', 'millepora', 'milli', 'mille'],
            'alveopora': ['alveopora'],
            'anacropora': ['anacropora'],
            'anthellia': ['anthellia', 'snowflake polyps'],
            'astreopora': ['astreopora'],
            'australophyllia': ['australophyllia', 'australophyllia wilsoni'],
            'blastomussa': ['blastomussa', 'blasto', 'black cherry'],
            'bubble coral': ['bubble coral'],
            'bubble tip anemone': ['bubble tip anemone', 'bta', 'bubble anemone', 'black widow anemone'],
            'candy cane': ['candy cane', 'canes', 'cane'],
            'carpet anemone': ['carpet anemone'],
            'cespitularia': ['cespitularia'],
            'chalice': ['chalice', 'mycedium', 'echinopora'],
            'clove polyps': ['clove polyps', 'clove polyp', 'daisy polyp', 'daisy polyps'],
            'condylactis anemone': ['condylactis anemone'],
            'tridacna clam': [
                'clam',
                'tridacna',
                'maxima',
                'derasa',
                'tevoroa',
                'gigas',
                'mbalavuana',
                'squamosina',
                'crocea',
                'squamosa'
            ],
            'cynarina': ['cynarina'],
            'cyphastrea': ['cyphastrea'],
            'diaseris': ['diaseris', 'razor coral', 'fragilis'],
            'duncan': ['duncan', 'duncans', 'elegance'],
            'euphyllia': ['euphyllia'],
            'euphyllia frogspawn': ['frogspawn', 'hammered frog'],
            'euphyllia hammer': ['hammer'],
            'euphyllia torch': ['torch'],
            'favia': ['favia', 'faviidae'],
            'favites': ['favites', 'war coral'],
            'galaxea': ['galaxea'],
            'goniastrea': ['goniastrea'],
            'goniopora': ['goniopora', 'gonio', 'goni'],
            'gorgonia': ['gorgonia'],
            'hydnophora': ['hydnophora', 'hydno'],
            'leather coral': ['leather coral', 'leather', 'toadstool', 'lobophytum'],
            'leptastrea': ['leptastrea', 'lepta'],
            'leptoseris': ['leptoseris', 'lepto'],
            'lithophyllon': ['lithophyllon'],
            'lobophyllia': ['lobo', 'lobophyllia', 'lobphyllia'],
            'merulina': ['merulina'],
            'mini carpet anemone': ['mini carpet', 'mini carpet anemone'],
            'montipora': [
                'montipora',
                'monti',
                'cap',
                'undata',
                'hispida',
                'capricornis',
                'setosa',
                'aequituberculata',
                'spongodes'
            ],
            'montipora digitata': ['digitata'],
            'mushroom': ['mushroom', 'shroom'],
            'mushroom bounce': ['bounce'],
            'mushroom discosoma': ['discosoma', 'disco mushroom'],
            'mushroom rhodactis': ['rhodactis', 'rhodactus', 'st thomas'],
            'mushroom ricordea': ['ricordea', 'ricodea', 'yuma', 'florida'],
            'oulophyllia': ['oulophyllia'],
            'pachyseris': ['pachyseris'],
            'pavona': ['pavona', 'cactus'],
            'pectinia': ['pectinia'],
            'pipe organ': ['pipe organ'],
            'plate coral fungia': ['plate coral', 'fungia', 'cycloseris'],
            'platygyra': ['platygyra', 'platy', 'platgyra', 'brain coral'],
            'plesiastrea': ['plesiastrea'],
            'pocillopora': ['pocillopora'],
            'porites': ['porites'],
            'psammocora': ['psammocora', 'psammy', 'psammacora'],
            'rock flower anemone': [
                'rock anemone',
                'flower anemone',
                'rock anemone',
                'rock flower',
                'rock nem'
            ],
            'waratah anemone': ['waratah anemone'],
            'scolymia': ['scolymia', 'scoly'],
            'seriatopora': ['seriatopora', 'bridsnest', 'birdsnest', 'birds nest', 'birdnest'],
            'star polyps': ['star polyps', 'star polyp'],
            'stylocoeniella': ['stylocoeniella'],
            'stylophora': ['stylophora', 'stylo'],
            'sun coral': ['sun coral'],
            'sympodium': ['sympodium'],
            'symphyllia': ['symphyllia', 'symphyllia wilsoni', 'smphyllia'],
            'trachyphyllia': ['trachyphyllia'],
            'tube anemone': ['tube anemone'],
            'turbinaria': ['turbinaria', 'scroll'],
            'wellsophyllia': ['wellsophyllia'],
            'christmas tree worm coral': ['worm', 'xmas tree', 'christmas tree'],
            'xenia': ['xenia'],
            'snake polyps': ['snake polyps'],
            'zoanthid & palys': [
                'zoanthids',
                'protopalythoa',
                'protopalythoas',
                'zoanthid',
                'zoas',
                'zoa',
                'paly',
                'palys',
                'palythoa',
                'palythoas',
                'bam bams',
                'dragon eyes'
            ]
        }

    def as_shopify(self, frame: pd.DataFrame, column_name: str) -> pd.DataFrame:
        logging.info("Data comes from Shopify site")
        return self._extract_target_label(frame, column_name)

    def as_woocommerce(self, frame: pd.DataFrame, column_name: str) -> pd.DataFrame:
        logging.info("Data comes from Woocommerce site")
        return self._extract_target_label(frame, column_name)

    def assign_classes(self, frame: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"File contains {len(frame)} products")

        if "Collection" in frame.columns:
            frame = self.as_shopify(frame, "Name")
        else:
            frame = self.as_woocommerce(frame, "product_name")

        return frame

    def _extract_target_label(self, frame: pd.DataFrame, column_name: str) -> pd.DataFrame:
        possible_duplicate_labels = [
            'acropora',
            'mushroom',
            'euphyllia',
            'stylophora',
            'montipora',
            'lobophyllia',
            'candy cane',
        ]

        frame_cpy = frame.copy()
        frame_cpy[self.label_column_name] = "unknown"

        # for each product in df
        for i, row in tqdm(frame.iterrows(), total=len(frame)):
            possible_labels = set()

            # loop over possible keywords in each target label and assign
            # possible label to this product
            for label in self.target_labels.keys():
                for key_word in self.target_labels[label]:
                    if re.search(r"(\b" + f"{key_word}" + r"\b)", row[column_name].lower()):
                        possible_labels.add(label)

            # if only one label was assigned then we can finish
            if len(possible_labels) == 1:
                frame_cpy.at[i, self.label_column_name] = list(possible_labels)[0]

            # if not then we need to decide which label to remove
            else:
                for label in possible_duplicate_labels:
                    if label not in possible_labels:
                        continue

                    possible_labels.remove(label)
                    if len(possible_labels) > 1:
                        logging.warning(f"Over 2 labels for product {(i, row[column_name])}")
                    frame_cpy.at[i, self.label_column_name] = list(possible_labels)[0]

        return frame_cpy


if __name__ == "__main__":
    logging.info("Running class assignment script")
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    logging.info(f"Found {len(csv_files)} files to process: {csv_files}")

    csv = "../../datasets/scrape/worldwidecorals.com.csv"

    logging.info(f"Loading file {csv}")

    df = pd.read_csv(csv)
    lex = LabelExtractor()
    df = lex.assign_classes(df)
