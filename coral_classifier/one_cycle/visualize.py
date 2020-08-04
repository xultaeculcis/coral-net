import pandas as pd
from PIL import Image

df = pd.read_csv('scoring_results_Aqua SD - Photos _ Facebook_files.csv')
acro_candidates = df[df.confidence > 0.75]
print(len(acro_candidates))

for i, row in acro_candidates.iterrows():
    Image.open(row.img_path).show()

print("dupa")
