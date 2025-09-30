from src.ner_model import model, ner_labels
from src.hf_datasets import dataset
from src.filtered_data import filtered_med_data

import os
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
import numpy as np
import random
from gliner import GLiNER
import torch

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
random.seed(42)
np.random.seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on:", device)


# Track terms
non_bio_term = []
bio_term = []

for q in tqdm(dataset['question'], total=len(dataset)):
    text = q[:150]  # truncate early
    entities = model.predict_entities(text, ner_labels, threshold=0.80)

    # Only keep label + score
    filtered_entities = [
        {"label": ent["label"], "score": ent["score"]}
        for ent in entities
    ]

    record = {
        "question": text,
        "entities": filtered_entities
    }

    if "bio" in text.lower():
        bio_term.append(record)
        print("\n")
        print(f"Found a bio term: {text}")
        print(f"Entities: {filtered_entities}")
        print("\n")
        continue

    if entities == []:
        non_bio_term.append(record)
    else:
        bio_term.append(record)
        print("\n")
        print(f"Found a bio term: {text}")
        print(f"Entities: {filtered_entities}")
        print("\n")



output_file_path_good = "Final_Bio_terms.txt"
with open(output_file_path_good, "w") as outfile:
    for item in bio_term:
        outfile.write(f"{item}\n")

print(f"Saved good terms to {output_file_path_good}, with {len(bio_term)} terms")


hf_med_filtered_data = filtered_med_data(output_file_path_good, dataset)

hf_med_filtered_data.save_to_disk("hf_med_filtered_data")

print(f"+--------------- Saved filtered data to {hf_med_filtered_data} ---------------+")



