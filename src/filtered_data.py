import ast
from datasets import Dataset

def filtered_med_data(file_path, dataset):
    matches = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                record = ast.literal_eval(line.strip())
                file_question = record.get("question", "")[:150]

                for row in dataset:
                    if row["question"][:150] == file_question:
                        matches.append(row)

    return Dataset.from_list(matches)