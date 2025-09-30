import torch
from gliner import GLiNER

device = "cuda" if torch.cuda.is_available() else "cpu"

model = GLiNER.from_pretrained("Ihor/gliner-biomed-large-v1.0").to(device)
model.eval()
print(f"-------------- Loaded NER model to {device} --------------")

ner_labels = [
    "Medicine",
    "Health",
    "Disease",
    "Pathology",
    "Pharmacology",
    "Surgery",
    "Nursing",
    "Ophthalmology",
    "Dermatology",
    "Radiology",
    "Immunology",
    "Epidemiology",
    "Neuroscience",
    "Diagnosis",
    "Treatment",
    "Disorder",
    "Medical Exams",
    'Genetics',
    "Medical",
    'Pediatric',
    'Forensic',
    'Parasitology',
    'Symptom',
    "Injury",
    "Organs"
]