import json
import os
import csv
from sentence_transformers import SentenceTransformer, util


DATASET_PATH = "datasets/UseCase5_01.csv" 
SIMILARITY_THRESHOLD = 0.9


def load_prompts(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return []

    prompts = []

    if filepath.endswith(".jsonl"):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    prompt = item.get("prompt") or item.get("input") or item.get("question")
                    if prompt:
                        prompts.append(prompt)
                except json.JSONDecodeError:
                    continue

    elif filepath.endswith(".json"):
        with open(filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, dict) and "Questions" in raw:
            for item in raw["Questions"]:
                prompt = item.get("RawQuestion") or item.get("ProcessedQuestion")
                if prompt:
                    prompts.append(prompt)
        elif isinstance(raw, dict):
            for key in ["questions", "data", "entries"]:
                if key in raw and isinstance(raw[key], list):
                    raw = raw[key]
                    break
            for item in raw:
                if isinstance(item, dict):
                    prompt = item.get("prompt") or item.get("input") or item.get("question")
                    if prompt:
                        prompts.append(prompt)
        elif isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    prompt = item.get("prompt") or item.get("input") or item.get("question")
                    if prompt:
                        prompts.append(prompt)

    elif filepath.endswith(".csv"):
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt = row.get("prompt") or row.get("input") or row.get("question")
                if prompt:
                    prompts.append(prompt)

    else:
        print("Unsupported file format.")
        return []

    return prompts


prompts = load_prompts(DATASET_PATH)
num_prompts = len(prompts)
if num_prompts == 0:
    print("No valid prompts found in dataset.")
    exit()

print(f"Loaded {num_prompts} prompts")


avg_length = sum(len(p.split()) for p in prompts) / num_prompts
print(f"Average Prompt Length: {avg_length:.2f} words")


print("Embedding prompts...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(prompts, convert_to_tensor=True)

print("Computing pairwise cosine similarities...")
similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings)

similar_pairs = 0
for i in range(num_prompts):
    for j in range(i + 1, num_prompts):
        if similarity_matrix[i][j] >= SIMILARITY_THRESHOLD:
            similar_pairs += 1

similar_percent = (similar_pairs / (num_prompts * (num_prompts - 1) / 2)) * 100

print(f"Similar Prompt Pairs (≥ {SIMILARITY_THRESHOLD} cosine): {similar_pairs}")
print(f"Semantic Similarity Rate: {similar_percent:.2f}%")


print("\nSample high-similarity prompt pairs:")
count = 0
for i in range(num_prompts):
    for j in range(i + 1, num_prompts):
        if similarity_matrix[i][j] >= SIMILARITY_THRESHOLD and count < 5:
            print(f"\nPrompt A: {prompts[i]}")
            print(f"Prompt B: {prompts[j]}")
            print(f"Cosine Similarity: {similarity_matrix[i][j]:.4f}")
            count += 1
