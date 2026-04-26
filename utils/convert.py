import os
import json
import cv2
import numpy as np
from datasets import load_from_disk
from tqdm import tqdm

# -------------------------------
# PATHS
# -------------------------------

DATASET_PATH = "/Volumes/T7 Shield/hf_dataset_300"
OUTPUT_DIR = os.path.expanduser("~/lora_dataset")

IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
META_FILE = os.path.join(OUTPUT_DIR, "metadata.jsonl")

os.makedirs(IMAGE_DIR, exist_ok=True)

# -------------------------------
# LOAD DATASET
# -------------------------------

print("Loading dataset...")

dataset = load_from_disk(DATASET_PATH)
train_dataset = dataset["train"]

print("Total samples:", len(train_dataset))

MAX_SAMPLES = 60000

if len(train_dataset) > MAX_SAMPLES:
    train_dataset = train_dataset.shuffle(seed=42).select(range(MAX_SAMPLES))
    print(f"Using subset: {len(train_dataset)} samples")

# -------------------------------
# CONVERT DATASET
# -------------------------------

meta = open(META_FILE, "w")
failed = 0

for i, row in enumerate(tqdm(train_dataset)):
    try:
        image = row["image"]
        prompt = row["text"]

        image = np.array(image)

        # Convert RGBA to RGB if needed
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]

        filename = f"{i}.jpg"
        save_path = os.path.join(IMAGE_DIR, filename)

        cv2.imwrite(save_path, image[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])

        record = {
            "file_name": f"images/{filename}",
            "text": prompt
        }

        meta.write(json.dumps(record) + "\n")

    except Exception as e:
        failed += 1
        print(f"Skipped sample {i}: {e}")
        continue

meta.close()

print("\nConversion finished")
print(f"Saved: {MAX_SAMPLES - failed} images")
print(f"Failed: {failed}")
print("Images saved to:", IMAGE_DIR)
print("Metadata file:", META_FILE)