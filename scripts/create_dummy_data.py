# ==========================================
# SCRIPTS/CREATE_DUMMY_DATA.PY - MAKES DUMMY DATA FOR TESTING
# ==========================================
import os
import json
import numpy as np
from PIL import Image


def create_dummy_dataset():
    dummy_frames_dir = "./dummy_extracted_frames"
    dummy_json_path = "./dummy_wlasl.json"

    # 1. Dummy Video IDs and Glosses
    dummy_data = [
        {
            "gloss": "book",
            "instances": [{"video_id": "00001", "split": "train"}, {"video_id": "00002", "split": "val"}]
        },
        {
            "gloss": "drink",
            "instances": [{"video_id": "00003", "split": "test"}]
        }
    ]

    with open(dummy_json_path, 'w', encoding='utf-8') as f:
        json.dump(dummy_data, f, indent=2)

    # 2. Create Dummy Frames (32 RGB images per video)
    vids = ["00001", "00002", "00003"]
    for vid in vids:
        vid_dir = os.path.join(dummy_frames_dir, vid)
        os.makedirs(vid_dir, exist_ok=True)
        for i in range(32):
            # Create a random noise RGB image (224x224)
            img_arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_arr)
            img.save(os.path.join(vid_dir, f"frame_{i:02d}.jpg"))

    print(f"✅ Dummy dataset created successfully!")
    print(f"📁 Frames at: {dummy_frames_dir}")
    print(f"📄 JSON at: {dummy_json_path}")


if __name__ == "__main__":
    create_dummy_dataset()