# ==========================================
# SRC/DATASET.PY - FULL DATASET (ALL CLASSES)
# ==========================================
import glob
import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import train_test_split


class AugmentedFrameDataset(Dataset):
    def __init__(self, video_ids, labels, is_train=False, frames_dir="./extracted_frames_yolo", num_frames=32):
        self.video_ids = video_ids
        self.labels = labels
        self.frames_dir = frames_dir
        self.num_frames = num_frames

        if is_train:
            self.transform = T.Compose([
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid_id = self.video_ids[idx]
        label = self.labels[idx]
        folder_path = os.path.join(self.frames_dir, vid_id)

        frame_tensors = []
        for i in range(self.num_frames):
            img_path = os.path.join(folder_path, f"frame_{i:02d}.jpg")
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                frame_tensors.append(self.transform(img))
            else:
                frame_tensors.append(torch.zeros((3, 224, 224)))

        video_tensor = torch.stack(frame_tensors, dim=1)
        return video_tensor, torch.tensor(label, dtype=torch.long)


def build_dataloaders(config: dict):
    """Parses WLASL classes and assigns train/val/test using the official WLASL JSON splits."""
    wlasl_json_path = config["wlasl_json_path"]
    frames_dir = config["extracted_frames_dir"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    num_frames = config["num_frames"]

    with open(wlasl_json_path, 'r', encoding='utf-8') as f:
        wlasl_data = json.load(f)

    # 1. Automatically build gloss-to-index mapping for ALL unique glosses
    all_glosses = sorted(list(set(entry['gloss'] for entry in wlasl_data)))
    class_mapping = {gloss: idx for idx, gloss in enumerate(all_glosses)}
    config["num_classes"] = len(all_glosses)

    # 2. Extract video IDs categorized by official WLASL split
    train_vids, train_labels = [], []
    val_vids, val_labels = [], []
    test_vids, test_labels = [], []

    extracted_folders = set(
        f for f in os.listdir(frames_dir) if os.path.isdir(os.path.join(frames_dir, f))
    )

    for entry in wlasl_data:
        gloss = entry['gloss']
        label = class_mapping[gloss]
        for inst in entry['instances']:
            vid_id = str(inst['video_id']).strip()
            split_type = inst.get('split', 'train').lower()

            # Match against extracted YOLO frame folders
            matched_id = None
            if vid_id in extracted_folders:
                matched_id = vid_id
            elif vid_id.zfill(5) in extracted_folders:
                matched_id = vid_id.zfill(5)

            if matched_id:
                if split_type == 'train':
                    train_vids.append(matched_id)
                    train_labels.append(label)
                elif split_type == 'val':
                    val_vids.append(matched_id)
                    val_labels.append(label)
                elif split_type == 'test':
                    test_vids.append(matched_id)
                    test_labels.append(label)

    print(f"🎯 Loaded Official WLASL Splits | Train: {len(train_vids)} | Val: {len(val_vids)} | Test: {len(test_vids)}")

    # 3. Create PyTorch Datasets & DataLoaders
    train_dataset = AugmentedFrameDataset(train_vids, train_labels, is_train=True, frames_dir=frames_dir, num_frames=num_frames)
    val_dataset = AugmentedFrameDataset(val_vids, val_labels, is_train=False, frames_dir=frames_dir, num_frames=num_frames)
    test_dataset = AugmentedFrameDataset(test_vids, test_labels, is_train=False, frames_dir=frames_dir, num_frames=num_frames)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, config