# ==========================================
# SCRIPTS/EXTRACT_YOLO.PY - FRAME EXTRACTION & CROPPING
# ==========================================
import os
import glob
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO


def extract_frames_yolo(
        video_dir="./videos",
        output_dir="./extracted_frames_yolo",
        num_frames=32,
        target_size=(224, 224),
        yolo_model_path="yolov8n.pt"
):
    """
    Extracts uniformly sampled frames from MP4 videos, applies YOLO detection
    to crop the main subject, resizes to target_size, and saves images.
    """
    os.makedirs(output_dir, exist_ok=True)
    video_files = glob.glob(os.path.join(video_dir, "*.mp4")) + glob.glob(os.path.join(video_dir, "*.avi"))

    print(f"🎬 Found {len(video_files)} videos in {video_dir}")
    if len(video_files) == 0:
        print(f"⚠️ No video files found in {video_dir}. Please place sample videos there.")
        return

    # Load YOLO Model
    print(f"🤖 Loading YOLO model ({yolo_model_path})...")
    model = YOLO(yolo_model_path)

    for vid_path in tqdm(video_files, desc="Processing Videos"):
        vid_name = os.path.splitext(os.path.basename(vid_path))[0]
        vid_out_dir = os.path.join(output_dir, vid_name)

        # Skip if already processed
        if os.path.exists(vid_out_dir) and len(os.listdir(vid_out_dir)) == num_frames:
            continue

        os.makedirs(vid_out_dir, exist_ok=True)
        cap = cv2.VideoCapture(vid_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            continue

        # Uniform Temporal Sampling (Exactly num_frames)
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        frame_idx = 0
        saved_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in frame_indices:
                # Convert BGR (cv2) to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # YOLO Person/Subject Bounding Box Detection
                results = model(rgb_frame, verbose=False)
                boxes = results[0].boxes

                crop_rgb = rgb_frame
                if len(boxes) > 0:
                    # Filter for 'person' class (cls == 0) if present, else pick largest box
                    person_boxes = [b for b in boxes if int(b.cls[0]) == 0]
                    target_box = person_boxes[0] if len(person_boxes) > 0 else boxes[0]

                    x1, y1, x2, y2 = map(int, target_box.xyxy[0].tolist())
                    h, w, _ = rgb_frame.shape
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    if (x2 - x1) > 10 and (y2 - y1) > 10:
                        crop_rgb = rgb_frame[y1:y2, x1:x2]

                # Resize cropped frame to target size (224x224)
                img = Image.fromarray(crop_rgb).resize(target_size, Image.Resampling.BILINEAR)
                save_path = os.path.join(vid_out_dir, f"frame_{saved_count:02d}.jpg")
                img.save(save_path)
                saved_count += 1

            frame_idx += 1

        cap.release()

    print(f"✅ Frame extraction complete! Images saved to: {output_dir}")


if __name__ == "__main__":
    extract_frames_yolo()