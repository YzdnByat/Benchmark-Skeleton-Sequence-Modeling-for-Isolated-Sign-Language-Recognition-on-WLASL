import os
import cv2
import urllib.request
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Download MediaPipe task models if not present
if not os.path.exists('pose_landmarker.task'):
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
        'pose_landmarker.task')
if not os.path.exists('hand_landmarker.task'):
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        'hand_landmarker.task')

pose_detector = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path='pose_landmarker.task', delegate=python.BaseOptions.Delegate.CPU)
))
hand_detector = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path='hand_landmarker.task', delegate=python.BaseOptions.Delegate.CPU),
    num_hands=2
))


def extract_keypoints(pose_res, hand_res):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     pose_res.pose_landmarks[0]]).flatten() if pose_res.pose_landmarks else np.zeros(33 * 4)
    lh, rh = np.zeros(21 * 3), np.zeros(21 * 3)
    if hand_res.hand_landmarks:
        for idx, hand_info in enumerate(hand_res.handedness):
            landmarks = np.array([[res.x, res.y, res.z] for res in hand_res.hand_landmarks[idx]]).flatten()
            if hand_info[0].category_name == 'Left':
                lh = landmarks
            elif hand_info[0].category_name == 'Right':
                rh = landmarks
    return np.concatenate([pose, lh, rh])


def extract_landmarks_from_video(video_path):
    """
    Reads a video file and extracts frame-by-frame raw MediaPipe keypoints (258 features/frame).
    """
    cap = cv2.VideoCapture(video_path)
    frames_data = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        frames_data.append(extract_keypoints(pose_detector.detect(mp_image), hand_detector.detect(mp_image)))
    cap.release()

    return np.array(frames_data)