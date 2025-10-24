import os
import shutil
import cv2
# import torch
from ultralytics import YOLO
from contour_gpu_tools import train_yolo_model, load_yolo_model, apply_yolo_model

path_to_yolo_folder = '/Users/jackmcclure/Desktop/pip_LabGym_version_control/Mouse Classifier.v1i.yolov11'

try_load = load_yolo_model(path_to_yolo_folder)
if not try_load:
    print("No yolo model found, training model")
    train_yolo_model(path_to_yolo_folder)
    model = load_yolo_model(path_to_yolo_folder)
    if not model:
        print("Failed to load freshly trained model")

path_to_test_frames = '/Users/jackmcclure/Desktop/pip_LabGym_version_control/Mouse Classifier.v1i.yolov11/test_frames'

test_frames = []
for filename in sorted(os.listdir(path_to_test_frames)):
    if filename.endswith(".jpg"):
        img_path = os.path.join(path_to_test_frames, filename)
        image = cv2.imread(img_path)
        if image is not None:
            test_frames.append(image)

apply_yolo_model(test_frames)