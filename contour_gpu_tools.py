import os
import shutil
import cv2
import numpy as np
import torch
from ultralytics import YOLO

def train_yolo_model(path_to_yolo_folder):
    model = YOLO("yolo11x-seg.pt")

    dataset_yaml = None

    for filename in os.listdir(path_to_yolo_folder):
        if filename.endswith(".yaml"):
            dataset_yaml = os.path.join(path_to_yolo_folder, filename)
            model_name = os.path.splitext(filename)[0]
            print("Found YAML file:", filename)

    if torch.cuda.is_available():
        print("GPU found:", torch.cuda.get_device_name(0))
        device = 0  # or device = "cuda"
    else:
        print("No GPU found, using CPU")
        device = "cpu"

    model.train(
        data=dataset_yaml,
        epochs=100,
        imgsz=640,
        batch=16,
        device=device,  # GPU id, or "cpu"
        workers=4,
        name=model_name
    )

    shutil.copy("runs/segment/"+model_name+"/weights/best.pt", path_to_yolo_folder+"/"+model_name+".pt")

def load_yolo_model(path_to_yolo_folder):
    for filename in os.listdir(path_to_yolo_folder):
        if filename.endswith(".pt"):
            model_file_path = os.path.join(path_to_yolo_folder, filename)
            print("Found PT weights file:", filename)
            model = YOLO(model_file_path)
            return model

    return False
    
def apply_yolo_model(self, frames):
    model = self.load_model()
    yoloed_frames = model.predict(source=frames, imgsz=640, device=0, verbose=False)
    print("Yoloed frames")
    frame_contours = []

    for idx, frame in enumerate(yoloed_frames):
        frame_contours = []
        frame = frames[idx].copy()
        frame_contours = []
        if frame.masks is not None:
            masks = frame.masks.data  # shape: [num_instances, H, W]
            for mask in masks:
                # Convert to uint8 binary mask
                mask_np = mask.cpu().numpy().astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                frame_contours.append(contours)

                # For testing
                cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
        
        # For testing
        cv2.imshow(f"Frame {idx}", frame)
        cv2.waitKey(0)  # Wait for key press to proceed
        cv2.destroyAllWindows()

        frame_contours.append(frame_contours)
    

    return frame_contours
