import cv2
import math
import torch
import shutil
import os
import numpy
from detector import Detector
from ultralytics import YOLO

class AnalyzeAnimalGPU():

    def __init__(self, path_to_detector, path_to_clip, behavior_interval):
        self.detector = None
        self.path_to_detector = path_to_detector
        self.path_to_clip = path_to_clip
        self.behavior_interval = behavior_interval
        self.path_to_yolo_folder = "/Users/jackmcclure/Desktop/pip_LabGym_version_control/Mouse Classifier.v1i.coco-segmentation"
    
    def get_detector(self):
        """Load the detector."""
        self.detector = Detector()
        # Select animal name from saved model parameters, in current LabGym UI
        self.detector.load(path_to_detector=self.path_to_detector, animal_kinds=["mouse"])
    
    def apply_detector(self, behavior_intervals):
        interval_lengths = []
        for interval in behavior_intervals:
            interval_lengths.append(len(interval))

        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda")
        print(f"Using device: {device}")

        # Need to feed a list of [{"image": image1}, {"image": image2}, ...] into detectron
        self.get_detector()
        tensor_dict_list = []
        for interval in behavior_intervals:
            for frame in interval:
                tensor_frame = torch.as_tensor(frame.astype("float32").transpose(2, 0, 1))
                tensor_dict_list.append({"image":tensor_frame.to(device)})
        print("Tensored images")
        
        detected_frames = self.detector.inference(tensor_dict_list)

        print("Detected images")

        detected_intervals = []
        i = 0
        for length in interval_lengths:
            detected_interval = []
            for k in range(length):
                detected_interval.append(detected_frames[i+k])
            detected_intervals.append(detected_interval)
            i += length
        
        image_intervals = []
        for i in range(len(detected_intervals)):
            image_interval = []
            for k in range(len(detected_intervals[i])):
                original_frame = behavior_intervals[i][k]
                detectron_output = detected_intervals[i][k]
                instances = detectron_output["instances"].to("cpu")
                if len(instances) == 0:
                    # If detectron didn't find anything
                    continue
                # Just take the first detected instance (highest confidence)
                box = instances.pred_boxes.tensor[0].numpy().astype(int)  # [x1, y1, x2, y2]

                x1, y1, x2, y2 = box
                cropped_frame = original_frame[y1:y2, x1:x2]  # Crop using the box

                cv2.imshow("Animal", cropped_frame)
                cv2.waitKey(0)
                image_interval.append(cropped_frame)
            image_intervals.append(image_interval)
        return image_intervals
    
    def train_yolo_model(self):
        model = YOLO("yolo11x-seg.pt")

        dataset_yaml = None

        for filename in os.listdir(self.path_to_yolo_folder):
            if filename.endswith(".yaml"):
                dataset_yaml = os.path.join(self.path_to_yolo_folder, filename)
                model_name = os.path.splitext(filename)[0]
                print("Found YAML file:", filename)

        model.train(
            data=dataset_yaml,
            epochs=100,
            imgsz=640,
            batch=16,
            device=0,  # GPU id, or "cpu"
            workers=4,
            name=model_name
        )

        shutil.copy("runs/segment/"+model_name+"/weights/best.pt", self.path_to_yolo_folder+"/"+model_name+".pt")

    def load_model(self):
        for filename in os.listdir(self.path_to_yolo_folder):
            if filename.endswith(".pt"):
                model_file_path = os.path.join(self.path_to_yolo_folder, filename)
                print("Found PT weights file:", filename)
        model = YOLO(model_file_path)

        return model
    
    def apply_yolo_model(self, behavior_intervals):
        interval_lengths = []
        for interval in behavior_intervals:
            interval_lengths.append(len(interval))

        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda")
        print(f"Using device: {device}")

        # Need to feed a list of [{"image": image1}, {"image": image2}, ...] into detectron
        numpy_frames = []
        for interval in behavior_intervals:
            for frame in interval:
                numpy_frame = frame.astype("uint8")
                numpy_frames.append(numpy_frame)
        print("Numpy uint8-ed images")

        model = self.load_model()
        yoloed_frames = model.predict(source=numpy_frames, imgsz=640, device=0, verbose=False)
        print("Yoloed frames")

        yoloed_intervals = []
        i = 0
        for length in interval_lengths:
            yoloed_interval = []
            for k in range(length):
                yoloed_interval.append(yoloed_frames[i+k])
            yoloed_intervals.append(yoloed_interval)
            i += length
        
        

        

    def pre_process(self):
        """Grab frames of each behavior interval and run through detectron."""
        cap = cv2.VideoCapture(self.path_to_clip)
        if not cap.isOpened():
            print("Error: Could not open clip for pre-processing.")
            exit(1)
        fps = cap.get(cv2.CAP_PROP_FPS)  # Automatically detects FPS
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        framesize = min(width, height)

        frames_per_interval = math.ceil(fps * float(self.behavior_interval)/1000.0)

        print(f"FPS: {fps}, BI: {self.behavior_interval}, frames per behavior interval: {frames_per_interval}")
        print(f"Frame size: {int(width)} x {int(height)}")

        behavior_intervals = []
        current_behavior_interval = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_behavior_interval.append(frame)

            if len(current_behavior_interval) == frames_per_interval:
                behavior_intervals.append(current_behavior_interval)
                current_behavior_interval = []
        print("Captured frames")

        cap.release()
        
        # image_intervals = self.apply_detector(behavior_intervals)