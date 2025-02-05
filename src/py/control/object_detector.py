import torch
from torch.amp import autocast
import cv2
import os
import pathlib

from py.control.model_type import ModelType

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

import sys
sys.path.append(r'src/py/yolov5')


class ObjectDetector:
    script_dir = os.path.dirname(__file__)
    yolov5_path = os.path.normpath(os.path.join(script_dir, '../../py/yolov5'))

    sys.path.append(yolov5_path)

    def __init__(self, model_type: ModelType):  # Lowered confidence threshold
        self.model_type = model_type

        if self.model_type == ModelType.PISTOL:
            model_path = os.path.normpath(os.path.join(self.script_dir, '../../resources/pistol.pt'))
            self.confidence_threshold = 0.6
        elif self.model_type == ModelType.COCO:
            model_path = os.path.normpath(os.path.join(self.script_dir, '../../resources/yolov5s.pt'))
            self.confidence_threshold = 0.4
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.model = torch.hub.load(
            ObjectDetector.yolov5_path,
            'custom',
            path=model_path,
            source='local'
        )

        self.stable_detections = None
        self.tracking_threshold = 3
        self.frame_stability_count = 0

        if self.model_type == ModelType.PISTOL:
            self.model.names = {0: 'Pistol'}

    def detect(self, frame):
        try:
            frame_height, frame_width = frame.shape[:2]
            frame_center_x = frame_width / 2
            frame_center_y = frame_height / 2

            with autocast(device_type='cuda', dtype=torch.float16):
                results = self.model(frame)
            detections = results.xyxy[0]

            # Filter based on confidence threshold
            conf_detections = detections[detections[:, 4] >= self.confidence_threshold]

            target_info = None
            if len(conf_detections) > 0:
                self.frame_stability_count = 0
                self.stable_detections = conf_detections

                # For pistol model, get highest confidence detection
                if self.model_type == ModelType.PISTOL:
                    best_detection = conf_detections[torch.argmax(conf_detections[:, 4])]

                    target_x = (best_detection[0] + best_detection[2]) / 2
                    target_y = (best_detection[1] + best_detection[3]) / 2

                    x_offset = (target_x - frame_center_x) / frame_width
                    y_offset = (target_y - frame_center_y) / frame_height

                    target_info = {
                        'x_offset': float(x_offset),
                        'y_offset': float(y_offset),
                        'confidence': float(best_detection[4])
                    }

                # Draw all detections
                self._draw_detections(frame, conf_detections)

                # Draw crosshair for pistol target
                if target_info and self.model_type == ModelType.PISTOL:
                    center_x = int(target_x)
                    center_y = int(target_y)
                    cv2.drawMarker(frame, (center_x, center_y), (0, 0, 255),
                                   cv2.MARKER_CROSS, 20, 2)
            else:
                self.frame_stability_count += 1

        except Exception as e:
            print(f"Detection error: {e}")
            target_info = None

        return frame, target_info

    def _draw_detections(self, frame, detections):
        for *xyxy, conf, cls in detections:
            label = f'{self.model.names[int(cls)]} {conf:.2f}'
            cv2.rectangle(frame,
                          (int(xyxy[0]), int(xyxy[1])),
                          (int(xyxy[2]), int(xyxy[3])),
                          (0, 255, 0), 2)
            cv2.putText(frame, label,
                        (int(xyxy[0]), int(xyxy[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)