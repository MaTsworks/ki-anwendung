import torch
from torch.amp import autocast
import cv2
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

import sys
sys.path.append(r'/yolov5')

class ObjectDetector:
    def __init__(self, model_path=r'C:\Users\3merz\PycharmProjects\ki-anwendung\src\control\pistol.pt', confidence_threshold=0.6):
        self.model = torch.hub.load(r'/yolov5', 'custom', path=model_path, source='local')
        self.confidence_threshold = confidence_threshold
        self.stable_detections = None
        self.tracking_threshold = 3  # Frames to keep detections stable
        self.frame_stability_count = 0
        self.model.names = {0: 'Pistol'}

    def detect(self, frame):
        try:
            with autocast(device_type='cuda', dtype=torch.float16):
                results = self.model(frame)
            detections = results.xyxy[0]

            # Filter high confidence detections
            high_conf_detections = detections[detections[:, 4] >= self.confidence_threshold]

            if len(high_conf_detections) > 0:
                # Reset stability counter and update detections
                self.frame_stability_count = 0
                self.stable_detections = high_conf_detections
            else:
                # Increment stability counter
                self.frame_stability_count += 1

            # Draw detections only if we have stable detections
            if self.stable_detections is not None and self.frame_stability_count < self.tracking_threshold:
                self._draw_detections(frame, self.stable_detections)

        except Exception as e:
            print(f"Detection error: {e}")

        return frame, self.stable_detections

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