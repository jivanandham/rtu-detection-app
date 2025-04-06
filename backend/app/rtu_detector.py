import torch
from ultralytics import YOLO
import cv2
import numpy as np
from typing import Dict
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RTUDetector:
    def __init__(self):
        logger.info("Initializing RTU Detector...")
        # Get the absolute path to the weights directory
        weights_dir = Path(__file__).parent.parent / "weights"
        model_path = weights_dir / "Yolov11.pt"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found at {model_path}. Please place your YOLOv11 model weights in the backend/weights directory.")

        try:
            # Load YOLOv11 model
            self.model = YOLO(str(model_path))
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            logger.info(f"Successfully loaded model from {model_path} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise Exception(f"Failed to load model: {str(e)}")

    def detect(self, image_path: str) -> Dict:
        try:
            logger.info(f"Processing image: {image_path}")
            # Validate image file exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # Perform inference
            results = self.model(image_path, device=self.device)
            result = results[0]
            
            # Extract boxes and scores
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            
            # Filter by confidence threshold
            conf_thresh = 0.5
            keep = scores > conf_thresh
            boxes = boxes[keep]
            scores = scores[keep]
            
            # Count detections
            rtu_count = len(boxes)
            logger.info(f"Found {rtu_count} RTUs with confidence > {conf_thresh}")
            
            # Process image for visualization
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to read image file: {image_path}")

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                conf = scores[i]
                
                # Draw rectangle
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add confidence text
                label = f"RTU: {conf:.2f}"
                cv2.putText(image, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                           (0, 255, 0), 2)
            
            # Add detection count
            cv2.putText(image, f'RTU Count: {rtu_count}', (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Save processed image
            processed_dir = Path(image_path).parent / "processed"
            processed_dir.mkdir(exist_ok=True)
            processed_filename = Path(image_path).name
            processed_path = processed_dir / processed_filename
            cv2.imwrite(str(processed_path), image)
            logger.info(f"Saved processed image to: {processed_path}")
            
            # Get the full URL path
            processed_url = f"/uploads/processed/{processed_filename}"
            
            return {
                "rtu_count": rtu_count,
                "address": "Address extraction not implemented yet",
                "processed_image": processed_url
            }
            
        except Exception as e:
            logger.error(f"Detection failed: {str(e)}")
            raise Exception(f"Detection failed: {str(e)}")

    def _extract_address(self, img) -> str:
        # TODO: Implement address extraction
        return "Address extraction not implemented yet"
