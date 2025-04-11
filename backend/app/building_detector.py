import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
from ultralytics import YOLO
from datetime import datetime
from geopy.geocoders import Nominatim
import json
import csv
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List
import logging
from concurrent.futures import ThreadPoolExecutor
import time
from roboflow import Roboflow
from inference_sdk import InferenceHTTPClient

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# === CONFIG ===
GOOGLE_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
ROBOFLOW_WORKSPACE = os.environ.get("ROBOFLOW_WORKSPACE", "")
ROBOFLOW_MODEL_ID = os.environ.get("ROBOFLOW_MODEL_ID", "")
RTU_MODEL_PATH = os.environ.get("RTU_MODEL_PATH", "models/Yolov11.pt")  # Adjust this path as needed
size_px = 640
initial_zoom = 19

# === FOLDER STRUCTURE ===
class BuildingDetector:
    def __init__(self):
        """Initialize the building detector"""
        # Load environment variables
        load_dotenv()
        
        # Initialize folders
        base_dir = Path(os.getenv("OUTPUT_DIR", "output"))
        self.folders = {
            "initial": base_dir / "initial",
            "deskewed": base_dir / "deskewed",
            "refined_crop": base_dir / "refined_crop",
            "rtu_detected": base_dir / "rtu_detected"
        }
        
        # Create folders if they don't exist
        for folder in self.folders.values():
            folder.mkdir(parents=True, exist_ok=True)
            
        # Initialize CSV file
        self.csv_path = base_dir / "detection_results.csv"
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["building_id", "latitude", "longitude", "address", "area_sqft", "rtu_count"])
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(base_dir / "detection.log")
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Initialize geolocator
        self.geolocator = Nominatim(user_agent="building_detector")
        
        # Initialize RTU model - use local YOLOv11 model
        project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = RTU_MODEL_PATH
        if not Path(model_path).exists():
            model_path = project_root / "weights" / "Yolov11.pt"
            
        if not Path(model_path).exists():
            self.logger.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        self.rtu_model = YOLO(str(model_path))
        self.logger.info(f"Loaded YOLOv11 model from {model_path}")
        
        # Initialize Roboflow client
        try:
            api_key = ROBOFLOW_API_KEY
            workspace = ROBOFLOW_WORKSPACE
            model_id = ROBOFLOW_MODEL_ID
            
            if not api_key:
                raise ValueError("ROBOFLOW_API_KEY not set in environment variables")
                
            if not workspace:
                raise ValueError("ROBOFLOW_WORKSPACE not set in environment variables")
                
            if not model_id:
                raise ValueError("ROBOFLOW_MODEL_ID not set in environment variables")
                
            self.logger.info(f"Initializing Roboflow with workspace: {workspace}, model: {model_id}")
            
            self.rf = Roboflow(api_key=api_key)
            self.project = self.rf.workspace(workspace).project(model_id.split('/')[0])
            self.model = self.project.version(int(model_id.split('/')[1])).model
            self.logger.info(f"Roboflow initialized successfully for project {self.project.id}")
            
            # Initialize inference client
            self.inference_client = InferenceHTTPClient(
                api_url="https://detect.roboflow.com",
                api_key=api_key
            )
            
        except Exception as e:
            self.logger.warning(f"Could not initialize Roboflow: {str(e)}")
            self.rf = None
            self.project = None
            self.model = None
            self.inference_client = None
        
        self.logger.info("Building detector initialized successfully")

    def zip_to_bounding_box(self, zipcode):
        """Convert a ZIP code to a bounding box using OpenStreetMap Nominatim API"""
        try:
            url = f"https://nominatim.openstreetmap.org/search?postalcode={zipcode}&country=USA&format=json&limit=1"
            headers = {"User-Agent": "rtu_detector"}
            response = requests.get(url, headers=headers)
            
            if response.status_code != 200:
                logger.error(f"Failed to get bounding box for ZIP {zipcode}: {response.status_code}")
                return None
                
            data = response.json()
            if not data:
                logger.error(f"No results found for ZIP {zipcode}")
                return None
                
            # Extract bounding box
            bbox = data[0]["boundingbox"]
            bounds = {
                "south": float(bbox[0]),
                "north": float(bbox[1]),
                "west": float(bbox[2]),
                "east": float(bbox[3])
            }
            
            logger.info(f"ZIP {zipcode} converted to bounds: {bounds}")
            return bounds
            
        except Exception as e:
            logger.error(f"Error converting ZIP to bounding box: {str(e)}")
            return None

    def fetch_buildings(self, bounds):
        """Fetch building coordinates within the bounding box using Overpass API"""
        try:
            # Construct Overpass API query
            overpass_url = "https://overpass-api.de/api/interpreter"
            query = f"""
            [out:json];
            (
              way["building"]({bounds['south']},{bounds['west']},{bounds['north']},{bounds['east']});
              relation["building"]({bounds['south']},{bounds['west']},{bounds['north']},{bounds['east']});
            );
            out center;
            """
            
            response = requests.post(overpass_url, data={"data": query})
            
            if response.status_code != 200:
                logger.error(f"Failed to fetch buildings: {response.status_code}")
                return []
                
            data = response.json()
            buildings = []
            
            for element in data.get("elements", []):
                if "center" in element:
                    buildings.append({
                        "id": element["id"],
                        "lat": element["center"]["lat"],
                        "lng": element["center"]["lon"]
                    })
            
            logger.info(f"Found {len(buildings)} buildings in the bounding box")
            return buildings
            
        except Exception as e:
            logger.error(f"Error fetching buildings: {str(e)}")
            return []

    def fetch_satellite_image(self, lat, lng, zoom=19):
        """Fetch satellite image from Google Maps Static API"""
        try:
            api_key = GOOGLE_API_KEY
            if not api_key:
                raise ValueError("GOOGLE_MAPS_API_KEY not set in environment variables")
                
            params = {
                "center": f"{lat},{lng}",
                "zoom": zoom,
                "size": "640x640",
                "maptype": "satellite",
                "key": api_key
            }
            
            self.logger.debug(f"Fetching satellite image with params: {params}")
            
            response = requests.get(
                "https://maps.googleapis.com/maps/api/staticmap",
                params=params
            )
            
            if response.status_code != 200:
                self.logger.error(f"Failed to fetch image: {response.status_code}")
                return None
                
            img = Image.open(BytesIO(response.content))
            self.logger.info(f"Successfully fetched satellite image of size {img.size}")
            return img
            
        except Exception as e:
            self.logger.error(f"Error fetching satellite image: {str(e)}", exc_info=True)
            raise

    def reverse_geocode(self, lat, lng):
        """Convert coordinates to address using OpenStreetMap Nominatim API"""
        try:
            location = self.geolocator.reverse((lat, lng), timeout=10)
            return location.address if location else "Unknown Address"
        except Exception as e:
            self.logger.error(f"Error in reverse geocoding: {str(e)}")
            return "Geocoding Failed"

    def auto_rotate_crop(self, img, polygon_np, output_path):
        """Deskew and crop the image based on the polygon"""
        try:
            img_cv = np.array(img)[:, :, ::-1]
            if polygon_np.shape[0] < 3:
                return None, 0
                
            contour = np.int32([polygon_np])
            rect = cv2.minAreaRect(contour)
            angle = rect[2]
            if angle < -45:
                angle += 90
                
            center = tuple(np.array(img_cv.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img_cv, rot_mat, img_cv.shape[1::-1], flags=cv2.INTER_LINEAR)
            
            points = np.array(polygon_np, dtype=np.float32).reshape(-1, 1, 2)
            rotated_contour = cv2.transform(points, rot_mat)
            x, y, w, h = cv2.boundingRect(rotated_contour)
            
            if w <= 0 or h <= 0 or x < 0 or y < 0:
                return None, angle
                
            cropped = rotated[y:y+h, x:x+w]
            cropped_pil = Image.fromarray(cropped[:, :, ::-1])
            cropped_pil.save(output_path)
            
            return cropped_pil, angle
            
        except Exception as e:
            self.logger.error(f"Error in auto_rotate_crop: {str(e)}")
            return None, 0

    def calculate_area_in_sq_meters(self, polygon_np):
        """Calculate building area in square meters"""
        try:
            if polygon_np.shape[0] < 3:
                return 0.0
                
            meters_per_pixel = 0.3
            area_pixels = 0.5 * np.abs(
                np.dot(polygon_np[:, 0], np.roll(polygon_np[:, 1], 1)) - 
                np.dot(polygon_np[:, 1], np.roll(polygon_np[:, 0], 1))
            )
            return area_pixels * (meters_per_pixel ** 2)
            
        except Exception as e:
            self.logger.error(f"Error calculating area: {str(e)}")
            return 0.0

    def visualize_detection(self, img, polygon_np, output_path):
        """Create a visualization of the detection"""
        try:
            img_vis = img.copy()
            draw = ImageDraw.Draw(img_vis)
            
            # Draw polygon
            polygon = [(x, y) for x, y in polygon_np]
            draw.polygon(polygon, outline="red", width=2)
            
            # Draw center point
            center = np.mean(polygon_np, axis=0)
            draw.ellipse([center[0]-5, center[1]-5, center[0]+5, center[1]+5], fill="red")
            
            # Convert to RGB before saving
            img_vis = img_vis.convert('RGB')
            img_vis.save(output_path)
            
        except Exception as e:
            self.logger.error(f"Error in visualize_detection: {str(e)}")
            raise

    def detect_rooftop(self, image_path):
        """Detect rooftop using Roboflow model"""
        try:
            if not self.inference_client:
                self.logger.error("Roboflow inference client not initialized")
                return {"success": False, "error": "Roboflow inference client not initialized"}
                
            self.logger.debug(f"Detecting rooftop in image: {image_path}")
            
            # Send to Roboflow
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # Get predictions
            result = self.inference_client.infer(image_data, model_id=self.project.id)
            self.logger.debug(f"Roboflow response: {result}")
            
            if not result or not result.get("predictions"):
                self.logger.warning("No predictions from Roboflow")
                return {"success": False, "error": "No predictions from Roboflow"}
                
            # Get the first prediction (assuming there's only one rooftop)
            prediction = result["predictions"][0]
            self.logger.debug(f"Prediction details: {prediction}")
            
            # Get the polygon points
            polygon = prediction.get("points", [])
            if not polygon:
                self.logger.warning("No polygon points in prediction")
                return {"success": False, "error": "No polygon points in prediction"}
                
            # Convert to numpy array
            polygon_np = np.array([[p["x"], p["y"]] for p in polygon])
            
            return {
                "success": True,
                "polygon": polygon,
                "polygon_np": polygon_np,
                "confidence": prediction.get("confidence", 0.0)
            }
            
        except Exception as e:
            self.logger.error(f"Error in detect_rooftop: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}

    def detect_rtus(self, image_path, from_rtu_detector=None):
        """Detect RTUs in the image"""
        try:
            # If we have an RTU detector passed in, use it
            if from_rtu_detector:
                logger.info(f"Using provided RTU detector for {image_path}")
                result = from_rtu_detector.detect(image_path)
                return result
                
            # If no RTU detector was provided, use the YOLOv11 model directly
            logger.info(f"Loading YOLOv11 model for RTU detection on {image_path}")
            from ultralytics import YOLO
            
            # Load the model
            model_path = RTU_MODEL_PATH
            model = YOLO(model_path)
            
            # Run inference
            results = model(image_path, conf=0.5)
            
            # Process results
            detections = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = box.conf[0].item()
                    class_id = int(box.cls[0].item())
                    
                    detections.append({
                        "x": int(x1),
                        "y": int(y1),
                        "width": int(x2 - x1),
                        "height": int(y2 - y1),
                        "confidence": confidence,
                        "class_id": class_id
                    })
            
            # Create a copy of the image with RTU boxes drawn
            img = cv2.imread(image_path)
            output_img = img.copy()
            for det in detections:
                cv2.rectangle(output_img, 
                             (det["x"], det["y"]), 
                             (det["x"] + det["width"], det["y"] + det["height"]), 
                             (0, 255, 0), 2)
                
                # Add confidence text
                text = f"{det['confidence']:.2f}"
                cv2.putText(output_img, text, (det["x"], det["y"] - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Save the output image
            output_dir = Path(image_path).parent / "processed"
            output_dir.mkdir(exist_ok=True)
            
            output_path = output_dir / Path(image_path).name
            cv2.imwrite(str(output_path), output_img)
            
            logger.info(f"Detected {len(detections)} RTUs in {image_path}")
            
            return {
                "success": True,
                "rtu_count": len(detections),
                "rtus": detections,
                "processed_image": str(output_path)
            }
            
        except Exception as e:
            logger.error(f"Error detecting RTUs: {str(e)}")
            return {"success": False, "error": str(e)}

    def deskew_rooftop(self, image_path, polygon):
        """Deskew and crop the rooftop based on the detected polygon"""
        try:
            # Load the image
            img = cv2.imread(image_path)
            
            # Convert polygon to numpy array
            polygon_np = np.array(polygon, dtype=np.int32)
            
            # Get the rotated rectangle that bounds the polygon
            rect = cv2.minAreaRect(polygon_np)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Get width and height of the rectangle
            width = int(rect[1][0])
            height = int(rect[1][1])
            
            # Get the transformation matrix
            src_pts = box.astype("float32")
            dst_pts = np.array([[0, height-1],
                                [0, 0],
                                [width-1, 0],
                                [width-1, height-1]], dtype="float32")
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            
            # Apply the transformation
            warped = cv2.warpPerspective(img, M, (width, height))
            
            # Save the deskewed image
            output_dir = Path(image_path).parent / "deskewed"
            output_dir.mkdir(exist_ok=True)
            
            output_path = output_dir / Path(image_path).name
            cv2.imwrite(str(output_path), warped)
            
            return {
                "success": True,
                "deskewed_path": str(output_path),
                "width": width,
                "height": height
            }
            
        except Exception as e:
            logger.error(f"Error deskewing rooftop: {str(e)}")
            return {"success": False, "error": str(e)}

    def save_error_image(self, error_message, output_path):
        """Create an error image with the error message"""
        # Create a blank image with error message
        img = Image.new('RGB', (800, 400), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Add error text
        try:
            font = ImageFont.truetype("Arial", 20)
        except IOError:
            font = ImageFont.load_default()
        
        draw.text((50, 50), f"Error: {error_message}", fill=(255, 0, 0), font=font)
        
        # Save the error image
        try:
            img.save(output_path)
            logging.info(f"Saved error image to {output_path}")
        except Exception as e:
            logging.error(f"Failed to save error image: {e}")

    def save_debug_image(self, image, debug_info, output_path):
        """Save a debug image with information overlay"""
        # Convert to PIL Image if it's a numpy array
        if isinstance(image, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            img = image.copy()
        
        draw = ImageDraw.Draw(img)
        
        # Add debug text
        try:
            font = ImageFont.truetype("Arial", 16)
        except IOError:
            font = ImageFont.load_default()
        
        y_position = 10
        for key, value in debug_info.items():
            draw.text((10, y_position), f"{key}: {value}", fill=(255, 0, 0), font=font)
            y_position += 20
        
        # Save the debug image
        try:
            img.save(output_path)
            logging.info(f"Saved debug image to {output_path}")
        except Exception as e:
            logging.error(f"Failed to save debug image: {e}")

    def detect_rooftop(self, image_path):
        """Detect building rooftop in an image"""
        try:
            # For demonstration, we'll use a simple contour detection approach
            # In a real application, this would use a more sophisticated model
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {"success": False, "error": "Failed to load image"}
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold
            _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find the largest contour (assuming it's the building)
            if not contours:
                return {"success": False, "error": "No contours found"}
                
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Simplify the polygon
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # Convert to list of points
            polygon = [[int(p[0][0]), int(p[0][1])] for p in approx_polygon]
            
            # Draw the polygon on the image
            result_image = image.copy()
            cv2.polylines(result_image, [approx_polygon], True, (0, 255, 0), 2)
            
            # Add text
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(result_image, "Rooftop", (10, 30), font, 1, (0, 255, 0), 2)
            
            # Save the result
            result_path = image_path.replace(".jpg", "_rooftop.jpg").replace(".png", "_rooftop.png")
            cv2.imwrite(result_path, result_image)
            
            # Create a mask of the rooftop
            mask = np.zeros_like(gray)
            cv2.fillPoly(mask, [approx_polygon], 255)
            
            # Add a semi-transparent overlay
            overlay = image.copy()
            overlay[mask > 0] = [0, 255, 0]  # Green color for the rooftop
            
            # Blend the overlay with the original image
            alpha = 0.3
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, result_image)
            
            # Add a label with the area
            area_px = cv2.contourArea(largest_contour)
            area_sqft = self.pixels_to_sqft(area_px)
            
            # Create a label with a background
            label = f"Area: {area_sqft:.2f} sq ft"
            (text_width, text_height), _ = cv2.getTextSize(label, font, 0.7, 2)
            
            # Position the label at the top of the image with padding
            padding = 10
            text_offset_x = padding
            text_offset_y = text_height + padding
            
            # Draw the background rectangle
            cv2.rectangle(
                result_image, 
                (text_offset_x - 5, text_offset_y - text_height - 5), 
                (text_offset_x + text_width + 5, text_offset_y + 5), 
                (0, 0, 0), 
                -1
            )
            
            # Draw the text
            cv2.putText(
                result_image, 
                label, 
                (text_offset_x, text_offset_y), 
                font, 
                0.7, 
                (255, 255, 255), 
                2
            )
            
            # Save the result
            cv2.imwrite(result_path, result_image)
            
            return {
                "success": True,
                "polygon": polygon,
                "area_px": area_px,
                "area_sqft": area_sqft,
                "result_path": result_path
            }
            
        except Exception as e:
            logger.error(f"Error detecting rooftop: {str(e)}")
            return {"success": False, "error": str(e)}

    def detect_rtus(self, image_path, rtu_detector=None):
        """Detect RTUs on a rooftop image"""
        try:
            # If no RTU detector is provided, use a simple placeholder detection
            if rtu_detector is None or self.use_mock_data:
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    return {"success": False, "error": "Failed to load image"}
                
                # For demonstration, we'll place random rectangles as "RTUs"
                # In a real application, this would use the YOLOv11 model
                height, width = image.shape[:2]
                
                # Generate random RTUs
                import random
                random.seed(42)  # For reproducibility
                
                num_rtus = random.randint(3, 8)
                rtus = []
                
                for _ in range(num_rtus):
                    # Generate random RTU dimensions and position
                    rtu_width = random.randint(20, 50)
                    rtu_height = random.randint(20, 50)
                    
                    x = random.randint(0, width - rtu_width)
                    y = random.randint(0, height - rtu_height)
                    
                    rtus.append({
                        "x": x,
                        "y": y,
                        "width": rtu_width,
                        "height": rtu_height,
                        "confidence": random.uniform(0.7, 0.95)
                    })
                
                # Draw RTUs on the image
                result_image = image.copy()
                
                for i, rtu in enumerate(rtus):
                    # Draw rectangle
                    cv2.rectangle(
                        result_image,
                        (rtu["x"], rtu["y"]),
                        (rtu["x"] + rtu["width"], rtu["y"] + rtu["height"]),
                        (0, 0, 255),
                        2
                    )
                    
                    # Add label
                    label = f"RTU {i+1}: {rtu['confidence']:.2f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    
                    # Calculate label position
                    (text_width, text_height), _ = cv2.getTextSize(label, font, 0.5, 1)
                    
                    # Position the label at the top of the RTU
                    text_offset_x = rtu["x"]
                    text_offset_y = rtu["y"] - 5
                    
                    # Ensure the label is within the image bounds
                    if text_offset_y < text_height:
                        text_offset_y = rtu["y"] + rtu["height"] + text_height
                    
                    # Draw the background rectangle
                    cv2.rectangle(
                        result_image, 
                        (text_offset_x, text_offset_y - text_height), 
                        (text_offset_x + text_width, text_offset_y), 
                        (0, 0, 255), 
                        -1
                    )
                    
                    # Draw the text
                    cv2.putText(
                        result_image, 
                        label, 
                        (text_offset_x, text_offset_y - 2), 
                        font, 
                        0.5, 
                        (255, 255, 255), 
                        1
                    )
                
                # Add summary text
                summary = f"Total RTUs: {num_rtus}"
                (text_width, text_height), _ = cv2.getTextSize(summary, font, 0.7, 2)
                
                # Draw the background rectangle
                cv2.rectangle(
                    result_image, 
                    (10, 10), 
                    (10 + text_width, 10 + text_height), 
                    (0, 0, 0), 
                    -1
                )
                
                # Draw the text
                cv2.putText(
                    result_image, 
                    summary, 
                    (10, 10 + text_height - 5), 
                    font, 
                    0.7, 
                    (255, 255, 255), 
                    2
                )
                
                # Save the result
                result_path = image_path.replace(".jpg", "_rtus.jpg").replace(".png", "_rtus.png")
                cv2.imwrite(result_path, result_image)
                
                return {
                    "success": True,
                    "rtus": rtus,
                    "rtu_count": num_rtus,
                    "processed_image": result_path
                }
            else:
                # Use the provided RTU detector (YOLOv11)
                result = rtu_detector.detect(image_path)
                return {
                    "success": True,
                    "rtus": result["detections"],
                    "rtu_count": len(result["detections"]),
                    "processed_image": result["processed_image"]
                }
                
        except Exception as e:
            logger.error(f"Error detecting RTUs: {str(e)}")
            return {"success": False, "error": str(e)}

    def process_zipcode(self, zipcode, rtu_detector=None, max_buildings=20, db=None):
        """Process all buildings in a ZIP code"""
        try:
            # Step 1: Convert ZIP code to bounding box
            bounds = self.zip_to_bounding_box(zipcode)
            if not bounds:
                return {"success": False, "error": "Failed to get bounding box for ZIP code"}
            
            # Step 2: Fetch building coordinates
            buildings = self.fetch_buildings(bounds)
            if not buildings:
                return {"success": False, "error": "No buildings found in this ZIP code"}
            
            # Limit the number of buildings to process
            buildings = buildings[:max_buildings]
            
            results = []
            for i, building in enumerate(buildings):
                logger.info(f"Processing building {i+1}/{len(buildings)}: {building['id']}")
                
                # Step 3: Fetch satellite image
                image_path = self.fetch_satellite_image(building["lat"], building["lng"])
                if not image_path:
                    logger.error(f"Failed to fetch satellite image for building {building['id']}")
                    continue
                
                # Step 4: Get building address
                address = self.reverse_geocode(building["lat"], building["lng"])
                
                # Step 5: Detect rooftop
                rooftop_result = self.detect_rooftop(image_path)
                if not rooftop_result.get("success", False):
                    logger.error(f"Failed to detect rooftop for building {building['id']}")
                    continue
                
                # Step 6: Calculate rooftop area
                rooftop_area = self.calculate_area_in_sq_meters(rooftop_result["polygon_np"])
                
                # Step 7: Deskew rooftop
                deskew_result = self.deskew_rooftop(image_path, rooftop_result["polygon"])
                
                # Step 8: Detect RTUs
                rtu_image_path = deskew_result.get("deskewed_path", image_path) if deskew_result.get("success", False) else image_path
                rtu_result = self.detect_rtus(rtu_image_path, rtu_detector)
                
                # Step 9: Check if building meets criteria (>5 RTUs and >30,000 sq ft)
                rtu_count = rtu_result.get("rtu_count", 0) if rtu_result.get("success", False) else 0
                meets_criteria = (rtu_count > 5 and rooftop_area > 30000)
                
                # Step 10: Save results to CSV
                with open(self.csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        building["id"],
                        building["lat"],
                        building["lng"],
                        address,
                        f"{rooftop_area:.2f}",
                        rtu_count,
                        "Yes" if meets_criteria else "No"
                    ])
                
                # Add to results
                processed_image = rtu_result.get("processed_image", None) if rtu_result.get("success", False) else None
                result_dict = {
                    "building_id": str(building["id"]),
                    "lat": building["lat"],
                    "lng": building["lng"],
                    "address": address,
                    "rooftop_area_sqft": rooftop_area,
                    "rtu_count": rtu_count,
                    "meets_criteria": meets_criteria,
                    "original_image": image_path,
                    "processed_image": processed_image
                }
                results.append(result_dict)
                
                # Save to database if db session is provided
                if db:
                    # Create a new BuildingDetection record
                    db_building = BuildingDetection(
                        building_id=str(building["id"]),
                        lat=building["lat"],
                        lng=building["lng"],
                        address=address,
                        rooftop_area_sqft=rooftop_area,
                        rtu_count=rtu_count,
                        meets_criteria=meets_criteria,
                        original_image=image_path,
                        processed_image=processed_image,
                        search_type="zipcode",
                        search_query=zipcode,
                        detection_data={
                            "rooftop": rooftop_result,
                            "rtus": rtu_result.get("rtus", []) if rtu_result.get("success", False) else []
                        }
                    )
                    db.add(db_building)
                    db.commit()
                    db.refresh(db_building)
                    
                    # Update the result with the database ID
                    result_dict["id"] = db_building.id
            
            # Step 11: Return summary
            total_buildings = len(buildings)
            buildings_with_rtus = sum(1 for r in results if r["rtu_count"] > 0)
            buildings_meeting_criteria = sum(1 for r in results if r["meets_criteria"])
            
            return {
                "success": True,
                "zipcode": zipcode,
                "total_buildings_processed": total_buildings,
                "buildings_with_rtus": buildings_with_rtus,
                "buildings_meeting_criteria": buildings_meeting_criteria,
                "results": results,
                "csv_path": str(self.csv_path)
            }
            
        except Exception as e:
            logger.error(f"Error processing ZIP code: {str(e)}")
            return {"success": False, "error": f"Error processing ZIP code: {str(e)}"}

    def process_county(self, county, state, rtu_detector=None, max_buildings=20, db=None):
        """Process all buildings in a county"""
        try:
            # Step 1: Convert county to bounding box using Nominatim
            url = f"https://nominatim.openstreetmap.org/search?county={county}&state={state}&country=USA&format=json&limit=1"
            headers = {"User-Agent": "rtu_detector"}
            response = requests.get(url, headers=headers)
            
            if response.status_code != 200:
                logger.error(f"Failed to get bounding box for {county}, {state}: {response.status_code}")
                return {"success": False, "error": "Failed to get bounding box for county"}
                
            data = response.json()
            if not data:
                logger.error(f"No results found for {county}, {state}")
                return {"success": False, "error": "No results found for county"}
                
            # Extract bounding box
            bbox = data[0]["boundingbox"]
            bounds = {
                "south": float(bbox[0]),
                "north": float(bbox[1]),
                "west": float(bbox[2]),
                "east": float(bbox[3])
            }
            
            logger.info(f"{county}, {state} converted to bounds: {bounds}")
            
            # The rest of the processing is the same as for ZIP codes
            # Step 2: Fetch building coordinates
            buildings = self.fetch_buildings(bounds)
            if not buildings:
                return {"success": False, "error": "No buildings found in this county"}
            
            # Limit the number of buildings to process
            buildings = buildings[:max_buildings]
            
            results = []
            for i, building in enumerate(buildings):
                logger.info(f"Processing building {i+1}/{len(buildings)}: {building['id']}")
                
                # Step 3: Fetch satellite image
                image_path = self.fetch_satellite_image(building["lat"], building["lng"])
                if not image_path:
                    logger.error(f"Failed to fetch satellite image for building {building['id']}")
                    continue
                
                # Step 4: Get building address
                address = self.reverse_geocode(building["lat"], building["lng"])
                
                # Step 5: Detect rooftop
                rooftop_result = self.detect_rooftop(image_path)
                if not rooftop_result.get("success", False):
                    logger.error(f"Failed to detect rooftop for building {building['id']}")
                    continue
                
                # Step 6: Calculate rooftop area
                rooftop_area = self.calculate_area_in_sq_meters(rooftop_result["polygon_np"])
                
                # Step 7: Deskew rooftop
                deskew_result = self.deskew_rooftop(image_path, rooftop_result["polygon"])
                
                # Step 8: Detect RTUs
                rtu_image_path = deskew_result.get("deskewed_path", image_path) if deskew_result.get("success", False) else image_path
                rtu_result = self.detect_rtus(rtu_image_path, rtu_detector)
                
                # Step 9: Check if building meets criteria (>5 RTUs and >30,000 sq ft)
                rtu_count = rtu_result.get("rtu_count", 0) if rtu_result.get("success", False) else 0
                meets_criteria = (rtu_count > 5 and rooftop_area > 30000)
                
                # Step 10: Save results to CSV
                with open(self.csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        building["id"],
                        building["lat"],
                        building["lng"],
                        address,
                        f"{rooftop_area:.2f}",
                        rtu_count,
                        "Yes" if meets_criteria else "No"
                    ])
                
                # Add to results
                processed_image = rtu_result.get("processed_image", None) if rtu_result.get("success", False) else None
                result_dict = {
                    "building_id": str(building["id"]),
                    "lat": building["lat"],
                    "lng": building["lng"],
                    "address": address,
                    "rooftop_area_sqft": rooftop_area,
                    "rtu_count": rtu_count,
                    "meets_criteria": meets_criteria,
                    "original_image": image_path,
                    "processed_image": processed_image
                }
                results.append(result_dict)
                
                # Save to database if db session is provided
                if db:
                    # Create a new BuildingDetection record
                    search_query = f"{county}, {state}"
                    db_building = BuildingDetection(
                        building_id=str(building["id"]),
                        lat=building["lat"],
                        lng=building["lng"],
                        address=address,
                        rooftop_area_sqft=rooftop_area,
                        rtu_count=rtu_count,
                        meets_criteria=meets_criteria,
                        original_image=image_path,
                        processed_image=processed_image,
                        search_type="county",
                        search_query=search_query,
                        detection_data={
                            "rooftop": rooftop_result,
                            "rtus": rtu_result.get("rtus", []) if rtu_result.get("success", False) else []
                        }
                    )
                    db.add(db_building)
                    db.commit()
                    db.refresh(db_building)
                    
                    # Update the result with the database ID
                    result_dict["id"] = db_building.id
            
            # Step 11: Return summary
            total_buildings = len(buildings)
            buildings_with_rtus = sum(1 for r in results if r["rtu_count"] > 0)
            buildings_meeting_criteria = sum(1 for r in results if r["meets_criteria"])
            
            return {
                "success": True,
                "county": county,
                "state": state,
                "total_buildings_processed": total_buildings,
                "buildings_with_rtus": buildings_with_rtus,
                "buildings_meeting_criteria": buildings_meeting_criteria,
                "results": results,
                "csv_path": str(self.csv_path)
            }
            
        except Exception as e:
            logger.error(f"Error processing county: {str(e)}")
            return {"success": False, "error": f"Error processing county: {str(e)}"}

    def process_building(self, lat, lng, building_id):
        """Process a single building"""
        try:
            # Step 1: Fetch satellite image
            initial_img_path = self.folders["initial"] / f"{building_id}_initial.jpg"
            img = self.fetch_satellite_image(lat, lng, initial_zoom)
            if img is None:
                return None
                
            # Convert image to RGB before saving
            img = img.convert('RGB')
            img.save(initial_img_path)
            logger.info(f"Step 1: Initial image saved to {initial_img_path}")
            
            # Step 2: Get building address
            address = self.reverse_geocode(lat, lng)
            logger.info(f"Building location: {address}")
            
            # Step 3: Detect rooftop using Roboflow
            result = self.detect_rooftop(initial_img_path)
            if not result.get("success", False):
                logger.error(f"No rooftops found for {building_id}")
                return None
                
            polygon = result["polygon"]
            polygon_np = np.array(polygon)
            
            # Step 4: Deskew and crop the image
            deskewed_path = self.folders["deskewed"] / f"{building_id}_deskewed.jpg"
            deskewed_img, angle = self.auto_rotate_crop(img, polygon_np, deskewed_path)
            if deskewed_img is None:
                logger.error(f"Failed to deskew image for {building_id}")
                return None
                
            # Step 5: Get refined rooftop detection
            refined_result = self.detect_rooftop(deskewed_path)
            if not refined_result.get("success", False):
                logger.error(f"No rooftops found in deskewed image for {building_id}")
                return None
                
            refined_polygon = refined_result["polygon"]
            refined_polygon_np = np.array(refined_polygon, dtype=np.int32)
            
            # Step 6: Calculate building area
            building_area_m2 = self.calculate_area_in_sq_meters(refined_polygon_np)
            building_area_sqft = building_area_m2 * 10.7639
            
            # Step 7: Crop the refined rooftop
            x, y, w, h = cv2.boundingRect(refined_polygon_np)
            img_np = np.array(deskewed_img)
            cropped_np = img_np[y:y+h, x:x+w]
            cropped_img = Image.fromarray(cropped_np)
            
            # Step 8: Save cropped image
            final_crop_path = self.folders["refined_crop"] / f"{building_id}_final_crop.jpg"
            cropped_img = cropped_img.convert('RGB')
            cropped_img.save(final_crop_path)
            logger.info(f"Step 6: Final cropped image saved to {final_crop_path}")
            
            # Step 9: Detect RTUs
            results = self.rtu_model(final_crop_path)
            rtu_count = 0
            
            # Step 10: Draw RTU detections
            rtu_img = Image.open(final_crop_path).convert("RGB")
            draw = ImageDraw.Draw(rtu_img)
            
            for result in results:
                for box in result.boxes:
                    cls_name = result.names[int(box.cls)]
                    if cls_name.lower() in ["rtu", "rooftop_unit"]:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
                        draw.text((x1, y1 - 10), cls_name, fill="green")
                        rtu_count += 1
            
            # Step 11: Save RTU detection image if criteria met
            if rtu_count > 5 and building_area_sqft > 30000:
                rtu_output_path = self.folders["rtu_detected"] / f"{building_id}_rtu_detected.jpg"
                rtu_img.save(rtu_output_path)
                logger.info(f"{rtu_count} RTUs detected for {building_id}. Meets criteria with {building_area_sqft:.2f} sq.ft")
                
                # Save to CSV
                with open(self.csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        building_id,
                        lat,
                        lng,
                        address,
                        f"{building_area_sqft:.2f}",
                        rtu_count
                    ])
                
                return {
                    "building_id": building_id,
                    "latitude": lat,
                    "longitude": lng,
                    "address": address,
                    "area_sqft": round(building_area_sqft, 2),
                    "rtu_count": rtu_count
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing building {building_id}: {str(e)}")
            raise

    def run_and_log_building(self, lat, lng, building_id):
        """Process a building and log results"""
        try:
            result = self.process_building(lat, lng, building_id)
            if result:
                return result
        except Exception as e:
            logger.error(f"Error processing building {building_id}: {str(e)}")
            return None

    def process_buildings(self, buildings, max_buildings=None):
        """Process multiple buildings with queue management"""
        try:
            total_buildings = len(buildings)
            processed_count = 0
            
            # Create a queue for processing
            queue = []
            results = []
            
            for building_id, lat_lng in buildings.items():
                lat, lng = lat_lng
                queue.append((building_id, lat, lng))
                
                if max_buildings and processed_count >= max_buildings:
                    break
                    
                processed_count += 1
                
            logger.info(f"Starting to process {len(queue)} buildings")
            
            # Process buildings in batches
            batch_size = 5
            for i in range(0, len(queue), batch_size):
                batch = queue[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1} of {len(queue)//batch_size + 1}")
                
                for building_id, lat, lng in batch:
                    try:
                        result = self.run_and_log_building(lat, lng, building_id)
                        if result:
                            results.append(result)
                            logger.info(f"Successfully processed building {building_id}")
                        
                    except Exception as e:
                        logger.error(f"Error processing building {building_id}: {str(e)}")
                        continue
                
                # Add a small delay between batches to avoid rate limiting
                time.sleep(1)
                
            logger.info(f"Finished processing {len(results)} buildings")
            return results
            
        except Exception as e:
            logger.error(f"Error in process_buildings: {str(e)}")
            raise

    def process_zip_code(self, zipcode, max_buildings=None):
        """Process all buildings in a ZIP code"""
        try:
            # Get bounding box for ZIP code
            bbox = self.zip_to_bounding_box(zipcode)
            if not bbox:
                logger.error(f"Could not get bounding box for ZIP code {zipcode}")
                return []
                
            # Fetch buildings within bounding box
            buildings = self.fetch_buildings(bbox)
            if not buildings:
                logger.error(f"No buildings found in ZIP code {zipcode}")
                return []
                
            logger.info(f"Found {len(buildings)} buildings in ZIP code {zipcode}")
            
            # Process buildings
            results = self.process_buildings(buildings, max_buildings)
            return results
            
        except Exception as e:
            logger.error(f"Error processing ZIP code {zipcode}: {str(e)}")
            raise

    def process_county(self, county, state, max_buildings=None):
        """Process all buildings in a county"""
        try:
            # Get bounding box for county
            bbox = self.county_to_bounding_box(county, state)
            if not bbox:
                logger.error(f"Could not get bounding box for {county}, {state}")
                return []
                
            # Fetch buildings within bounding box
            buildings = self.fetch_buildings(bbox)
            if not buildings:
                logger.error(f"No buildings found in {county}, {state}")
                return []
                
            logger.info(f"Found {len(buildings)} buildings in {county}, {state}")
            
            # Process buildings
            results = self.process_buildings(buildings, max_buildings)
            return results
            
        except Exception as e:
            logger.error(f"Error processing {county}, {state}: {str(e)}")
            raise

    def test_process_single_building(self, lat, lng, building_id="test_building"):
        """Process a single building and visualize each step"""
        try:
            # Step 1: Fetch initial satellite image
            initial_img_path = self.folders["initial"] / f"{building_id}_initial.jpg"
            img = self.fetch_satellite_image(lat, lng, initial_zoom)
            if img is None:
                return None
                
            # Convert image to RGB before saving
            img = img.convert('RGB')
            img.save(initial_img_path)
            print(f"Step 1: Initial image saved to {initial_img_path}")
            
            # Step 2: Detect rooftop
            result = self.detect_rooftop(initial_img_path)
            if not result.get("success", False):
                print("No rooftops found in initial detection")
                return None
                
            polygon = result["polygon"]
            polygon_np = np.array(polygon)
            
            # Step 3: Create visualization of initial detection
            initial_vis_path = self.folders["initial"] / f"{building_id}_initial_detection.jpg"
            self.visualize_detection(img, polygon_np, initial_vis_path)
            print(f"Step 2: Initial detection visualization saved to {initial_vis_path}")
            
            # Step 4: Deskew and crop the image
            deskewed_path = self.folders["deskewed"] / f"{building_id}_deskewed.jpg"
            deskewed_img, angle = self.auto_rotate_crop(img, polygon_np, deskewed_path)
            if deskewed_img is None:
                print(f"Failed to deskew image. Angle: {angle}")
                return None
                
            print(f"Step 3: Deskewed image saved to {deskewed_path}")
            
            # Step 5: Get refined rooftop detection
            refined_result = self.detect_rooftop(deskewed_path)
            if not refined_result.get("success", False):
                print("No rooftops found in deskewed image")
                return None
                
            refined_polygon = refined_result["polygon"]
            refined_polygon_np = np.array(refined_polygon, dtype=np.int32)
            
            # Step 6: Create visualization of refined detection
            refined_vis_path = self.folders["deskewed"] / f"{building_id}_refined_detection.jpg"
            self.visualize_detection(deskewed_img, refined_polygon_np, refined_vis_path)
            print(f"Step 4: Refined detection visualization saved to {refined_vis_path}")
            
            # Step 7: Calculate building area
            building_area_m2 = self.calculate_area_in_sq_meters(refined_polygon_np)
            building_area_sqft = building_area_m2 * 10.7639
            print(f"Step 5: Building area: {building_area_sqft:.2f} sq.ft")
            
            # Step 8: Crop the refined rooftop
            x, y, w, h = cv2.boundingRect(refined_polygon_np)
            img_np = np.array(deskewed_img)
            cropped_np = img_np[y:y+h, x:x+w]
            cropped_img = Image.fromarray(cropped_np)
            
            # Step 9: Save cropped image
            final_crop_path = self.folders["refined_crop"] / f"{building_id}_final_crop.jpg"
            cropped_img = cropped_img.convert('RGB')
            cropped_img.save(final_crop_path)
            print(f"Step 6: Final cropped image saved to {final_crop_path}")
            
            # Step 10: Detect RTUs
            results = self.rtu_model(final_crop_path)
            rtu_count = 0
            
            # Step 11: Draw RTU detections
            rtu_img = Image.open(final_crop_path).convert("RGB")
            draw = ImageDraw.Draw(rtu_img)
            
            for result in results:
                for box in result.boxes:
                    cls_name = result.names[int(box.cls)]
                    if cls_name.lower() in ["rtu", "rooftop_unit"]:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
                        draw.text((x1, y1 - 10), cls_name, fill="green")
                        rtu_count += 1
            
            # Step 12: Save RTU detection image
            rtu_output_path = self.folders["rtu_detected"] / f"{building_id}_rtu_detected.jpg"
            rtu_img.save(rtu_output_path)
            print(f"Step 7: RTU detection image saved to {rtu_output_path}")
            print(f"Found {rtu_count} RTUs")
            
            return {
                "building_id": building_id,
                "latitude": lat,
                "longitude": lng,
                "area_sqft": round(building_area_sqft, 2),
                "rtu_count": rtu_count,
                "images": {
                    "initial": str(initial_img_path),
                    "initial_detection": str(initial_vis_path),
                    "deskewed": str(deskewed_path),
                    "refined_detection": str(refined_vis_path),
                    "final_crop": str(final_crop_path),
                    "rtu_detected": str(rtu_output_path)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in test_process_single_building: {str(e)}")
            raise
