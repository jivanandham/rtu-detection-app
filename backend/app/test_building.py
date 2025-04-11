import os
from dotenv import load_dotenv
import cv2
import numpy as np
from PIL import Image, ImageDraw
import requests
from io import BytesIO
from inference_sdk import InferenceHTTPClient
from ultralytics import YOLO
from datetime import datetime
import time
import csv
from concurrent.futures import ThreadPoolExecutor
from geopy.geocoders import Nominatim

# Load environment variables
load_dotenv()

# === CONFIG ===
GOOGLE_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE")
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "jsr-hgtq0/2")
RTU_MODEL_PATH = os.getenv("RTU_MODEL_PATH", "/Users/jeevakrishnasamy/Desktop/rtu-detection-app/backend/weights/Yolov11.pt")
size_px = 640
initial_zoom = 19

print(f"\n=== Configuration ===")
print(f"Google Maps API Key: {'Set' if GOOGLE_API_KEY else 'Not set'}")
print(f"Roboflow API Key: {'Set' if ROBOFLOW_API_KEY else 'Not set'}")
print(f"Roboflow Workspace: {ROBOFLOW_WORKSPACE}")
print(f"Roboflow Model ID: {ROBOFLOW_MODEL_ID}")
print(f"RTU Model Path: {RTU_MODEL_PATH}")

# === FOLDER STRUCTURE ===
# Create main output directory
main_output_dir = os.getenv("OUTPUT_DIR", "rooftop_pipeline_output")
os.makedirs(main_output_dir, exist_ok=True)

# Create timestamped directory inside main output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_folder = os.path.join(main_output_dir, f"pipeline_{timestamp}")
folders = {
    "initial": os.path.join(base_folder, "01_initial"),
    "overlay": os.path.join(base_folder, "02_mask_overlay"),
    "deskewed": os.path.join(base_folder, "03_deskewed"),
    "refined_overlay": os.path.join(base_folder, "04_refined_overlay"),
    "refined_crop": os.path.join(base_folder, "05_final_crop"),
    "rtu_detected": os.path.join(base_folder, "06_rtu_detected")
}
for path in folders.values():
    os.makedirs(path, exist_ok=True)

# === FUNCTIONS ===
def get_bounds_from_county(county_name):
    url = f"https://nominatim.openstreetmap.org/search?county={county_name}&country=USA&format=json&limit=1"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    data = response.json()
    if not data:
        raise Exception(f"Failed to geocode county: {county_name}")
    bbox = data[0]["boundingbox"]
    return {
        "south": float(bbox[0]),
        "north": float(bbox[1]),
        "west": float(bbox[2]),
        "east": float(bbox[3])
    }

def fetch_satellite_image(lat, lng, zoom, save_path):
    url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{lat},{lng}",
        "zoom": str(zoom),
        "size": f"{size_px}x{size_px}",
        "maptype": "satellite",
        "key": GOOGLE_API_KEY
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content)).convert("RGB")
    img.save(save_path)
    return img

def auto_rotate_crop(pil_img, polygon_np, output_path):
    img_cv = np.array(pil_img)[:, :, ::-1]
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

def fetch_buildings(bounds):
    overpass_url = "http://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    (
      way["building"="industrial"]({bounds['south']},{bounds['west']},{bounds['north']},{bounds['east']});
      way["building"="warehouse"]({bounds['south']},{bounds['west']},{bounds['north']},{bounds['east']});
      way["building"="industrial_hall"]({bounds['south']},{bounds['west']},{bounds['north']},{bounds['east']});
      way["building"="factory"]({bounds['south']},{bounds['west']},{bounds['north']},{bounds['east']});
      way["building"="manufacture"]({bounds['south']},{bounds['west']},{bounds['north']},{bounds['east']});
    );
    out center;
    """
    response = requests.post(overpass_url, data={"data": query})
    response.raise_for_status()
    data = response.json()
    buildings = [
        {"id": el["id"], "lat": el["center"]["lat"], "lng": el["center"]["lon"]}
        for el in data["elements"] if el["type"] == "way" and "center" in el
    ]
    return buildings

def calculate_area_in_sq_meters(polygon_np):
    meters_per_pixel = 0.3
    if polygon_np.shape[0] < 3:
        return 0.0
    area_pixels = 0.5 * np.abs(np.dot(polygon_np[:, 0], np.roll(polygon_np[:, 1], 1)) - np.dot(polygon_np[:, 1], np.roll(polygon_np[:, 0], 1)))
    return area_pixels * (meters_per_pixel ** 2)

def reverse_geocode(lat, lng):
    try:
        geolocator = Nominatim(user_agent="rtu_detector")
        location = geolocator.reverse((lat, lng), timeout=10)
        return location.address if location else "Unknown Address"
    except:
        return "Geocoding Failed"

def process_building(lat, lng, building_id):
    try:
        if not ROBOFLOW_API_KEY:
            print(f"❌ Skipping {building_id}: Roboflow API key not configured")
            return None
            
        if not ROBOFLOW_WORKSPACE:
            print(f"❌ Skipping {building_id}: Roboflow workspace not configured")
            return None
            
        if not ROBOFLOW_MODEL_ID:
            print(f"❌ Skipping {building_id}: Roboflow model ID not configured")
            return None
            
        initial_img_path = os.path.join(folders["initial"], f"{building_id}_initial.jpg")
        img = fetch_satellite_image(lat, lng, initial_zoom, initial_img_path)
        if not img:
            print(f"❌ Failed to fetch satellite image for {building_id}")
            return None
            
        address = reverse_geocode(lat, lng)
        print(f"📍 Building location: {address}")
        
        # Parse the model ID correctly
        model_parts = ROBOFLOW_MODEL_ID.split('/')
        if len(model_parts) == 2:
            project_id, version = model_parts
        else:
            project_id = ROBOFLOW_MODEL_ID
            version = "1"  # Default to version 1 if not specified
            
        full_model_id = f"{project_id}/{version}"
        print(f"🔍 Using Roboflow model: {full_model_id}")
        
        client = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=ROBOFLOW_API_KEY)
        result = client.infer(initial_img_path, model_id=full_model_id)
        
        if not result or not result.get("predictions"):
            print(f"❌ No rooftops found for {building_id}")
            return None
            
        polygon = result["predictions"][0]["points"]
        polygon_np = np.array([[p["x"], p["y"]] for p in polygon])
        
        deskewed_path = os.path.join(folders["deskewed"], f"{building_id}_deskewed.jpg")
        deskewed_img, angle = auto_rotate_crop(img, polygon_np, deskewed_path)
        
        if deskewed_img is None:
            print(f"❌ Failed to deskew image for {building_id}")
            return None
            
        refined_result = client.infer(deskewed_path, model_id=full_model_id)
        
        if not refined_result or not refined_result.get("predictions"):
            print(f"❌ No rooftops found in deskewed image for {building_id}")
            return None
            
        refined_polygon = refined_result["predictions"][0]["points"]
        refined_polygon_np = np.array([[p["x"], p["y"]] for p in refined_polygon], dtype=np.int32)
        
        building_area_m2 = calculate_area_in_sq_meters(refined_polygon_np)
        building_area_sqft = building_area_m2 * 10.7639
        print(f"🏢 Total build space: {building_area_sqft:.2f} square feet")
        
        x, y, w, h = cv2.boundingRect(refined_polygon_np)
        img_np = np.array(deskewed_img)
        cropped_np = img_np[y:y+h, x:x+w]
        cropped_img = Image.fromarray(cropped_np)
        
        final_crop_path = os.path.join(folders["refined_crop"], f"{building_id}_final_crop.jpg")
        cropped_img.save(final_crop_path)
        
        rtu_model = YOLO(RTU_MODEL_PATH)
        results = rtu_model(final_crop_path)
        rtu_count = 0
        
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
        
        if rtu_count > 5 and building_area_sqft > 30000:
            rtu_output_path = os.path.join(folders["rtu_detected"], f"{building_id}_rtu_detected.jpg")
            rtu_img.save(rtu_output_path)
            print(f"✅ {rtu_count} RTU(s) detected for {building_id}. Meets criteria with {building_area_sqft:.2f} sq.ft")
            return {
                "building_id": building_id,
                "latitude": lat,
                "longitude": lng,
                "address": address,
                "area_sqft": round(building_area_sqft, 2),
                "rtu_count": rtu_count
            }
        
        print(f"❌ {building_id} did not meet criteria (Area: {building_area_sqft:.2f} sq.ft, RTUs: {rtu_count})")
        return None
        
    except Exception as e:
        print(f"❌ Error processing building {building_id}: {str(e)}")
        return None

def process_county(county_name):
    try:
        print(f"\nProcessing county: {county_name}")
        bounds = get_bounds_from_county(county_name)
        buildings = fetch_buildings(bounds)
        print(f"Found {len(buildings)} buildings in {county_name}")
        
        processed_locations = set()
        tasks = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            for idx, b in enumerate(buildings):
                coords_key = (round(b["lat"], 5), round(b["lng"], 5))
                if coords_key in processed_locations:
                    continue
                processed_locations.add(coords_key)
                building_id = f"{county_name}_{b['id']}"
                tasks.append(executor.submit(process_building, b["lat"], b["lng"], building_id))
                
        print(f"✅ Completed processing {len(tasks)} buildings in {county_name}")
        
    except Exception as e:
        print(f"❌ Error processing county {county_name}: {str(e)}")

def main():
    # Create CSV file for results
    results_csv = os.path.join(base_folder, "filtered_results.csv")
    with open(results_csv, mode='w', newline='') as csvfile:
        fieldnames = ["building_id", "latitude", "longitude", "address", "area_sqft", "rtu_count"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    # Process counties
    counties = ["Allegheny County"]  # Add more counties as needed
    for county in counties:
        try:
            process_county(county)
            time.sleep(2)
        except Exception as e:
            print(f"❌ Error processing county {county}: {str(e)}")
    
    print(f"\n🎯 All results saved under: {base_folder}")
    print(f"📄 CSV with filtered buildings saved to: {results_csv}")

if __name__ == "__main__":
    main()
