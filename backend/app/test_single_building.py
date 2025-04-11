import os
from dotenv import load_dotenv
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
from ultralytics import YOLO
from datetime import datetime
import time
import csv
from geopy.geocoders import Nominatim

# Load environment variables
load_dotenv()

# === CONFIG ===
GOOGLE_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
RTU_MODEL_PATH = os.getenv("RTU_MODEL_PATH")
BUILDING_MODEL_PATH = os.getenv("BUILDING_MODEL_PATH")
size_px = 640
initial_zoom = 15  # Start with a moderate zoom level
zoom_levels = [15, 16, 17, 18, 19]  # Progressive zoom levels up to 19
final_zoom = 19    # Use zoom level 19 for final analysis

print(f"\n=== Configuration ===")
print(f"Google Maps API Key: {'Set' if GOOGLE_API_KEY else 'Not set'}")
print(f"RTU Model Path: {RTU_MODEL_PATH}")
print(f"Building Model Path: {BUILDING_MODEL_PATH}")
print(f"Initial Zoom: {initial_zoom}")
print(f"Zoom Levels: {zoom_levels}")
print(f"Final Zoom: {final_zoom}")

# === FOLDER STRUCTURE ===
# Create main output directory
main_output_dir = os.getenv("OUTPUT_DIR", "rooftop_pipeline_output")
os.makedirs(main_output_dir, exist_ok=True)

# Create timestamped directory inside main output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_folder = os.path.join(main_output_dir, f"single_building_{timestamp}")
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

def fetch_satellite_image(lat, lng, zoom, output_path, scale=2, map_type="satellite"):
    """Fetch satellite image from Google Maps Static API with enhanced parameters for complete rooftop capture"""
    try:
        # Calculate the appropriate size based on zoom level
        # At higher zoom levels, we need a larger image to capture the complete rooftop
        if zoom >= 18:
            # For zoom level 18 and higher, use a larger image size
            img_size = f"{size_px}x{size_px}"
            # Use scale=2 for higher resolution
            scale_param = 2
        else:
            img_size = f"{size_px}x{size_px}"
            scale_param = scale
        
        params = {
            "center": f"{lat},{lng}",
            "zoom": zoom,
            "size": img_size,
            "maptype": map_type,
            "key": GOOGLE_API_KEY,
            "scale": scale_param
        }
        
        response = requests.get("https://maps.googleapis.com/maps/api/staticmap", params=params)
        
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(response.content)
            return Image.open(output_path)
        else:
            print(f"❌ Failed to fetch satellite image: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error fetching satellite image: {str(e)}")
        return None

def analyze_rooftop_at_zoom18(image_path):
    """Analyze the rooftop at zoom level 18 to determine if adjustments are needed"""
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        return None, False
    
    # Detect building outline
    polygon_np = detect_building_outline(image_path)
    
    if polygon_np is None or len(polygon_np) < 3:
        return None, False
    
    # Check if the building polygon is close to the image edge
    img_height, img_width = img.shape[:2]
    margin = img_width * 0.05  # 5% margin
    
    # Get bounding box of the polygon
    x, y, w, h = cv2.boundingRect(polygon_np)
    
    # Check if the building is too close to the edge
    too_close_to_edge = (
        x < margin or 
        y < margin or 
        x + w > img_width - margin or 
        y + h > img_height - margin
    )
    
    # Check if the building is too large for the frame
    building_ratio = (w * h) / (img_width * img_height)
    too_large = building_ratio > 0.9  # Building takes up more than 90% of the image
    
    # Check if the building is too small
    too_small = building_ratio < 0.3  # Building takes up less than 30% of the image
    
    needs_adjustment = too_close_to_edge or too_large or too_small
    
    # Calculate the adjustment needed
    adjustment = {
        "needs_adjustment": needs_adjustment,
        "too_close_to_edge": too_close_to_edge,
        "too_large": too_large,
        "too_small": too_small,
        "polygon": polygon_np,
        "bbox": (x, y, w, h),
        "center": (x + w/2, y + h/2),
        "img_center": (img_width/2, img_height/2)
    }
    
    return polygon_np, adjustment

def get_optimal_zoom_for_building(lat, lng, initial_zoom=15, max_zoom=18):
    """Determine the optimal zoom level to capture the complete building"""
    current_zoom = initial_zoom
    current_lat, current_lng = lat, lng
    
    while current_zoom <= max_zoom:
        # Fetch image at current zoom
        temp_img_path = f"temp_zoom_{current_zoom}.jpg"
        fetch_satellite_image(current_lat, current_lng, current_zoom, temp_img_path)
        
        # Analyze the image
        polygon, adjustment = analyze_rooftop_at_zoom18(temp_img_path)
        
        if polygon is None:
            # If we can't detect a building, try a lower zoom
            if current_zoom > initial_zoom:
                current_zoom -= 1
                continue
            else:
                # If we're already at the lowest zoom, just return it
                return current_zoom, current_lat, current_lng
        
        # If the building is too small, increase zoom
        if adjustment["too_small"] and current_zoom < max_zoom:
            current_zoom += 1
            # Update coordinates to center on the building
            img_center_x, img_center_y = adjustment["img_center"]
            building_center_x, building_center_y = adjustment["center"]
            
            # Calculate offset in pixels
            offset_x = building_center_x - img_center_x
            offset_y = building_center_y - img_center_y
            
            # Convert pixel offset to coordinate offset
            lat_offset = offset_y * 0.00005 * (19 - current_zoom)
            lng_offset = offset_x * 0.00005 * (19 - current_zoom) / np.cos(np.radians(current_lat))
            
            # Update coordinates
            current_lat = current_lat - lat_offset
            current_lng = current_lng + lng_offset
        
        # If the building is too large or too close to the edge, decrease zoom
        elif (adjustment["too_large"] or adjustment["too_close_to_edge"]) and current_zoom > initial_zoom:
            current_zoom -= 1
        else:
            # We found a good zoom level
            break
    
    # Clean up temp file
    try:
        os.remove(temp_img_path)
    except:
        pass
    
    return current_zoom, current_lat, current_lng

def detect_building_outline(image_path):
    """Detect building outline using contour detection with enhanced parameters"""
    # Read the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding to better handle different lighting conditions
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Morphological operations to clean up the image
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area - only keep larger ones
    min_area = img.shape[0] * img.shape[1] * 0.05  # At least 5% of image
    large_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    # If no large contours found, try using Canny edge detection instead
    if not large_contours:
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    # If still no large contours, return the largest one we have
    if not large_contours and contours:
        largest_contour = max(contours, key=cv2.contourArea)
    elif large_contours:
        largest_contour = max(large_contours, key=cv2.contourArea)
    else:
        return None
    
    # Simplify the contour
    epsilon = 0.01 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # If the approximated polygon has too few points, use the original contour
    if len(approx) < 4:
        epsilon = 0.005 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Convert to numpy array of points
    polygon_np = np.array([point[0] for point in approx])
    
    return polygon_np

def get_building_center_coordinates(lat, lng, zoom, img_path=None):
    """Get the center coordinates of the building from image"""
    # If no image path provided, fetch a new image
    if img_path is None:
        img_path = f"temp_img_zoom_{zoom}.jpg"
        fetch_satellite_image(lat, lng, zoom, img_path)
    
    # Detect building outline
    polygon_np = detect_building_outline(img_path)
    
    if polygon_np is None or len(polygon_np) < 3:
        return lat, lng  # Return original coordinates if detection fails
    
    # Calculate center of the polygon in pixels
    center_x, center_y = calculate_center_of_polygon(polygon_np)
    
    # Calculate offset from image center in pixels
    img_center_x, img_center_y = size_px / 2, size_px / 2
    offset_x = center_x - img_center_x
    offset_y = center_y - img_center_y
    
    # Only adjust if the offset is significant
    if abs(offset_x) < size_px * 0.05 and abs(offset_y) < size_px * 0.05:
        return lat, lng
    
    # Convert pixel offset to coordinate offset (approximate)
    # The conversion factor depends on the zoom level and latitude
    # This is a simplified approximation that works better at higher zoom levels
    lat_offset = offset_y * 0.00005 * (21 - zoom)
    lng_offset = offset_x * 0.00005 * (21 - zoom) / np.cos(np.radians(lat))
    
    # Calculate new center coordinates
    new_lat = lat - lat_offset  # Subtract because y increases downward in image
    new_lng = lng + lng_offset
    
    return new_lat, new_lng

def calculate_center_of_polygon(polygon_np):
    """Calculate the center point of a polygon"""
    return np.mean(polygon_np, axis=0)

def auto_rotate_crop(pil_img, polygon_np, output_path):
    """Rotate and crop the image based on the polygon"""
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

def calculate_area_in_sq_meters(polygon_np):
    """Calculate area of polygon in square meters"""
    meters_per_pixel = 0.3
    if polygon_np.shape[0] < 3:
        return 0.0
    area_pixels = 0.5 * np.abs(np.dot(polygon_np[:, 0], np.roll(polygon_np[:, 1], 1)) - np.dot(polygon_np[:, 1], np.roll(polygon_np[:, 0], 1)))
    return area_pixels * (meters_per_pixel ** 2)

def reverse_geocode(lat, lng):
    """Get address from coordinates"""
    try:
        geolocator = Nominatim(user_agent="rtu_detector")
        location = geolocator.reverse((lat, lng), timeout=10)
        return location.address if location else "Unknown Address"
    except:
        return "Geocoding Failed"

def visualize_detection(img, polygon_np, output_path):
    """Create a visualization of the detection"""
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

def process_single_building():
    """Process a single building with visualization at each step"""
    try:
        # Costco in Pittsburgh with corrected coordinates
        lat = 40.44052861089158
        lng = -79.95752703458197
        building_id = "costco_pittsburgh"
        
        print(f"\n=== Processing Building: {building_id} ===")
        
        # Step 1: Fetch satellite images at progressive zoom levels
        zoom_images = {}
        current_lat, current_lng = lat, lng
        
        for i, zoom in enumerate(zoom_levels):
            zoom_img_path = os.path.join(folders["initial"], f"{building_id}_zoom_{zoom}.jpg")
            zoom_img = fetch_satellite_image(current_lat, current_lng, zoom, zoom_img_path)
            zoom_images[zoom] = zoom_img_path
            
            print(f"✅ Step 1.{i+1}: Fetched satellite image at zoom level {zoom}")
            
            # For all but the last zoom level, detect building and recenter
            if zoom < final_zoom:
                # Get centered coordinates for next zoom level
                next_lat, next_lng = get_building_center_coordinates(current_lat, current_lng, zoom, zoom_img_path)
                
                # Calculate how much the center would be adjusted
                distance = ((next_lat - current_lat)**2 + (next_lng - current_lng)**2)**0.5
                
                # Only update if the change is significant but not too large
                if 0.0001 < distance < 0.01:
                    current_lat, current_lng = next_lat, next_lng
                    print(f"   Recentering for next zoom level: {current_lat:.6f}, {current_lng:.6f}")
                else:
                    print(f"   No significant adjustment needed (or adjustment too large)")
        
        # Get address
        address = reverse_geocode(current_lat, current_lng)
        print(f"📍 Building location: {address}")
        
        # Calculate total adjustment from original coordinates
        distance_moved = ((current_lat - lat)**2 + (current_lng - lng)**2)**0.5
        print(f"✅ Step 2: Total coordinate adjustment: {distance_moved:.6f} degrees")
        print(f"   Original: {lat:.6f}, {lng:.6f}")
        print(f"   Final: {current_lat:.6f}, {current_lng:.6f}")
        
        # Step 3: Use the final zoom level (19) for final processing
        # First, analyze the final zoom level image to ensure the building is centered
        final_zoom_img_path = zoom_images[final_zoom]
        polygon_np, adjustment = analyze_rooftop_at_zoom18(final_zoom_img_path)
        
        if polygon_np is not None and adjustment["needs_adjustment"]:
            # Calculate the center of the building in the image
            img_center_x, img_center_y = adjustment["img_center"]
            building_center_x, building_center_y = adjustment["center"]
            
            # Calculate offset in pixels
            offset_x = building_center_x - img_center_x
            offset_y = building_center_y - img_center_y
            
            # Only adjust if the offset is significant
            if abs(offset_x) > size_px * 0.1 or abs(offset_y) > size_px * 0.1:
                # Convert pixel offset to coordinate offset
                lat_offset = offset_y * 0.00002 * (20 - final_zoom)
                lng_offset = offset_x * 0.00002 * (20 - final_zoom) / np.cos(np.radians(current_lat))
                
                # Update coordinates for final image
                final_lat = current_lat - lat_offset
                final_lng = current_lng + lng_offset
                
                print(f"✅ Step 3: Adjusting coordinates for final zoom level {final_zoom}")
                print(f"   From: {current_lat:.6f}, {current_lng:.6f}")
                print(f"   To: {final_lat:.6f}, {final_lng:.6f}")
                
                # Fetch a new image with the adjusted coordinates
                final_img_path = os.path.join(folders["initial"], f"{building_id}_final_zoom_{final_zoom}_centered.jpg")
                fetch_satellite_image(final_lat, final_lng, final_zoom, final_img_path)
                current_lat, current_lng = final_lat, final_lng
            else:
                final_img_path = final_zoom_img_path
                print(f"✅ Step 3: Building is already well-centered at zoom level {final_zoom}")
        else:
            final_img_path = final_zoom_img_path
            print(f"✅ Step 3: Using existing image at zoom level {final_zoom}")
        
        img = Image.open(final_img_path).convert("RGB")
        
        # Step 4: Detect building outline in high-resolution image
        polygon_np = detect_building_outline(final_img_path)
        
        if polygon_np is None or len(polygon_np) < 3:
            print(f"❌ Failed to detect building outline for {building_id}")
            return None
            
        # Step 5: Visualize initial detection
        initial_vis_path = os.path.join(folders["overlay"], f"{building_id}_initial_detection.jpg")
        visualize_detection(img, polygon_np, initial_vis_path)
        print(f"✅ Step 4: Detected building outline and saved visualization to {initial_vis_path}")
        
        # Step 6: Deskew and crop the image
        try:
            deskewed_path = os.path.join(folders["deskewed"], f"{building_id}_deskewed.jpg")
            deskewed_img, angle = auto_rotate_crop(img, polygon_np, deskewed_path)
            
            if deskewed_img is None:
                print(f"❌ Failed to deskew image for {building_id}, using original image instead")
                deskewed_img = img
                deskewed_path = final_img_path
                angle = 0
            else:
                print(f"✅ Step 5: Deskewed image and saved to {deskewed_path} (rotation angle: {angle:.2f}°)")
        except Exception as e:
            print(f"❌ Error during deskewing: {str(e)}, using original image instead")
            deskewed_img = img
            deskewed_path = final_img_path
            angle = 0
        
        # Step 7: Get refined building outline
        try:
            refined_polygon_np = detect_building_outline(deskewed_path)
            
            if refined_polygon_np is None or len(refined_polygon_np) < 3:
                print(f"❌ Failed to detect refined building outline, using original polygon")
                refined_polygon_np = polygon_np
            
            # Step 8: Visualize refined detection
            refined_vis_path = os.path.join(folders["refined_overlay"], f"{building_id}_refined_detection.jpg")
            visualize_detection(deskewed_img, refined_polygon_np, refined_vis_path)
            print(f"✅ Step 6: Refined detection and saved visualization to {refined_vis_path}")
        except Exception as e:
            print(f"❌ Error during refined detection: {str(e)}, using original polygon")
            refined_polygon_np = polygon_np
            refined_vis_path = initial_vis_path
        
        # Step 9: Calculate building area
        building_area_m2 = calculate_area_in_sq_meters(refined_polygon_np)
        building_area_sqft = building_area_m2 * 10.7639
        print(f"✅ Step 7: Building area: {building_area_sqft:.2f} sq.ft")
        
        # Step 10: Crop the refined rooftop
        try:
            x, y, w, h = cv2.boundingRect(refined_polygon_np)
            img_np = np.array(deskewed_img)
            cropped_np = img_np[y:y+h, x:x+w]
            cropped_img = Image.fromarray(cropped_np)
            
            final_crop_path = os.path.join(folders["refined_crop"], f"{building_id}_final_crop.jpg")
            cropped_img.save(final_crop_path)
            print(f"✅ Step 8: Final cropped image saved to {final_crop_path}")
        except Exception as e:
            print(f"❌ Error during cropping: {str(e)}, using deskewed image")
            final_crop_path = deskewed_path
            cropped_img = deskewed_img
        
        # Step 11: Detect RTUs
        try:
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
            
            rtu_output_path = os.path.join(folders["rtu_detected"], f"{building_id}_rtu_detected.jpg")
            rtu_img.save(rtu_output_path)
            print(f"✅ Step 9: Detected {rtu_count} RTUs and saved visualization to {rtu_output_path}")
        except Exception as e:
            print(f"❌ Error during RTU detection: {str(e)}")
            rtu_count = 0
            rtu_output_path = final_crop_path
        
        # Step 12: Create combined visualization of all zoom levels
        try:
            zoom_combined_path = os.path.join(base_folder, f"{building_id}_zoom_levels.jpg")
            zoom_images_dict = {f"Zoom Level {zoom}": path for zoom, path in zoom_images.items()}
            create_combined_visualization(zoom_combined_path, zoom_images_dict)
            print(f"✅ Step 10: Created zoom levels visualization at {zoom_combined_path}")
        except Exception as e:
            print(f"❌ Error creating zoom levels visualization: {str(e)}")
            zoom_combined_path = "Not created"
        
        # Step 13: Create combined visualization of processing steps
        try:
            combined_vis_path = os.path.join(base_folder, f"{building_id}_combined.jpg")
            images_dict = {
                "1. Final High Zoom": final_img_path,
                "2. Building Detection": initial_vis_path,
                "3. Deskewed Image": deskewed_path,
                "4. Refined Detection": refined_vis_path,
                "5. Cropped Building": final_crop_path,
                "6. RTU Detection": rtu_output_path
            }
            create_combined_visualization(combined_vis_path, images_dict)
            print(f"✅ Step 11: Created processing steps visualization at {combined_vis_path}")
        except Exception as e:
            print(f"❌ Error creating combined visualization: {str(e)}")
            combined_vis_path = "Not created"
        
        # Step 14: Save results to CSV
        csv_path = os.path.join(base_folder, "results.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["building_id", "latitude", "longitude", "final_latitude", "final_longitude", "address", "area_sqft", "rtu_count"])
            writer.writerow([building_id, lat, lng, current_lat, current_lng, address, round(building_area_sqft, 2), rtu_count])
        
        print(f"✅ Step 12: Saved results to {csv_path}")
        
        print(f"\n=== Summary ===")
        print(f"Building ID: {building_id}")
        print(f"Location: {address}")
        print(f"Original Coordinates: {lat:.6f}, {lng:.6f}")
        print(f"Final Coordinates: {current_lat:.6f}, {current_lng:.6f}")
        print(f"Area: {building_area_sqft:.2f} sq.ft")
        print(f"RTU Count: {rtu_count}")
        print(f"All results saved to: {base_folder}")
        print(f"Zoom levels visualization: {zoom_combined_path}")
        print(f"Processing steps visualization: {combined_vis_path}")
        
        return {
            "building_id": building_id,
            "latitude": lat,
            "longitude": lng,
            "final_latitude": current_lat,
            "final_longitude": current_lng,
            "address": address,
            "area_sqft": round(building_area_sqft, 2),
            "rtu_count": rtu_count
        }
        
    except Exception as e:
        print(f"❌ Error processing building: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_combined_visualization(output_path, images_dict):
    """Create a combined visualization of all processing steps"""
    # Determine the size of the combined image
    n_images = len(images_dict)
    grid_size = int(np.ceil(np.sqrt(n_images)))
    
    # Load all images and resize them to the same size
    loaded_images = []
    max_width = 0
    max_height = 0
    
    for title, path in images_dict.items():
        if os.path.exists(path):
            img = Image.open(path).convert('RGB')
            loaded_images.append((title, img))
            max_width = max(max_width, img.width)
            max_height = max(max_height, img.height)
    
    # Create a blank canvas
    thumb_size = (400, 400)
    margin = 10
    font_size = 12
    
    canvas_width = (thumb_size[0] + margin) * grid_size + margin
    canvas_height = (thumb_size[1] + margin + font_size + 5) * grid_size + margin
    
    canvas = Image.new('RGB', (canvas_width, canvas_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("Arial", font_size)
    except:
        font = ImageFont.load_default()
    
    # Place images on the canvas
    for i, (title, img) in enumerate(loaded_images):
        # Calculate position
        row = i // grid_size
        col = i % grid_size
        
        x = margin + col * (thumb_size[0] + margin)
        y = margin + row * (thumb_size[1] + margin + font_size + 5)
        
        # Resize image
        img_resized = img.resize(thumb_size, Image.Resampling.LANCZOS)
        
        # Paste image
        canvas.paste(img_resized, (x, y))
        
        # Add title
        text_width = draw.textlength(title, font=font)
        text_x = x + (thumb_size[0] - text_width) // 2
        text_y = y + thumb_size[1] + 5
        draw.text((text_x, text_y), title, fill=(0, 0, 0), font=font)
    
    # Save the combined image
    canvas.save(output_path)
    return canvas

if __name__ == "__main__":
    process_single_building()
