from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from app.models import ImageUpload, ImageUploadResponse, ImageResponse, ApprovalResponse
from app.database import engine, Base, SessionLocal
from app.rtu_detector import RTUDetector
from app.building_detector import BuildingDetector
from sqlalchemy.orm import Session
from pathlib import Path
import shutil
import logging
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import func
from pydantic import BaseModel
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
SQLALCHEMY_DATABASE_URL = "postgresql://postgres:postgres@localhost/rtu_detection"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

app = FastAPI(title="RTU Detection API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Initialize RTU detector
rtu_detector = RTUDetector()

# Initialize Building detector
building_detector = BuildingDetector()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/upload", response_model=ImageResponse)
async def upload_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    try:
        logger.info(f"Received file upload: {file.filename}")
        
        # Validate file type
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise HTTPException(
                status_code=400,
                detail="File must be a PNG or JPEG image"
            )

        # Save uploaded file
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        file_path = upload_dir / file.filename
        
        try:
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(f"Saved uploaded file to: {file_path}")
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save uploaded file: {str(e)}"
            )

        # Detect RTUs
        try:
            detection_result = rtu_detector.detect(str(file_path))
            logger.info(f"Detection completed for {file.filename}")
        except Exception as e:
            logger.error(f"Error during RTU detection: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to detect RTUs: {str(e)}"
            )
        
        # Create response
        response = ImageResponse(
            filename=file.filename,
            rtu_count=detection_result["rtu_count"],
            processed_image=f"/uploads/processed/{file.filename}"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/save_upload", response_model=ImageUploadResponse)
async def save_upload(
    building_name: str = Form(...),
    address: str = Form(...),
    city: str = Form(...),
    state: str = Form(...),
    zip_code: str = Form(...),
    processed_image: str = Form(...),
    rtu_count: int = Form(...),
    db: Session = Depends(get_db)
):
    try:
        # Extract filename from processed_image path
        filename = processed_image.split('/')[-1]
        
        # Create database record
        db_image = ImageUpload(
            filename=filename,
            processed_image=processed_image,
            rtu_count=rtu_count,
            building_name=building_name,
            address=address,
            city=city,
            state=state,
            zip_code=zip_code,
            status="pending",
            approved=False
        )
        db.add(db_image)
        db.commit()
        db.refresh(db_image)

        # Create response with all required fields
        response = ImageUploadResponse(
            id=db_image.id,
            filename=db_image.filename,
            processed_image=db_image.processed_image,
            rtu_count=db_image.rtu_count,
            building_name=db_image.building_name,
            address=db_image.address,
            city=db_image.city,
            state=db_image.state,
            zip_code=db_image.zip_code,
            status=db_image.status,
            approved=db_image.approved,
            created_at=db_image.created_at,
            updated_at=db_image.updated_at
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Save failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/approve/{upload_id}", response_model=ApprovalResponse)
async def approve_upload(upload_id: int, db: Session = Depends(get_db)):
    try:
        # Get the upload record
        upload = db.query(ImageUpload).filter(ImageUpload.id == upload_id).first()
        if not upload:
            raise HTTPException(
                status_code=404,
                detail="Upload not found"
            )

        # Update approval status
        upload.approved = True
        upload.status = "approved"
        db.commit()
        
        return ApprovalResponse(
            id=upload_id,
            approved=True,
            message="Upload approved successfully"
        )
        
    except Exception as e:
        logger.error(f"Approval failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.delete("/delete/{id}")
async def delete_upload(id: int, db: Session = Depends(get_db)):
    try:
        # Get the record
        db_image = db.query(ImageUpload).filter(ImageUpload.id == id).first()
        if not db_image:
            raise HTTPException(
                status_code=404,
                detail="Record not found"
            )

        # Delete the record
        db.delete(db_image)
        db.commit()

        # Delete the processed image file if it exists
        if db_image.processed_image:
            processed_path = Path(db_image.processed_image)
            if processed_path.exists():
                processed_path.unlink()

        return {"message": "Record deleted successfully"}
    except Exception as e:
        logger.error(f"Delete failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to delete record"
        )

@app.get("/history", response_model=list[ImageUploadResponse])
async def get_history(db: Session = Depends(get_db)):
    try:
        history = db.query(ImageUpload).all()
        logger.info(f"Retrieved {len(history)} records from history")
        
        # Convert to list of dictionaries with proper path handling
        history_data = []
        for item in history:
            item_dict = item.to_dict()
            
            # Ensure processed_image path is correct
            if item_dict["processed_image"]:
                # Make sure path starts with /uploads
                if not item_dict["processed_image"].startswith('/uploads/processed/'):
                    filename = item_dict["processed_image"].split('/')[-1]
                    item_dict["processed_image"] = f'/uploads/processed/{filename}'
            
            history_data.append(item_dict)
        
        return history_data
    except Exception as e:
        logger.error(f"Failed to retrieve history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve history"
        )

@app.get("/analytics", response_model=dict)
async def get_analytics(db: Session = Depends(get_db)):
    total_images = db.query(func.count(ImageUpload.id)).scalar()
    total_rtus = db.query(func.sum(ImageUpload.rtu_count)).scalar() or 0
    avg_lead_score = db.query(func.avg(ImageUpload.lead_score)).scalar() or 0
    total_buildings = db.query(func.count(func.distinct(ImageUpload.building_name))).scalar()
    
    # Calculate success rate
    successful_detections = db.query(func.count(ImageUpload.id)).filter(ImageUpload.rtu_count > 0).scalar()
    success_rate = (successful_detections / total_images * 100) if total_images > 0 else 0
    
    # Calculate failed detections
    failed_detections = db.query(func.count(ImageUpload.id)).filter(ImageUpload.rtu_count == 0).scalar()
    
    return {
        "totalImages": total_images,
        "totalRtuCount": total_rtus,
        "successRate": round(success_rate, 2),
        "avgLeadScore": round(avg_lead_score, 2),
        "failedDetections": failed_detections,
        "totalBuildings": total_buildings
    }

# Building Detection models
class BuildingDetectionRequest(BaseModel):
    zipcode: Optional[str] = None
    county: Optional[str] = None
    state: Optional[str] = None
    max_buildings: Optional[int] = 20

class BuildingResult(BaseModel):
    building_id: str
    lat: float
    lng: float
    address: str
    rooftop_area_sqft: float
    rtu_count: int
    meets_criteria: bool
    original_image: str
    processed_image: Optional[str] = None

class BuildingDetectionResponse(BaseModel):
    success: bool
    error: Optional[str] = None
    zipcode: Optional[str] = None
    county: Optional[str] = None
    state: Optional[str] = None
    total_buildings_processed: Optional[int] = None
    buildings_with_rtus: Optional[int] = None
    buildings_meeting_criteria: Optional[int] = None
    results: Optional[List[BuildingResult]] = None
    csv_path: Optional[str] = None

@app.post("/detect-buildings-by-zip", response_model=BuildingDetectionResponse)
async def detect_buildings_by_zip(request: BuildingDetectionRequest, db: Session = Depends(get_db)):
    try:
        if not request.zipcode:
            raise HTTPException(status_code=400, detail="ZIP code is required")
            
        result = building_detector.process_zipcode(
            request.zipcode, 
            rtu_detector=rtu_detector,
            max_buildings=request.max_buildings,
            db=db
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Failed to process ZIP code"))
            
        return result
    except Exception as e:
        logger.error(f"Building detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Building detection failed: {str(e)}")

@app.post("/detect-buildings-by-county", response_model=BuildingDetectionResponse)
async def detect_buildings_by_county(request: BuildingDetectionRequest, db: Session = Depends(get_db)):
    try:
        if not request.county or not request.state:
            raise HTTPException(status_code=400, detail="County and state are required")
            
        result = building_detector.process_county(
            request.county,
            request.state,
            rtu_detector=rtu_detector,
            max_buildings=request.max_buildings,
            db=db
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Failed to process county"))
            
        return result
    except Exception as e:
        logger.error(f"Building detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Building detection failed: {str(e)}")

@app.get("/building-detections", response_model=List[BuildingResult])
async def get_building_detections(db: Session = Depends(get_db)):
    """Get all building detection results from the database"""
    try:
        buildings = db.query(BuildingDetection).all()
        return [BuildingResult(
            building_id=b.building_id,
            lat=b.lat,
            lng=b.lng,
            address=b.address,
            rooftop_area_sqft=b.rooftop_area_sqft,
            rtu_count=b.rtu_count,
            meets_criteria=b.meets_criteria,
            original_image=b.original_image,
            processed_image=b.processed_image
        ) for b in buildings]
    except Exception as e:
        logger.error(f"Failed to retrieve building detections: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve building detections: {str(e)}")

@app.get("/building-detections/{building_id}", response_model=BuildingResult)
async def get_building_detection(building_id: int, db: Session = Depends(get_db)):
    """Get a specific building detection result from the database"""
    try:
        building = db.query(BuildingDetection).filter(BuildingDetection.id == building_id).first()
        if not building:
            raise HTTPException(status_code=404, detail="Building not found")
        
        return BuildingResult(
            building_id=building.building_id,
            lat=building.lat,
            lng=building.lng,
            address=building.address,
            rooftop_area_sqft=building.rooftop_area_sqft,
            rtu_count=building.rtu_count,
            meets_criteria=building.meets_criteria,
            original_image=building.original_image,
            processed_image=building.processed_image
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve building detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve building detection: {str(e)}")

@app.get("/building-detections/search", response_model=List[BuildingResult])
async def search_building_detections(
    search_type: Optional[str] = None,
    search_query: Optional[str] = None,
    meets_criteria: Optional[bool] = None,
    min_rtu_count: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Search building detection results with filters"""
    try:
        query = db.query(BuildingDetection)
        
        if search_type:
            query = query.filter(BuildingDetection.search_type == search_type)
        
        if search_query:
            query = query.filter(BuildingDetection.search_query.ilike(f"%{search_query}%"))
        
        if meets_criteria is not None:
            query = query.filter(BuildingDetection.meets_criteria == meets_criteria)
        
        if min_rtu_count is not None:
            query = query.filter(BuildingDetection.rtu_count >= min_rtu_count)
        
        buildings = query.all()
        
        return [BuildingResult(
            building_id=b.building_id,
            lat=b.lat,
            lng=b.lng,
            address=b.address,
            rooftop_area_sqft=b.rooftop_area_sqft,
            rtu_count=b.rtu_count,
            meets_criteria=b.meets_criteria,
            original_image=b.original_image,
            processed_image=b.processed_image
        ) for b in buildings]
    except Exception as e:
        logger.error(f"Failed to search building detections: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to search building detections: {str(e)}")
