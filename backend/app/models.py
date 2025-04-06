from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List

Base = declarative_base()

class ImageUpload(Base):
    __tablename__ = "image_uploads"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    processed_image = Column(String, nullable=True, default='')  # Make processed_image nullable and add default value
    rtu_count = Column(Integer)
    building_name = Column(String)
    address = Column(String)
    city = Column(String)
    state = Column(String)
    zip_code = Column(String)
    status = Column(String, default="pending")
    approved = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "processed_image": self.processed_image or '',  # Return empty string if null
            "rtu_count": self.rtu_count,
            "building_name": self.building_name,
            "address": self.address,
            "city": self.city,
            "state": self.state,
            "zip_code": self.zip_code,
            "status": self.status,
            "approved": self.approved,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"User(id={self.id}, username='{self.username}', email='{self.email}')"

class ImageResponse(BaseModel):
    filename: str
    rtu_count: int
    processed_image: str

class ImageUploadResponse(BaseModel):
    id: int
    filename: str
    processed_image: str
    rtu_count: int
    building_name: str
    address: str
    city: str
    state: str
    zip_code: str
    status: str
    approved: bool
    created_at: datetime
    updated_at: Optional[datetime]

class ApprovalResponse(BaseModel):
    id: int
    approved: bool
    message: str
