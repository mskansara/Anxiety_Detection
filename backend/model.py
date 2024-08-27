from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional
from bson import ObjectId
from datetime import datetime


class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")


class Contact(BaseModel):
    phone: str
    email: EmailStr


class Patient(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    name: str
    dob: datetime
    contact: Contact
    doctors: Optional[List[PyObjectId]] = []

    class Config:
        allow_population_by_field_name = True
        json_encoders = {ObjectId: str}


class Doctor(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    name: str
    specialization: str
    contact: Contact
    patients: Optional[List[PyObjectId]] = []

    class Config:
        allow_population_by_field_name = True
        json_encoders = {ObjectId: str}


class Session(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    patient_id: PyObjectId
    doctor_id: PyObjectId
    session_date: datetime
    summary: str

    class Config:
        allow_population_by_field_name = True
        json_encoders = {ObjectId: str}
