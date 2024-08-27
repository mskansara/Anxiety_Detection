from fastapi import FastAPI, Depends, HTTPException, APIRouter, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# Constants
ATLAS_APP_ID = os.getenv("REALM_APP_ID")
ATLAS_LOGIN_URL = f"https://realm.mongodb.com/api/client/v2.0/app/{ATLAS_APP_ID}/auth/providers/local-userpass/login"
# MongoDB connection
client = AsyncIOMotorClient(os.getenv("DB_URL"))
db = client["patient_management_system"]

router = APIRouter()


class Token(BaseModel):
    access_token: str
    token_type: str
    role: str


class TokenData(BaseModel):
    email: str
    role: str


async def get_user_from_db(email: str):
    email = email.lower()  # Ensure email is lowercase for consistent querying

    # Check in the doctors collection
    user = await db.doctors.find_one({"contact.email": email})
    if user:
        # Assign role as 'doctor' if found in doctors collection
        user["role"] = "doctor"
        return user

    # Check in the patients collection
    user = await db.patients.find_one({"contact.email": email})
    if user:
        # Assign role as 'patient' if found in patients collection
        user["role"] = "patient"
        return user

    return None


@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    print(form_data.username)
    print(form_data.password)
    response = requests.post(
        ATLAS_LOGIN_URL,
        json={
            "username": form_data.username,
            "password": form_data.password,
        },
    )

    if response.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

    access_token = response.json()["access_token"]

    # Fetch user role from your MongoDB collection
    user = await get_user_from_db(form_data.username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    role = user["role"]

    return {"access_token": access_token, "token_type": "bearer", "role": role}


# OAuth2 scheme to protect routes
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


async def get_current_user(token: str = Depends(oauth2_scheme)):
    # Retrieve the user profile from MongoDB Realm (Atlas App Services)
    response = requests.get(
        "https://realm.mongodb.com/api/client/v2.0/auth/profile",
        headers={"Authorization": f"Bearer {token}"},
    )

    if response.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )

    # Extract email from the user profile
    user_profile = response.json()
    email = user_profile["data"]["email"]

    # Fetch user from your MongoDB collection (either doctors or patients)
    user = await get_user_from_db(email)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    # Add the role to the user's details
    role = user.get("role", "unknown")  # Default to 'unknown' if role isn't found

    # Return user details including access token, role, and email
    return {"email": email, "access_token": token, "role": role}


async def get_current_doctor(current_user: TokenData = Depends(get_current_user)):
    print(current_user)
    if current_user["role"] != "doctor":
        raise HTTPException(status_code=403, detail="Not authorized as a doctor")
    return current_user


async def get_current_patient(current_user: TokenData = Depends(get_current_user)):
    if current_user["role"] != "patient":
        raise HTTPException(status_code=403, detail="Not authorized as a patient")
    return current_user


@router.post("/logout")
async def logout(token: str = Depends(oauth2_scheme)):
    # Here, you could add token to a blacklist or perform other logout-related actions
    return {"message": "Logout successful"}
