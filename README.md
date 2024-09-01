# Anxiety Detection System

This project is designed to manage patient sessions in a psychologist's clinic, including streaming data processing and session summarization using machine learning models.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Setup Instructions](#setup-instructions)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
- [Running the Application](#running-the-application)
  - [Running the Backend Server](#running-the-backend-server)
  - [Running the Frontend Server](#running-the-frontend-server)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)

## Features

- **Patient Management**: Manage patient and doctor records.
- **Session Management**: Start and stop patient sessions, and automatically summarize sessions.
- **Streaming Data**: Process and analyze streaming data (audio and video) from devices.
- **AI Integration**: Summarize session data using AI models.

## Requirements

- Python 3.8+
- Node.js 14+
- MongoDB
- [Install MongoDB Community Edition](https://docs.mongodb.com/manual/installation/)
- [Install Node.js](https://nodejs.org/)

## Setup Instructions

### Backend Setup

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-repo/anxiety-detection-system.git
    cd anxiety-detection-system/backend
    ```

2. **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Setup Environment Variables**:
    - Create a `.env` file in the `backend` directory and add the following environment variables:
      ```bash
      REALM_APP_ID=your_realm_app_id
      DB_URL=your_mongodb_connection_string
      AUTH_TOKEN=your_auth_token
      SSH_PASSPHRASE=your_ssh_passphrase
      ```

5. **Run the backend server**:
    ```bash
    uvicorn api_test:app --host 0.0.0.0 --port 8000
    ```

### Frontend Setup

1. **Navigate to the frontend directory**:
    ```bash
    cd ../frontend
    ```

2. **Install the required packages**:
    ```bash
    npm install
    ```

3. **Run the frontend server**:
    ```bash
    npm start
    ```

## Running the Application

### Running the Backend Server

- Activate the virtual environment:
    ```bash
    source venv/bin/activate
    ```
- Start the backend server:
    ```bash
    uvicorn api_test:app --host 0.0.0.0 --port 8000
    ```

### Running the Frontend Server

- Ensure you are in the `frontend` directory, then start the frontend server:
    ```bash
    npm start
    ```

## API Endpoints

### Authentication
- `POST /token`: Authenticate users and retrieve a JWT token.

### Doctors
- `POST /doctors/`: Create a new doctor.
- `GET /doctors/{doctor_id}`: Retrieve doctor details by ID.

### Patients
- `POST /patients/`: Create a new patient.
- `GET /patients/{patient_id}`: Retrieve patient details by ID.

### Sessions
- `POST /sessions/`: Create a new session.
- `GET /sessions/{session_id}`: Retrieve session details by ID.
- `GET /generateSummary`: Generate and update the session summary.

## Project Structure
anxiety-detection-system/
│
├── backend/
│   ├── api_test.py             # Main FastAPI backend code
│   ├── model.py                # Pydantic models for database entities
│   ├── requirements.txt        # Python dependencies
│   ├── .env                    # Environment variables (not included in the repo)
│   └── …                     # Other backend files
│
└── frontend/
├── src/
│   ├── components/
│   │   └── NewSession.js   # NewSession React component
│   └── …                 # Other frontend files
├── package.json            # Node.js dependencies
└── …                     # Other frontend files
