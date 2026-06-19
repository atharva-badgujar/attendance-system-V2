from typing import List
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from attendance_config import DB_CONFIG
from database_manager import DatabaseManager

app = FastAPI(
    title="Attendance System API",
    description="Backend API for Face Recognition Attendance System",
    version="1.0.0"
)

db = DatabaseManager(DB_CONFIG)


class StudentRegistrationPayload(BaseModel):
    prn: str
    class_id: int
    roll_no: int
    name: str
    email: str
    face_encoding: List[float]


@app.get("/")
async def root():
    return {
        "message": "Attendance API is running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/classes")
async def list_classes():
    classes = db.get_all_classes()
    return [
        {"id": class_id, "name": class_name}
        for class_name, class_id in classes.items()
    ]


@app.get("/subjects")
async def list_subjects():
    subjects = db.get_all_subjects()
    return [
        {"id": subject_id, "name": subject_name}
        for subject_name, subject_id in subjects.items()
    ]


@app.post("/register")
async def register_student(payload: StudentRegistrationPayload):
    try:
        encoding_array = np.array(payload.face_encoding, dtype=np.float32)

        success, message = db.register_student(
            payload.prn,
            payload.class_id,
            payload.roll_no,
            payload.name,
            payload.email,
            encoding_array,
        )

        if not success:
            raise HTTPException(status_code=400, detail=message)

        return {"detail": message}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/students")
async def get_students():
    if not hasattr(db, "get_all_students"):
        return {"message": "Student listing not implemented yet"}
    return db.get_all_students()


@app.get("/attendance")
async def get_attendance():
    if not hasattr(db, "get_attendance_logs"):
        return {"message": "Attendance logs not implemented yet"}
    return db.get_attendance_logs()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)