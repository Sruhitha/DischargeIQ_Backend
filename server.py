from fastapi import FastAPI, APIRouter, Depends, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session, joinedload
import os
import logging
import re
import json
from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, date, timezone

from models import (
    Patient, Diagnosis, Medication, RiskAlert, FollowUp, Note,
    DischargeSummary, get_db, init_db
)
# from emergentintegrations.llm.chat import LlmChat, UserMessage
import google.generativeai as genai

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

app = FastAPI(title="Discharge IQ API")
api_router = APIRouter(prefix="/api")
v1_router = APIRouter(prefix="")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ── Pydantic Schemas ──────────────────────────────────────────────

class DiagnosisCreate(BaseModel):
    code: Optional[str] = None
    description: str
    diagnosis_type: str = "secondary"
    diagnosed_date: Optional[date] = None

class DiagnosisOut(BaseModel):
    id: int
    patient_id: int
    code: Optional[str] = None
    description: str
    diagnosis_type: str
    diagnosed_date: Optional[date] = None

class MedicationCreate(BaseModel):
    name: str
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    route: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    instructions: Optional[str] = None

class MedicationOut(BaseModel):
    id: int
    patient_id: int
    name: str
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    route: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    instructions: Optional[str] = None

class RiskAlertCreate(BaseModel):
    alert_type: str
    severity: str = "medium"
    description: str

class RiskAlertOut(BaseModel):
    id: int
    patient_id: int
    alert_type: str
    severity: str
    description: str
    created_at: Optional[datetime] = None

class FollowUpCreate(BaseModel):
    department: Optional[str] = None
    physician: Optional[str] = None
    scheduled_date: Optional[date] = None
    instructions: Optional[str] = None

class FollowUpOut(BaseModel):
    id: int
    patient_id: int
    department: Optional[str] = None
    physician: Optional[str] = None
    scheduled_date: Optional[date] = None
    instructions: Optional[str] = None

class NoteCreate(BaseModel):
    author: Optional[str] = None
    content: str
    note_type: str = "progress"

class NoteOut(BaseModel):
    id: int
    patient_id: int
    author: Optional[str] = None
    content: str
    note_type: str
    created_at: Optional[datetime] = None

class SummaryOut(BaseModel):
    id: int
    patient_id: int
    stay_description: Optional[str] = None
    patient_friendly_summary: Optional[str] = None
    generated_at: Optional[datetime] = None

class PatientCreate(BaseModel):
    mir_number: str
    first_name: str
    last_name: str
    date_of_birth: date
    gender: str
    admission_date: datetime
    discharge_date: Optional[datetime] = None
    ward: Optional[str] = None
    attending_physician: Optional[str] = None
    status: str = "admitted"
    diagnoses: List[DiagnosisCreate] = []
    medications: List[MedicationCreate] = []
    risk_alerts: List[RiskAlertCreate] = []
    follow_ups: List[FollowUpCreate] = []
    notes: List[NoteCreate] = []

class PatientOut(BaseModel):
    id: int
    mir_number: str
    first_name: str
    last_name: str
    date_of_birth: date
    gender: str
    admission_date: datetime
    discharge_date: Optional[datetime] = None
    ward: Optional[str] = None
    attending_physician: Optional[str] = None
    status: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    diagnoses: List[DiagnosisOut] = []
    medications: List[MedicationOut] = []
    risk_alerts: List[RiskAlertOut] = []
    follow_ups: List[FollowUpOut] = []
    notes: List[NoteOut] = []
    summary: Optional[SummaryOut] = None

class PatientListItem(BaseModel):
    id: int
    mir_number: str
    first_name: str
    last_name: str
    date_of_birth: date
    gender: str
    admission_date: datetime
    discharge_date: Optional[datetime] = None
    ward: Optional[str] = None
    attending_physician: Optional[str] = None
    status: str
    has_summary: bool = False


# ── Serializers ───────────────────────────────────────────────────

def _serialize_diagnosis(d):
    return {"id": d.id, "patient_id": d.patient_id, "code": d.code, "description": d.description, "diagnosis_type": d.diagnosis_type, "diagnosed_date": d.diagnosed_date}

def _serialize_medication(m):
    return {"id": m.id, "patient_id": m.patient_id, "name": m.name, "dosage": m.dosage, "frequency": m.frequency, "route": m.route, "start_date": m.start_date, "end_date": m.end_date, "instructions": m.instructions}

def _serialize_risk_alert(r):
    return {"id": r.id, "patient_id": r.patient_id, "alert_type": r.alert_type, "severity": r.severity, "description": r.description, "created_at": r.created_at}

def _serialize_follow_up(f):
    return {"id": f.id, "patient_id": f.patient_id, "department": f.department, "physician": f.physician, "scheduled_date": f.scheduled_date, "instructions": f.instructions}

def _serialize_note(n):
    return {"id": n.id, "patient_id": n.patient_id, "author": n.author, "content": n.content, "note_type": n.note_type, "created_at": n.created_at}

def _serialize_summary(s):
    if not s:
        return None
    return {"id": s.id, "patient_id": s.patient_id, "stay_description": s.stay_description, "patient_friendly_summary": s.patient_friendly_summary, "generated_at": s.generated_at}

def serialize_patient(p):
    return {
        "id": p.id, "mir_number": p.mir_number, "first_name": p.first_name, "last_name": p.last_name,
        "date_of_birth": p.date_of_birth, "gender": p.gender, "admission_date": p.admission_date,
        "discharge_date": p.discharge_date, "ward": p.ward, "attending_physician": p.attending_physician,
        "status": p.status, "created_at": p.created_at, "updated_at": p.updated_at,
        "diagnoses": [_serialize_diagnosis(d) for d in p.diagnoses],
        "medications": [_serialize_medication(m) for m in p.medications],
        "risk_alerts": [_serialize_risk_alert(r) for r in p.risk_alerts],
        "follow_ups": [_serialize_follow_up(f) for f in p.follow_ups],
        "notes": [_serialize_note(n) for n in p.notes],
        "summary": _serialize_summary(p.summary),
    }

def serialize_list_item(p):
    return {
        "id": p.id, "mir_number": p.mir_number, "first_name": p.first_name, "last_name": p.last_name,
        "date_of_birth": p.date_of_birth, "gender": p.gender, "admission_date": p.admission_date,
        "discharge_date": p.discharge_date, "ward": p.ward, "attending_physician": p.attending_physician,
        "status": p.status, "has_summary": p.summary is not None,
    }


# ── Helper: parse LLM JSON ───────────────────────────────────────

def parse_llm_response(text):
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    return {"stayDescription": text, "patientFriendlySummary": text}


def build_prompt(patient):
    diag = "\n".join([f"- {d.description} (Code: {d.code or 'N/A'}, Type: {d.diagnosis_type})" for d in patient.diagnoses]) or "None recorded"
    meds = "\n".join([f"- {m.name} {m.dosage or ''} {m.frequency or ''} via {m.route or 'N/A'}" for m in patient.medications]) or "None recorded"
    risks = "\n".join([f"- {r.alert_type}: {r.description} (Severity: {r.severity})" for r in patient.risk_alerts]) or "None recorded"
    fups = "\n".join([f"- {f.department or 'N/A'} with {f.physician or 'N/A'} on {f.scheduled_date or 'TBD'}" for f in patient.follow_ups]) or "None scheduled"
    notes = "\n".join([f"- [{n.note_type}] {n.content}" for n in patient.notes]) or "None"

    return f"""Generate a discharge summary for the following patient. Return ONLY a JSON object with two fields: "stayDescription" and "patientFriendlySummary".

Patient Information:
- Name: {patient.first_name} {patient.last_name}
- MIR: {patient.mir_number}
- DOB: {patient.date_of_birth}
- Gender: {patient.gender}
- Admission: {patient.admission_date}
- Discharge: {patient.discharge_date or 'Not yet discharged'}
- Ward: {patient.ward or 'N/A'}
- Physician: {patient.attending_physician or 'N/A'}
- Status: {patient.status}

Diagnoses:
{diag}

Medications:
{meds}

Risk Alerts:
{risks}

Follow-ups:
{fups}

Notes:
{notes}

Instructions:
1. "stayDescription": Professional clinical summary of the hospital stay (admission reason, treatments, outcomes). 2-3 paragraphs.
2. "patientFriendlySummary": Plain-language explanation for the patient and family, avoiding medical jargon. 2-3 paragraphs.
3. "recommendations": Specific medical recommendations for follow-up care, monitoring, or additional treatments, tailored to the patient's diagnoses and medications. 1-2 paragraphs.
4. "lifestyleTips": Practical lifestyle advice for recovery and health maintenance, customized based on the patient's specific conditions, diagnoses, and risk alerts. 1-2 paragraphs.
5. "foodHealthTips": Dietary recommendations and nutrition advice specifically tailored to the patient's diagnoses, medications, and any dietary restrictions or needs indicated by their conditions. 1-2 paragraphs.

Ensure all tips and recommendations are personalized to this patient's unique clinical profile. Do not provide generic advice - base them on the specific diagnoses, medications, and risk alerts listed above.

Return ONLY the JSON object."""


# ── API Routes ────────────────────────────────────────────────────

@v1_router.get("/")
def api_root():
    return {"message": "Discharge IQ API v1", "status": "healthy"}


@v1_router.get("/patients", response_model=List[PatientListItem])
def list_patients(db: Session = Depends(get_db)):
    patients = db.query(Patient).options(joinedload(Patient.summary)).order_by(Patient.admission_date.desc()).all()
    return [serialize_list_item(p) for p in patients]


@v1_router.get("/patients/search", response_model=List[PatientListItem])
def search_patients(q: str = "", db: Session = Depends(get_db)):
    query = db.query(Patient).options(joinedload(Patient.summary))
    if q:
        search = f"%{q}%"
        query = query.filter(
            (Patient.first_name.ilike(search)) |
            (Patient.last_name.ilike(search)) |
            (Patient.mir_number.ilike(search))
        )
    patients = query.order_by(Patient.admission_date.desc()).all()
    return [serialize_list_item(p) for p in patients]


@v1_router.get("/patients/{patient_id}", response_model=PatientOut)
def get_patient(patient_id: int, db: Session = Depends(get_db)):
    patient = db.query(Patient).options(
        joinedload(Patient.diagnoses), joinedload(Patient.medications),
        joinedload(Patient.risk_alerts), joinedload(Patient.follow_ups),
        joinedload(Patient.notes), joinedload(Patient.summary)
    ).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return serialize_patient(patient)


@v1_router.post("/patients", response_model=PatientOut, status_code=201)
def create_patient(data: PatientCreate, db: Session = Depends(get_db)):
    if db.query(Patient).filter(Patient.mir_number == data.mir_number).first():
        raise HTTPException(status_code=422, detail="Patient with this MIR number already exists")

    patient = Patient(
        mir_number=data.mir_number, first_name=data.first_name, last_name=data.last_name,
        date_of_birth=data.date_of_birth, gender=data.gender, admission_date=data.admission_date,
        discharge_date=data.discharge_date, ward=data.ward,
        attending_physician=data.attending_physician, status=data.status,
    )
    db.add(patient)
    db.flush()

    for d in data.diagnoses:
        db.add(Diagnosis(patient_id=patient.id, **d.model_dump()))
    for m in data.medications:
        db.add(Medication(patient_id=patient.id, **m.model_dump()))
    for r in data.risk_alerts:
        db.add(RiskAlert(patient_id=patient.id, **r.model_dump()))
    for f in data.follow_ups:
        db.add(FollowUp(patient_id=patient.id, **f.model_dump()))
    for n in data.notes:
        db.add(Note(patient_id=patient.id, **n.model_dump()))

    db.commit()
    db.refresh(patient)
    return serialize_patient(patient)


@v1_router.delete("/patients/{patient_id}", status_code=204)
def delete_patient(patient_id: int, db: Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    db.delete(patient)
    db.commit()
    return None


# ── Clinical record CRUD ─────────────────────────────────────────

@v1_router.post("/patients/{patient_id}/diagnoses", response_model=DiagnosisOut, status_code=201)
def add_diagnosis(patient_id: int, data: DiagnosisCreate, db: Session = Depends(get_db)):
    if not db.query(Patient).filter(Patient.id == patient_id).first():
        raise HTTPException(status_code=404, detail="Patient not found")
    obj = Diagnosis(patient_id=patient_id, **data.model_dump())
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return _serialize_diagnosis(obj)

@v1_router.delete("/patients/{patient_id}/diagnoses/{record_id}", status_code=204)
def delete_diagnosis(patient_id: int, record_id: int, db: Session = Depends(get_db)):
    obj = db.query(Diagnosis).filter(Diagnosis.id == record_id, Diagnosis.patient_id == patient_id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Diagnosis not found")
    db.delete(obj)
    db.commit()
    return None

@v1_router.post("/patients/{patient_id}/medications", response_model=MedicationOut, status_code=201)
def add_medication(patient_id: int, data: MedicationCreate, db: Session = Depends(get_db)):
    if not db.query(Patient).filter(Patient.id == patient_id).first():
        raise HTTPException(status_code=404, detail="Patient not found")
    obj = Medication(patient_id=patient_id, **data.model_dump())
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return _serialize_medication(obj)

@v1_router.delete("/patients/{patient_id}/medications/{record_id}", status_code=204)
def delete_medication(patient_id: int, record_id: int, db: Session = Depends(get_db)):
    obj = db.query(Medication).filter(Medication.id == record_id, Medication.patient_id == patient_id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Medication not found")
    db.delete(obj)
    db.commit()
    return None

@v1_router.post("/patients/{patient_id}/risk-alerts", response_model=RiskAlertOut, status_code=201)
def add_risk_alert(patient_id: int, data: RiskAlertCreate, db: Session = Depends(get_db)):
    if not db.query(Patient).filter(Patient.id == patient_id).first():
        raise HTTPException(status_code=404, detail="Patient not found")
    obj = RiskAlert(patient_id=patient_id, **data.model_dump())
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return _serialize_risk_alert(obj)

@v1_router.delete("/patients/{patient_id}/risk-alerts/{record_id}", status_code=204)
def delete_risk_alert(patient_id: int, record_id: int, db: Session = Depends(get_db)):
    obj = db.query(RiskAlert).filter(RiskAlert.id == record_id, RiskAlert.patient_id == patient_id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Risk alert not found")
    db.delete(obj)
    db.commit()
    return None

@v1_router.post("/patients/{patient_id}/follow-ups", response_model=FollowUpOut, status_code=201)
def add_follow_up(patient_id: int, data: FollowUpCreate, db: Session = Depends(get_db)):
    if not db.query(Patient).filter(Patient.id == patient_id).first():
        raise HTTPException(status_code=404, detail="Patient not found")
    obj = FollowUp(patient_id=patient_id, **data.model_dump())
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return _serialize_follow_up(obj)

@v1_router.delete("/patients/{patient_id}/follow-ups/{record_id}", status_code=204)
def delete_follow_up(patient_id: int, record_id: int, db: Session = Depends(get_db)):
    obj = db.query(FollowUp).filter(FollowUp.id == record_id, FollowUp.patient_id == patient_id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Follow-up not found")
    db.delete(obj)
    db.commit()
    return None

@v1_router.post("/patients/{patient_id}/notes", response_model=NoteOut, status_code=201)
def add_note(patient_id: int, data: NoteCreate, db: Session = Depends(get_db)):
    if not db.query(Patient).filter(Patient.id == patient_id).first():
        raise HTTPException(status_code=404, detail="Patient not found")
    obj = Note(patient_id=patient_id, **data.model_dump())
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return _serialize_note(obj)

@v1_router.delete("/patients/{patient_id}/notes/{record_id}", status_code=204)
def delete_note(patient_id: int, record_id: int, db: Session = Depends(get_db)):
    obj = db.query(Note).filter(Note.id == record_id, Note.patient_id == patient_id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Note not found")
    db.delete(obj)
    db.commit()
    return None


# ── Summary Generation ────────────────────────────────────────────

@v1_router.post("/patients/{patient_id}/generate-summary")
async def generate_summary(patient_id: int, db: Session = Depends(get_db)):
    patient = db.query(Patient).options(
        joinedload(Patient.diagnoses), joinedload(Patient.medications),
        joinedload(Patient.risk_alerts), joinedload(Patient.follow_ups),
        joinedload(Patient.notes)
    ).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    prompt = build_prompt(patient)

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        result_text = response.text.strip()
        # Remove markdown code blocks if present
        if result_text.startswith('```json'):
            result_text = result_text[7:]
        if result_text.endswith('```'):
            result_text = result_text[:-3]
        result = json.loads(result_text)
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        # Fallback to simple summary
        def _join_items(items, attr="description"):
            return ", ".join([getattr(i, attr) for i in items]) if items else "none recorded"

        diagnoses_text = _join_items(patient.diagnoses, "description")
        meds_text = _join_items(patient.medications, "name")
        risks_text = _join_items(patient.risk_alerts, "description")
        followups_text = _join_items(patient.follow_ups, "department")

        stay_desc = f"Admission: {patient.admission_date}. Discharge: {patient.discharge_date or 'N/A'}. Diagnoses: {diagnoses_text}. Medications given: {meds_text}. Risk alerts: {risks_text}. Follow-ups: {followups_text}."
        patient_friendly = f"You were admitted on {patient.admission_date.date()} and discharged on {(patient.discharge_date.date() if patient.discharge_date else 'N/A')}. Main diagnoses: {diagnoses_text}. Medications: {meds_text}. Please follow up with: {followups_text}."

        result = {
            "stayDescription": stay_desc,
            "patientFriendlySummary": patient_friendly,
            "recommendations": "Follow up with your primary care physician within 1 week. Monitor for any signs of infection or complications.",
            "lifestyleTips": "Get adequate rest, stay hydrated, and avoid strenuous activities for the next few weeks.",
            "foodHealthTips": "Eat a balanced diet rich in fruits, vegetables, and lean proteins. Avoid excessive salt and processed foods.",
        }

    existing = db.query(DischargeSummary).filter(DischargeSummary.patient_id == patient_id).first()
    if existing:
        existing.stay_description = result.get("stayDescription", "")
        existing.patient_friendly_summary = result.get("patientFriendlySummary", "")
        existing.recommendations = result.get("recommendations", "")
        existing.lifestyle_tips = result.get("lifestyleTips", "")
        existing.food_health_tips = result.get("foodHealthTips", "")
        existing.generated_at = datetime.now(timezone.utc)
    else:
        db.add(DischargeSummary(
            patient_id=patient_id,
            stay_description=result.get("stayDescription", ""),
            patient_friendly_summary=result.get("patientFriendlySummary", ""),
            recommendations=result.get("recommendations", ""),
            lifestyle_tips=result.get("lifestyleTips", ""),
            food_health_tips=result.get("foodHealthTips", ""),
        ))
    db.commit()

    return {
        "stayDescription": result.get("stayDescription", ""),
        "patientFriendlySummary": result.get("patientFriendlySummary", ""),
        "recommendations": result.get("recommendations", ""),
        "lifestyleTips": result.get("lifestyleTips", ""),
        "foodHealthTips": result.get("foodHealthTips", ""),
    }


@v1_router.get("/patients/{patient_id}/summary")
def get_summary(patient_id: int, db: Session = Depends(get_db)):
    patient = db.query(Patient).options(
        joinedload(Patient.summary), joinedload(Patient.medications), joinedload(Patient.follow_ups)
    ).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    if not patient.summary:
        raise HTTPException(status_code=404, detail="No summary generated yet")
    return {
        "id": patient.summary.id,
        "patient_id": patient.summary.patient_id,
        "stayDescription": patient.summary.stay_description,
        "patientFriendlySummary": patient.summary.patient_friendly_summary,
        "recommendations": patient.summary.recommendations,
        "lifestyleTips": patient.summary.lifestyle_tips,
        "foodHealthTips": patient.summary.food_health_tips,
        "generated_at": patient.summary.generated_at,
        "patient_name": f"{patient.first_name} {patient.last_name}",
        "mir_number": patient.mir_number,
        "medications": [{"name": m.name, "dosage": m.dosage, "frequency": m.frequency, "instructions": m.instructions} for m in patient.medications],
        "follow_ups": [{"department": f.department, "physician": f.physician, "scheduled_date": str(f.scheduled_date) if f.scheduled_date else None, "instructions": f.instructions} for f in patient.follow_ups],
    }


# ── Wire up routers ──────────────────────────────────────────────

api_router.include_router(v1_router)
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    init_db()
    genai.configure(api_key=os.environ['GEMINI_API_KEY'])
    logger.info("Discharge IQ API started — MySQL tables created/verified")

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "service": "Discharge IQ API"}
