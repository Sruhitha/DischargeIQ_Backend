from sqlalchemy import create_engine, Column, Integer, String, Text, Date, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime, timezone
import os
from dotenv import load_dotenv
from pathlib import Path

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

MYSQL_URL = os.environ['MYSQL_URL']
engine = create_engine(MYSQL_URL, pool_pre_ping=True, pool_recycle=3600)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


class Patient(Base):
    __tablename__ = 'patients'
    id = Column(Integer, primary_key=True, autoincrement=True)
    mir_number = Column(String(50), unique=True, nullable=False, index=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    date_of_birth = Column(Date, nullable=False)
    gender = Column(String(20), nullable=False)
    admission_date = Column(DateTime, nullable=False)
    discharge_date = Column(DateTime, nullable=True)
    ward = Column(String(100))
    attending_physician = Column(String(200))
    status = Column(String(20), default='admitted')
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    diagnoses = relationship('Diagnosis', back_populates='patient', cascade='all, delete-orphan')
    medications = relationship('Medication', back_populates='patient', cascade='all, delete-orphan')
    risk_alerts = relationship('RiskAlert', back_populates='patient', cascade='all, delete-orphan')
    follow_ups = relationship('FollowUp', back_populates='patient', cascade='all, delete-orphan')
    notes = relationship('Note', back_populates='patient', cascade='all, delete-orphan')
    summary = relationship('DischargeSummary', back_populates='patient', uselist=False, cascade='all, delete-orphan')


class Diagnosis(Base):
    __tablename__ = 'diagnoses'
    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey('patients.id', ondelete='CASCADE'), nullable=False)
    code = Column(String(20))
    description = Column(String(500), nullable=False)
    diagnosis_type = Column(String(20), default='secondary')
    diagnosed_date = Column(Date, nullable=True)
    patient = relationship('Patient', back_populates='diagnoses')


class Medication(Base):
    __tablename__ = 'medications'
    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey('patients.id', ondelete='CASCADE'), nullable=False)
    name = Column(String(200), nullable=False)
    dosage = Column(String(100))
    frequency = Column(String(100))
    route = Column(String(50))
    start_date = Column(Date, nullable=True)
    end_date = Column(Date, nullable=True)
    instructions = Column(Text)
    patient = relationship('Patient', back_populates='medications')


class RiskAlert(Base):
    __tablename__ = 'risk_alerts'
    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey('patients.id', ondelete='CASCADE'), nullable=False)
    alert_type = Column(String(50), nullable=False)
    severity = Column(String(20), default='medium')
    description = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    patient = relationship('Patient', back_populates='risk_alerts')


class FollowUp(Base):
    __tablename__ = 'follow_ups'
    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey('patients.id', ondelete='CASCADE'), nullable=False)
    department = Column(String(100))
    physician = Column(String(200))
    scheduled_date = Column(Date, nullable=True)
    instructions = Column(Text)
    patient = relationship('Patient', back_populates='follow_ups')


class Note(Base):
    __tablename__ = 'notes'
    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey('patients.id', ondelete='CASCADE'), nullable=False)
    author = Column(String(200))
    content = Column(Text, nullable=False)
    note_type = Column(String(50), default='progress')
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    patient = relationship('Patient', back_populates='notes')


class DischargeSummary(Base):
    __tablename__ = 'discharge_summaries'
    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey('patients.id', ondelete='CASCADE'), nullable=False, unique=True)
    stay_description = Column(Text)
    patient_friendly_summary = Column(Text)
    recommendations = Column(Text)
    lifestyle_tips = Column(Text)
    food_health_tips = Column(Text)
    generated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    patient = relationship('Patient', back_populates='summary')


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
