from sqlalchemy import String, Integer, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from datetime import datetime
from .db import Base

class Campaign(Base):
    __tablename__ = "campaign"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    customer: Mapped[str | None] = mapped_column(String(200), nullable=True)
    objective: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    experiments = relationship("Experiment", back_populates="campaign")

class Experiment(Base):
    __tablename__ = "experiment"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    campaign_id: Mapped[int] = mapped_column(ForeignKey("campaign.id"))
    run_name: Mapped[str] = mapped_column(String(200))
    operator: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    campaign = relationship("Campaign", back_populates="experiments")
    points = relationship("MeasurementPoint", back_populates="experiment")

class MeasurementPoint(Base):
    __tablename__ = "measurement_point"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiment.id"))
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    temperature_C: Mapped[float | None] = mapped_column(Float, nullable=True)
    pressure_bar: Mapped[float | None] = mapped_column(Float, nullable=True)
    oxygen_ppm: Mapped[float | None] = mapped_column(Float, nullable=True)
    hefa_fraction: Mapped[float | None] = mapped_column(Float, nullable=True)
    flow_rate: Mapped[float | None] = mapped_column(Float, nullable=True)

    experiment = relationship("Experiment", back_populates="points")
    sensor_data = relationship("SensorData", back_populates="point")

class SensorData(Base):
    __tablename__ = "sensor_data"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    mp_id: Mapped[int] = mapped_column(ForeignKey("measurement_point.id"))
    sensor_type: Mapped[str] = mapped_column(String(50))  # Raman/UV/SERS/...
    storage_uri: Mapped[str] = mapped_column(Text)        # s3://bucket/key
    checksum: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    point = relationship("MeasurementPoint", back_populates="sensor_data")

class Plan(Base):
    __tablename__ = "plan"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiment.id"))
    name: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class PlanStep(Base):
    __tablename__ = "plan_step"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    plan_id: Mapped[int] = mapped_column(ForeignKey("plan.id"))
    step_index: Mapped[int] = mapped_column(Integer)
    duration_s: Mapped[int] = mapped_column(Integer, default=300)

    temperature_C: Mapped[float] = mapped_column(Float)
    pressure_bar: Mapped[float] = mapped_column(Float)
    oxygen_ppm: Mapped[float] = mapped_column(Float)
    hefa_fraction: Mapped[float] = mapped_column(Float)

    note: Mapped[str | None] = mapped_column(Text, nullable=True)
