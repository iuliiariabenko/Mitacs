from pydantic import BaseModel
from datetime import datetime

class CampaignCreate(BaseModel):
    name: str
    customer: str | None = None
    objective: str | None = None

class CampaignOut(CampaignCreate):
    id: int
    created_at: datetime

class ExperimentCreate(BaseModel):
    campaign_id: int
    run_name: str
    operator: str | None = None

class ExperimentOut(ExperimentCreate):
    id: int
    created_at: datetime

class PlanCreate(BaseModel):
    experiment_id: int
    name: str

class PlanStepCreate(BaseModel):
    step_index: int
    duration_s: int = 300
    temperature_C: float
    pressure_bar: float
    oxygen_ppm: float
    hefa_fraction: float
    note: str | None = None

class PlanOut(BaseModel):
    id: int
    experiment_id: int
    name: str
    created_at: datetime

class PresignOut(BaseModel):
    upload_url: str
    s3_uri: str
