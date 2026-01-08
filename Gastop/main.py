import os
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from .db import Base, engine, get_db
from . import schemas, crud
from .services.storage import ensure_bucket, presign_put

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Fuel EDM API")

@app.on_event("startup")
def _startup():
    ensure_bucket(os.getenv("S3_BUCKET", "spectra"))

@app.post("/campaign", response_model=schemas.CampaignOut)
def api_create_campaign(payload: schemas.CampaignCreate, db: Session = Depends(get_db)):
    obj = crud.create_campaign(db, payload.name, payload.customer, payload.objective)
    return obj

@app.post("/experiment", response_model=schemas.ExperimentOut)
def api_create_experiment(payload: schemas.ExperimentCreate, db: Session = Depends(get_db)):
    obj = crud.create_experiment(db, payload.campaign_id, payload.run_name, payload.operator)
    return obj

@app.post("/plan", response_model=schemas.PlanOut)
def api_create_plan(payload: schemas.PlanCreate, db: Session = Depends(get_db)):
    obj = crud.create_plan(db, payload.experiment_id, payload.name)
    return obj

@app.post("/plan/{plan_id}/steps")
def api_add_steps(plan_id: int, steps: list[schemas.PlanStepCreate], db: Session = Depends(get_db)):
    crud.add_plan_steps(db, plan_id, [s.model_dump() for s in steps])
    return {"status": "ok", "count": len(steps)}

@app.get("/plan/{plan_id}/steps")
def api_list_steps(plan_id: int, db: Session = Depends(get_db)):
    steps = crud.list_plan_steps(db, plan_id)
    return [dict(
        id=s.id, step_index=s.step_index, duration_s=s.duration_s,
        temperature_C=s.temperature_C, pressure_bar=s.pressure_bar,
        oxygen_ppm=s.oxygen_ppm, hefa_fraction=s.hefa_fraction, note=s.note
    ) for s in steps]

@app.post("/presign")
def api_presign(sensor_type: str, mp_id: int):
    bucket = os.getenv("S3_BUCKET", "spectra")
    key = f"{sensor_type}/mp_{mp_id}.h5"
    url = presign_put(bucket, key)
    return {"upload_url": url, "s3_uri": f"s3://{bucket}/{key}"}
