from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ..db import get_db
from ..models import PlanStep, MeasurementPoint

router = APIRouter()

@router.post("/plan/{plan_id}/run")
def run_plan(plan_id: int, db: Session = Depends(get_db)):
    steps = db.query(PlanStep).filter(PlanStep.plan_id == plan_id).order_by(PlanStep.step_index).all()

    created = []
    for s in steps:
        mp = MeasurementPoint(
            experiment_id=s.plan.experiment_id,
            temperature_C=s.temperature_C,
            pressure_bar=s.pressure_bar,
            oxygen_ppm=s.oxygen_ppm,
            hefa_fraction=s.hefa_fraction
        )
        db.add(mp)
        db.flush()
        created.append(mp.id)

    db.commit()
    return {"status": "running", "measurement_points": created}
