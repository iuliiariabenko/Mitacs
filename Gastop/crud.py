from sqlalchemy.orm import Session
from . import models

def create_campaign(db: Session, name: str, customer=None, objective=None):
    obj = models.Campaign(name=name, customer=customer, objective=objective)
    db.add(obj); db.commit(); db.refresh(obj)
    return obj

def create_experiment(db: Session, campaign_id: int, run_name: str, operator=None):
    obj = models.Experiment(campaign_id=campaign_id, run_name=run_name, operator=operator)
    db.add(obj); db.commit(); db.refresh(obj)
    return obj

def create_plan(db: Session, experiment_id: int, name: str):
    obj = models.Plan(experiment_id=experiment_id, name=name)
    db.add(obj); db.commit(); db.refresh(obj)
    return obj

def add_plan_steps(db: Session, plan_id: int, steps: list[dict]):
    for s in steps:
        db.add(models.PlanStep(plan_id=plan_id, **s))
    db.commit()

def list_plan_steps(db: Session, plan_id: int):
    return db.query(models.PlanStep).filter(models.PlanStep.plan_id == plan_id).order_by(models.PlanStep.step_index).all()
