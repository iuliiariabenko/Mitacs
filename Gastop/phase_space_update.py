from fastapi import APIRouter
import numpy as np
from ..services.phase_space import A_nullcline, P_nullcline, risk_map
from ..config import params

router = APIRouter()

@router.get("/phase-map")
def get_phase_map(h: float, H: float = 0.8, n: int = 200):
    A = np.linspace(0,1,n)
    P = np.linspace(0,1,n)
    AA, PP = np.meshgrid(A,P)

    R = risk_map(AA, PP, h, params)
    Pline = np.linspace(0,1,n)

    return dict(
        A=A.tolist(),
        P=P.tolist(),
        R=R.tolist(),
        A_null=A_nullcline(Pline, H, h, params).tolist(),
        P_null=P_nullcline(Pline, H, params).tolist(),
        Ac=params.Ac
    )
