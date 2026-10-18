"""P1 — CG & Static Stability Analysis
   Uses numpy for moment calculations + sklearn LinearRegression for ML stability scoring.
"""
import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from models import CGAnalysisResult
from schemas import CGInput, CGResult

router = APIRouter(prefix="/api/p1", tags=["P1 CG Stability"])

# ── Train a simple ML model on startup (synthetic data from aero textbook values)
def _build_stability_model():
    """Train a Linear Regression to predict stability score from SM%."""
    np.random.seed(42)
    # Features: [static_margin_pct, CG_frac_MAC, NP_frac_chord]
    # Target: stability_score (0-1)
    # Synthetic data based on standard aero stability criteria
    sm_vals = np.linspace(-20, 25, 200)
    stability = np.clip(0.5 + sm_vals / 40.0 + np.random.normal(0, 0.02, 200), 0, 1)
    X = sm_vals.reshape(-1, 1)
    y = stability
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    acc = r2_score(y, y_pred)
    return model, acc

_stability_model, _model_accuracy = _build_stability_model()


def _calc_cg(components) -> tuple[float, float]:
    """Returns (cg_m, total_weight_kg)"""
    total_w = sum(c.weight_kg for c in components)
    if total_w == 0:
        raise HTTPException(status_code=400, detail="Total weight cannot be zero")
    cg = sum(c.weight_kg * c.arm_m for c in components) / total_w
    return cg, total_w


def _grade_stability(sm: float) -> str:
    if sm >= 10:   return "Highly Stable"
    if sm >= 5:    return "Stable"
    if sm >= 0:    return "Marginally Stable"
    if sm >= -5:   return "Marginally Unstable"
    return "Unstable"


@router.post("/calculate", response_model=CGResult)
async def calculate_cg(data: CGInput, db: Session = Depends(get_db)):
    cg_m, total_w_kg = _calc_cg(data.components)

    static_margin_m = data.neutral_point_m - cg_m
    static_margin_pct = (static_margin_m / data.mac) * 100 if data.mac > 0 else 0.0
    is_stable = static_margin_pct >= 0

    ml_score_raw = float(_stability_model.predict([[static_margin_pct]])[0])
    ml_score = float(np.clip(ml_score_raw, 0.0, 1.0))

    contributions = [
        {
            "name": c.name,
            "weight_kg": round(c.weight_kg, 4),
            "arm_m": round(c.arm_m, 4),
            "moment_kgm": round(c.weight_kg * c.arm_m, 4),
            "weight_pct": round(c.weight_kg / total_w_kg * 100, 2),
        }
        for c in data.components
    ]

    record = CGAnalysisResult(
        session_id=data.session_id,
        components_json=json.dumps([c.model_dump() for c in data.components]),
        cg_position=round(cg_m, 4),
        total_weight=round(total_w_kg, 4),
        static_margin_pct=round(static_margin_pct, 2),
        neutral_point=round(data.neutral_point_m, 4),
        is_stable="stable" if is_stable else "unstable",
        ml_stability_score=round(ml_score, 4),
    )
    db.add(record)
    db.commit()

    return CGResult(
        cg_m=round(cg_m, 4),
        total_weight_kg=round(total_w_kg, 4),
        total_weight_n=round(total_w_kg * 9.81, 3),
        static_margin_pct=round(static_margin_pct, 2),
        neutral_point_m=round(data.neutral_point_m, 4),
        is_stable=is_stable,
        stability_grade=_grade_stability(static_margin_pct),
        ml_stability_score=round(ml_score * 100, 1),
        ml_accuracy_pct=round(_model_accuracy * 100, 2),
        component_contributions=contributions,
    )


@router.get("/history")
async def get_history(db: Session = Depends(get_db)):
    records = db.query(CGAnalysisResult).order_by(CGAnalysisResult.created_at.desc()).limit(50).all()
    return [
        {
            "id": r.id,
            "cg_m": r.cg_position,
            "total_weight_kg": r.total_weight,
            "static_margin_pct": r.static_margin_pct,
            "is_stable": r.is_stable,
            "ml_score": r.ml_stability_score,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in records
    ]


@router.get("/sensitivity")
async def cg_sensitivity():
    """Return sensitivity data: SM% vs NP position for frontend chart."""
    np_range = np.linspace(0.3, 1.2, 80)
    cg_fixed = 0.868
    mac = 0.23
    points = [
        {"np_m": round(float(np), 4), "sm_pct": round(float((np - cg_fixed) / mac * 100), 2)}
        for np in np_range
    ]
    return {"points": points, "cg_fixed_m": cg_fixed, "mac_m": mac}
