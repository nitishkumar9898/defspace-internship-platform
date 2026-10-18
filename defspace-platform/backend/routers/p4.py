"""P4 — Cropped Delta Wing UAV Aerodynamic Analysis
   Full alpha-sweep using scipy + sklearn MLP for coefficient prediction.
   Most compute-heavy router.
"""
import json
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.optimize import minimize_scalar
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database import get_db
from models import DeltaWingAnalysis
from schemas import DeltaWingInput, DeltaWingResult, AeroPoint

router = APIRouter(prefix="/api/p4", tags=["P4 Delta Wing"])

RHO = 1.225


# ── Train MLP on synthetic NACA thin-airfoil / delta-wing data ──────────────
def _build_aero_mlp():
    np.random.seed(42)
    n = 2000
    alpha_r = np.random.uniform(-5, 18, n)          # degrees
    ar      = np.random.uniform(1.5, 6.0, n)        # aspect ratio
    e       = np.random.uniform(0.6, 0.95, n)       # Oswald
    cd0     = np.random.uniform(0.01, 0.04, n)

    cl_alpha = np.pi * ar / (1 + np.sqrt(1 + (ar / 2) ** 2))  # DATCOM
    cl = cl_alpha * np.radians(alpha_r)
    cd = cd0 + cl**2 / (np.pi * ar * e)
    ld = np.where(cd > 0, cl / cd, 0)

    X = np.column_stack([alpha_r, ar, e, cd0])
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    mlp_cl = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=0).fit(X_s, cl)
    mlp_cd = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=0).fit(X_s, cd)

    acc_cl = r2_score(cl, mlp_cl.predict(X_s))
    acc_cd = r2_score(cd, mlp_cd.predict(X_s))
    avg_r2 = (acc_cl + acc_cd) / 2
    return mlp_cl, mlp_cd, scaler, avg_r2

_mlp_cl, _mlp_cd, _aero_scaler, _mlp_r2 = _build_aero_mlp()


def _compute_aero_point(alpha_deg, cl_alpha, cd0, ar, e, wing_area, weight_n, rho):
    alpha_r = np.radians(alpha_deg)
    cl = cl_alpha * alpha_r
    cd = cd0 + cl**2 / (np.pi * ar * e)
    ld = cl / cd if cd > 1e-9 else 0.0
    v  = np.sqrt(2 * weight_n / (rho * wing_area * cl)) if cl > 0.01 else 0.0
    thrust = cd / cl * weight_n if cl > 0.01 else 999.0
    power  = thrust * v if v > 0 else 0.0
    cl15cd = (cl ** 1.5) / cd if (cd > 1e-9 and cl > 0) else 0.0
    return AeroPoint(
        alpha_deg=round(alpha_deg, 1),
        cl=round(float(cl), 4),
        cd=round(float(cd), 5),
        ld=round(float(ld), 2),
        velocity_ms=round(float(v), 3),
        thrust_n=round(float(thrust), 4),
        power_w=round(float(power), 3),
        cl15_cd=round(float(cl15cd), 4),
    )


def _ml_predict_point(alpha_deg, ar, e, cd0):
    X = _aero_scaler.transform([[alpha_deg, ar, e, cd0]])
    cl = float(_mlp_cl.predict(X)[0])
    cd = float(_mlp_cd.predict(X)[0])
    ld = cl / cd if cd > 1e-9 else 0.0
    return {"alpha_deg": round(alpha_deg, 1),
            "ml_cl": round(cl, 4), "ml_cd": round(cd, 5), "ml_ld": round(ld, 2)}


@router.post("/calculate", response_model=DeltaWingResult)
async def calculate_delta_wing(data: DeltaWingInput, db: Session = Depends(get_db)):
    alpha_range = np.arange(-5, 16, 0.5)
    table = [
        _compute_aero_point(
            float(a), data.cl_alpha, data.cd0, data.aspect_ratio,
            data.oswald_e, data.wing_area_m2, data.aircraft_weight_n, data.rho
        )
        for a in alpha_range
    ]

    # Drag polar (CL vs CD)
    drag_polar = [{"cl": p.cl, "cd": p.cd} for p in table if p.cl >= 0]

    # Best L/D
    ld_vals = [(p.ld, p.alpha_deg) for p in table if p.cl > 0.05]
    max_ld, max_ld_alpha = max(ld_vals, key=lambda x: x[0]) if ld_vals else (0, 0)

    # Min drag velocity (best glide)
    valid = [p for p in table if p.velocity_ms > 0 and p.cl > 0.05]
    min_thrust_point = min(valid, key=lambda p: p.thrust_n)

    # Min power
    min_power_point = min(valid, key=lambda p: p.power_w)

    # Stall: CL drops — use max CL point
    cl_values = [p.cl for p in table if p.cl > 0]
    max_cl = max(cl_values) if cl_values else 1.2
    cl_max_point = next((p for p in reversed(table) if p.cl == max(cl_values)), table[-1])
    stall_v = cl_max_point.velocity_ms

    # ML predictions
    ml_preds = [_ml_predict_point(float(a), data.aspect_ratio, data.oswald_e, data.cd0)
                for a in np.arange(-5, 16, 2)]

    record = DeltaWingAnalysis(
        session_id=data.session_id,
        cl_alpha=data.cl_alpha, cd0=data.cd0,
        aspect_ratio=data.aspect_ratio, oswald_e=data.oswald_e,
        wing_area=data.wing_area_m2, aircraft_weight=data.aircraft_weight_n,
        results_json=json.dumps([p.model_dump() for p in table[:20]]),  # store sample
        max_ld=round(max_ld, 2),
        trim_velocity=round(min_thrust_point.velocity_ms, 2),
        min_thrust=round(min_thrust_point.thrust_n, 4),
        ml_accuracy=round(_mlp_r2 * 100, 2),
    )
    db.add(record)
    db.commit()

    return DeltaWingResult(
        trim_table=table,
        max_ld=round(max_ld, 2),
        max_ld_alpha=round(max_ld_alpha, 1),
        min_drag_v=round(min_thrust_point.velocity_ms, 3),
        min_thrust_n=round(min_thrust_point.thrust_n, 4),
        min_power_w=round(min_power_point.power_w, 3),
        stall_speed_ms=round(stall_v, 3),
        drag_polar=drag_polar,
        ml_r2_score=round(_mlp_r2, 4),
        ml_predictions=ml_preds,
    )


@router.get("/model-info")
async def model_info():
    return {
        "model": "MLP Neural Network (64→32 hidden units)",
        "targets": ["CL", "CD"],
        "r2_score": round(_mlp_r2, 4),
        "accuracy_pct": round(_mlp_r2 * 100, 2),
        "training_samples": 2000,
        "features": ["alpha_deg", "aspect_ratio", "oswald_e", "cd0"],
    }


@router.get("/history")
async def get_history(db: Session = Depends(get_db)):
    rows = db.query(DeltaWingAnalysis).order_by(DeltaWingAnalysis.created_at.desc()).limit(30).all()
    return [{"id": r.id, "max_ld": r.max_ld, "min_thrust_n": r.min_thrust,
             "trim_v": r.trim_velocity, "ml_acc": r.ml_accuracy,
             "created_at": r.created_at.isoformat() if r.created_at else None}
            for r in rows]
