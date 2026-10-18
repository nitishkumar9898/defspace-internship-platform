"""P3 — RC Trainer Aircraft DBT
   Build log + flight test log storage.
   Preliminary analysis calculations + performance prediction.
"""
import json
import math
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from models import RCAircraftLog
from schemas import FlightLogEntry, FlightLogResult, AircraftAnalysisInput, AircraftAnalysisResult

router = APIRouter(prefix="/api/p3", tags=["P3 RC Aircraft"])

RHO = 1.225        # kg/m³ air density at sea level
G   = 9.81         # m/s²
CL_MAX = 1.2       # typical flat-plate / Clark-Y wing


@router.post("/analyse", response_model=AircraftAnalysisResult)
async def analyse_aircraft(data: AircraftAnalysisInput):
    span_m    = data.wingspan_mm / 1000
    chord_m   = data.chord_mm / 1000
    fuselage_m = data.fuselage_length_mm / 1000
    auw_kg    = data.auw_g / 1000
    auw_n     = auw_kg * G

    s_w = span_m * chord_m
    wing_loading = auw_n / s_w          # N/m²

    v_stall = math.sqrt(2 * auw_n / (RHO * s_w * CL_MAX))

    # Cruise ~ 1.3 * v_stall for trainers
    v_cruise = 1.3 * v_stall

    # Prop thrust estimate: T ~ 0.7 * kV * cells * 0.84 * pitch effect (rough)
    nominal_voltage = data.battery_cells * 3.7
    prop_d_m = data.prop_diameter_in * 0.0254
    prop_pitch_m = prop_d_m * 0.45          # assume pitch ≈ 0.45 * diameter
    rpm_max = data.motor_kv * nominal_voltage
    # Static thrust formula (Adkins-Liebeck approximation)
    thrust_n = 0.000015 * (rpm_max ** (4/3)) * (prop_d_m ** (2/3))
    thrust_n = min(thrust_n, auw_n * 2.5)  # cap sanity
    tw_ratio = thrust_n / auw_n

    # Recommended CG: 25-35% chord from LE → use 30%
    cg_mm = round(0.30 * data.chord_mm, 1)

    # Reynolds number at cruise
    visc = 1.5e-5   # kinematic viscosity air
    re = v_cruise * chord_m / visc

    # Endurance estimate (very rough): energy = battery_mah * cells * 3.7 * 3600 J
    energy_j = (data.battery_mah / 1000) * nominal_voltage * 3600
    cruise_power_w = thrust_n * v_cruise / 0.75   # prop efficiency ~75%
    endurance_min = (energy_j / cruise_power_w) / 60

    return AircraftAnalysisResult(
        wing_area_m2=round(s_w, 4),
        wing_loading_n_m2=round(wing_loading, 2),
        stall_speed_ms=round(v_stall, 2),
        cruise_speed_ms=round(v_cruise, 2),
        tw_ratio=round(tw_ratio, 3),
        cg_recommended_mm=cg_mm,
        reynolds_number=round(re),
        endurance_min=round(endurance_min, 1),
    )


@router.post("/log", response_model=FlightLogResult)
async def add_log(entry: FlightLogEntry, db: Session = Depends(get_db)):
    valid_types   = {"build", "test", "preflight"}
    valid_status  = {"pass", "fail", "pending"}
    if entry.log_type not in valid_types:
        raise HTTPException(400, f"log_type must be one of {valid_types}")
    if entry.status not in valid_status:
        raise HTTPException(400, f"status must be one of {valid_status}")

    record = RCAircraftLog(
        log_type=entry.log_type,
        phase=entry.phase,
        notes=entry.notes,
        status=entry.status,
        parameters_json=json.dumps(entry.parameters or {}),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return FlightLogResult(
        id=record.id,
        log_type=record.log_type,
        phase=record.phase,
        notes=record.notes,
        status=record.status,
        parameters=json.loads(record.parameters_json),
        created_at=record.created_at.isoformat() if record.created_at else "",
    )


@router.get("/logs")
async def get_logs(log_type: str = None, db: Session = Depends(get_db)):
    q = db.query(RCAircraftLog)
    if log_type:
        q = q.filter(RCAircraftLog.log_type == log_type)
    records = q.order_by(RCAircraftLog.created_at.desc()).all()
    return [
        {
            "id": r.id,
            "log_type": r.log_type,
            "phase": r.phase,
            "notes": r.notes,
            "status": r.status,
            "parameters": json.loads(r.parameters_json or "{}"),
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in records
    ]


@router.get("/specs")
async def get_specs():
    """Return the fixed build specifications for the CUJ RC Trainer."""
    return {
        "airframe": {
            "wingspan_mm": 1050,
            "fuselage_length_mm": 720,
            "auw_g": 860,
            "wing_area_dm2": 19.3,
            "wing_loading_n_m2": 44.6,
            "material": "5mm foam board + carbon spars",
        },
        "electronics": {
            "motor": "2206 BLDC 1600-2600KV",
            "esc": "30A BEC ESC",
            "battery": "2S 2200mAh 25C LiPo",
            "propeller": "8x4.5 APC",
            "receiver": "6-ch 2.4GHz",
            "servos": "4 × 9g micro servo",
        },
        "performance": {
            "stall_speed_ms": 7.2,
            "cruise_speed_ms": 12.0,
            "tw_ratio": 0.58,
            "cg_mm_from_le": 45,
            "flight_test": "Successful",
        },
    }
