from pydantic import BaseModel
from typing import List, Optional, Dict, Any


# ── P1 CG & Stability ───────────────────────────────────────────────────────
class Component(BaseModel):
    name: str
    weight_kg: float
    arm_m: float          # distance from nose/datum in metres

class CGInput(BaseModel):
    session_id: Optional[str] = "default"
    components: List[Component]
    mac: float            # mean aerodynamic chord in metres
    neutral_point_m: float  # neutral point from datum
    wing_ac_m: float      # aerodynamic centre from datum

class CGResult(BaseModel):
    cg_m: float
    total_weight_kg: float
    total_weight_n: float
    static_margin_pct: float
    neutral_point_m: float
    is_stable: bool
    stability_grade: str
    ml_stability_score: float
    ml_accuracy_pct: float
    component_contributions: List[Dict[str, Any]]


# ── P2 Neutral Point ─────────────────────────────────────────────────────────
class NPInput(BaseModel):
    session_id: Optional[str] = "default"
    wing_span_m: float
    wing_chord_m: float
    wing_ac_fraction: float     # as fraction of chord from LE (typically 0.25)
    tail_span_m: float
    tail_chord_m: float
    tail_moment_arm_m: float    # distance from wing AC to tail AC
    cg_m: float                 # CG from nose
    tail_efficiency: Optional[float] = 0.9

class NPResult(BaseModel):
    wing_area_m2: float
    tail_area_m2: float
    wing_ac_m: float
    neutral_point_m: float
    static_margin_m: float
    static_margin_pct_mac: float
    is_stable: bool
    stability_grade: str
    np_vs_sep: List[Dict[str, float]]  # for chart: NP vs tail separation
    sm_vs_cg: List[Dict[str, float]]   # for chart: SM vs CG position


# ── P3 RC Aircraft ───────────────────────────────────────────────────────────
class FlightLogEntry(BaseModel):
    log_type: str         # "build" | "test" | "preflight"
    phase: str
    notes: str
    status: str           # "pass" | "fail" | "pending"
    parameters: Optional[Dict[str, Any]] = {}

class FlightLogResult(BaseModel):
    id: int
    log_type: str
    phase: str
    notes: str
    status: str
    parameters: Dict[str, Any]
    created_at: str

class AircraftAnalysisInput(BaseModel):
    wingspan_mm: float
    chord_mm: float
    fuselage_length_mm: float
    auw_g: float
    motor_kv: int
    battery_cells: int
    battery_mah: int
    prop_diameter_in: float

class AircraftAnalysisResult(BaseModel):
    wing_area_m2: float
    wing_loading_n_m2: float
    stall_speed_ms: float
    cruise_speed_ms: float
    tw_ratio: float
    cg_recommended_mm: float
    reynolds_number: float
    endurance_min: float


# ── P4 Delta Wing Aerodynamics ────────────────────────────────────────────────
class DeltaWingInput(BaseModel):
    session_id: Optional[str] = "default"
    cl_alpha: float        # /rad lift curve slope
    cd0: float             # zero-lift drag
    aspect_ratio: float
    oswald_e: float
    cm0: float             # pitching moment at zero lift
    cm_alpha: float        # dCm/dAlpha /rad
    cm_delta_e: float      # elevator effectiveness /rad
    wing_area_m2: float
    aircraft_weight_n: float
    rho: Optional[float] = 1.225

class AeroPoint(BaseModel):
    alpha_deg: float
    cl: float
    cd: float
    ld: float
    velocity_ms: float
    thrust_n: float
    power_w: float
    cl15_cd: float         # CL^1.5 / CD (endurance parameter)

class DeltaWingResult(BaseModel):
    trim_table: List[AeroPoint]
    max_ld: float
    max_ld_alpha: float
    min_drag_v: float
    min_thrust_n: float
    min_power_w: float
    stall_speed_ms: float
    drag_polar: List[Dict[str, float]]
    ml_r2_score: float
    ml_predictions: List[Dict[str, float]]


# ── P5 VTOL Drone ─────────────────────────────────────────────────────────────
class VTOLInput(BaseModel):
    session_id: Optional[str] = "default"
    auw_g: float
    num_motors: int
    prop_diameter_in: float
    prop_pitch_in: float
    battery_mah: int
    battery_cells: int
    motor_kv: int
    payload_g: float

class VTOLResult(BaseModel):
    prop_disk_area_m2: float
    disk_loading_n_m2: float
    thrust_per_motor_n: float
    total_thrust_n: float
    tw_ratio: float
    hover_throttle_pct: float
    induced_velocity_ms: float
    hover_power_per_motor_w: float
    total_hover_power_w: float
    battery_wh: float
    flight_time_min: float
    max_climb_rate_ms: float
    thrust_curve: List[Dict[str, float]]
    flight_time_vs_payload: List[Dict[str, float]]
    ml_flight_time_min: float
    ml_accuracy_pct: float
    optimiser_solution: Dict[str, Any]


# ── P6 GenAI Skill Analysis ───────────────────────────────────────────────────
class SkillAnalysisInput(BaseModel):
    session_id: Optional[str] = "default"
    role_title: str
    jd_text: str
    resume_text: str

class SkillEntry(BaseModel):
    skill: str
    frequency: int
    in_resume: bool
    action: str           # "have" | "learn" | "demonstrate" | "skip"
    confidence: float     # 0-1

class SkillAnalysisResult(BaseModel):
    similarity_score: float
    skills: List[SkillEntry]
    top_missing: List[str]
    interview_questions: List[str]
    thirty_day_plan: List[Dict[str, str]]
    explanation_12th: str
    explanation_professional: str
    explanation_fiveyear: str
    ml_match_pct: float
