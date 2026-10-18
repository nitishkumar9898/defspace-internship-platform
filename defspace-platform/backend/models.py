from sqlalchemy import Column, Integer, Float, String, Text, DateTime, JSON
from sqlalchemy.sql import func
from database import Base


class CGAnalysisResult(Base):
    __tablename__ = "p1_cg_results"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    components_json = Column(Text)          # JSON array of {name, weight, arm}
    cg_position = Column(Float)            # metres from reference
    total_weight = Column(Float)           # kg
    static_margin_pct = Column(Float)      # % MAC
    neutral_point = Column(Float)
    is_stable = Column(String)             # "stable" | "unstable" | "neutral"
    ml_stability_score = Column(Float)     # 0-1 ML prediction
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class NeutralPointResult(Base):
    __tablename__ = "p2_np_results"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    wing_area = Column(Float)
    tail_area = Column(Float)
    wing_mac = Column(Float)
    wing_ac_fraction = Column(Float)
    tail_moment_arm = Column(Float)
    cg_position = Column(Float)
    neutral_point = Column(Float)
    static_margin = Column(Float)
    is_stable = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class RCAircraftLog(Base):
    __tablename__ = "p3_aircraft_logs"
    id = Column(Integer, primary_key=True, index=True)
    log_type = Column(String)              # "build" | "test" | "preflight"
    phase = Column(String)
    notes = Column(Text)
    status = Column(String)               # "pass" | "fail" | "pending"
    parameters_json = Column(Text)        # any key-value data
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class DeltaWingAnalysis(Base):
    __tablename__ = "p4_delta_wing"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    cl_alpha = Column(Float)
    cd0 = Column(Float)
    aspect_ratio = Column(Float)
    oswald_e = Column(Float)
    wing_area = Column(Float)
    aircraft_weight = Column(Float)
    results_json = Column(Text)            # full alpha sweep
    max_ld = Column(Float)
    trim_velocity = Column(Float)
    min_thrust = Column(Float)
    ml_accuracy = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class VTOLDesign(Base):
    __tablename__ = "p5_vtol"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    frame_size_mm = Column(Integer)
    num_motors = Column(Integer)
    motor_kv = Column(Integer)
    prop_diameter_in = Column(Float)
    battery_mah = Column(Integer)
    battery_cells = Column(Integer)
    auw_g = Column(Float)
    payload_g = Column(Float)
    thrust_per_motor_n = Column(Float)
    total_thrust_n = Column(Float)
    tw_ratio = Column(Float)
    hover_throttle_pct = Column(Float)
    hover_power_w = Column(Float)
    flight_time_min = Column(Float)
    optimised_params_json = Column(Text)
    ml_flight_time_prediction = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class GenAISkillAnalysis(Base):
    __tablename__ = "p6_genai"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    role_title = Column(String)
    jd_text = Column(Text)
    resume_text = Column(Text)
    skills_json = Column(Text)             # extracted skills + frequency
    gap_analysis_json = Column(Text)       # have / learn / demonstrate / skip
    similarity_score = Column(Float)       # TF-IDF cosine similarity (0-100)
    top_missing_skills = Column(Text)      # comma-separated
    created_at = Column(DateTime(timezone=True), server_default=func.now())
