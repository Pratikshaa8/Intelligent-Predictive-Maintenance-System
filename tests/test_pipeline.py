from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.pipeline import expert_rules, validate_input

def sample(**kwargs):
    p=dict(machine_type="M",air_temp=298.1,process_temp=308.6,speed=1551,torque=42.8,wear=15,evidence=False)
    p.update(kwargs)
    return p

def test_healthy_default_rule():
    assert expert_rules(sample())[0:2] == ("No Failure","R6")

def test_tool_wear_rule():
    assert expert_rules(sample(wear=215))[0:2] == ("TWF (risk)","R4")

def test_heat_rule():
    assert expert_rules(sample(air_temp=300.8,process_temp=308.0,speed=1300,wear=100,torque=40))[0:2] == ("HDF","R2")

def test_overstrain_type_threshold():
    assert expert_rules(sample(machine_type="L",wear=191,torque=65.7))[0:2] == ("OSF","R1")

def test_invalid_negative_speed():
    errors,_=validate_input(sample(speed=-1))
    assert errors

def test_xgboost_pipeline():
    from src.pipeline import load_system, assess_machine
    from xgboost import XGBClassifier
    csv_path = Path(__file__).resolve().parents[1] / "data" / "ai4i2020.csv"
    system = load_system(csv_path)
    assert isinstance(system["ml"], XGBClassifier)
    assert "fallback" not in system["model_note"].lower()
    assert "xgboost" in system["model_note"].lower()
    assert system["holdout_auc"] > 0.90
    res = assess_machine(system, sample())
    assert 0.0 <= res["risk_score"] <= 1.0
    assert 0.0 <= res["ml_probability"] <= 1.0
