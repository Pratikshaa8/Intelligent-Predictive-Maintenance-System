from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

FEATURES = ["Type_M","Type_L","Rotational speed [rpm]","Torque [Nm]","Temp_Diff_K","Wear_Torque_Product","Power_Deviation_W"]
TYPE_THRESHOLDS = {"L":11000,"M":12000,"H":13000}

def engineer(payload):
    temp_diff = float(payload["process_temp"]) - float(payload["air_temp"])
    power = float(payload["torque"]) * float(payload["speed"]) * 0.10472
    # Power deviation is measured against the dataset's nominal mean mechanical power.
    return {
        "Type_M": int(payload["machine_type"]=="M"),
        "Type_L": int(payload["machine_type"]=="L"),
        "Rotational speed [rpm]": float(payload["speed"]),
        "Torque [Nm]": float(payload["torque"]),
        "Temp_Diff_K": temp_diff,
        "Wear_Torque_Product": float(payload["wear"])*float(payload["torque"]),
        "Power_Deviation_W": power-6000.0,
    }

def validate_input(p):
    errors=[]; warnings_list=[]
    if p["machine_type"] not in ("L","M","H"): errors.append("Product type must be L, M or H.")
    for key,label,lo,hi in [("air_temp","Air temperature",250,350),("process_temp","Process temperature",250,360),("speed","Rotational speed",1,5000),("torque","Torque",0.1,150),("wear","Tool wear",0,500)]:
        try: v=float(p[key])
        except Exception: errors.append(f"{label} must be numeric."); continue
        if not np.isfinite(v): errors.append(f"{label} must be finite.")
        elif v<lo or v>hi: errors.append(f"{label} must be between {lo} and {hi}.")
    if p.get("process_temp",0) <= p.get("air_temp",0): warnings_list.append("Process temperature is not above air temperature; verify sensor readings.")
    if p.get("speed",0)>2886: warnings_list.append("Rotational speed is above the benchmark maximum (2886 rpm).")
    if p.get("wear",0)>253: warnings_list.append("Tool wear is above the benchmark maximum (253 min).")
    return errors,warnings_list

def expert_rules(p):
    typ=p["machine_type"]; wear=float(p["wear"]); torque=float(p["torque"]); speed=float(p["speed"])
    gap=float(p["process_temp"])-float(p["air_temp"]); power=torque*speed*0.10472
    wt=wear*torque
    trace=[f"R1 check: wear × torque = {wt:.1f}; threshold for Type {typ} = {TYPE_THRESHOLDS[typ]}."]
    if wt>TYPE_THRESHOLDS[typ]:
        return "OSF","R1",1.0,"Reduce cutting load and inspect/replace the tool.", "Immediately",trace+["R1 FIRED — Overstrain Failure. Lower-priority rules skipped."]
    trace.append(f"R2 check: temperature gap = {gap:.1f} K (< 8.6 K) and speed = {speed:.0f} rpm (< 1380 rpm).")
    if gap<8.6 and speed<1380:
        return "HDF","R2",1.0,"Inspect the cooling system and improve heat dissipation; verify spindle operating conditions.","Immediately",trace+["R2 FIRED — Heat Dissipation Failure."]
    trace.append(f"R3 check: mechanical power = {power:.1f} W; expected band 3500–9000 W.")
    if power<3500 or power>9000:
        return "PWF","R3",1.0,"Inspect power transmission and adjust torque/speed to the rated operating band.","Immediately",trace+["R3 FIRED — Power Failure."]
    trace.append(f"R4 check: tool wear = {wear:.0f} min; risk interval 200–240 min inclusive.")
    if 200<=wear<=240:
        return "TWF (risk)","R4",0.7,"Schedule preventive tool inspection and replacement.","Within 24 hours",trace+["R4 FIRED — Tool Wear Failure risk."]
    trace.append(f"R5 check: observed failure evidence = {bool(p.get('evidence',False))}.")
    if bool(p.get("evidence",False)):
        return "RNF","R5",0.5,"Log the incident and investigate for a non-deterministic failure cause.","Immediately",trace+["R5 FIRED — Random Failure."]
    return "No Failure","R6",0.0,"Continue routine monitoring and follow the preventive maintenance schedule.","Routine monitoring",trace+["R6 FIRED — No deterministic failure rule matched."]

def _features_frame(df):
    d=df.copy()
    d["Temp_Diff_K"]=d["Process temperature [K]"]-d["Air temperature [K]"]
    d["Wear_Torque_Product"]=d["Tool wear [min]"]*d["Torque [Nm]"]
    power=d["Torque [Nm]"]*d["Rotational speed [rpm]"]*0.10472
    d["Power_Deviation_W"]=power-6000.0
    d["Type_M"]=(d["Type"]=="M").astype(int)
    d["Type_L"]=(d["Type"]=="L").astype(int)
    return d[FEATURES]

def _ensure_xgboost_omp():
    """Ensure libomp runtime is discoverable for XGBoost on macOS if not installed via Homebrew."""
    import platform
    if platform.system() != "Darwin":
        return
    try:
        import xgboost  # Already works
        return
    except Exception:
        pass

    import sys, importlib.util, shutil, subprocess
    spec = importlib.util.find_spec("xgboost")
    if not spec or not spec.submodule_search_locations:
        return
    xgb_dir = Path(list(spec.submodule_search_locations)[0])
    xgb_lib_dir = xgb_dir / "lib"
    xgb_dylib = xgb_lib_dir / "libxgboost.dylib"
    if not xgb_dylib.exists():
        return

    # Check sklearn's bundled libomp.dylib or elsewhere in sys.prefix
    candidates = list(Path(sys.prefix).glob("**/libomp.dylib"))
    if not candidates:
        return
    source_omp = candidates[0]
    dest_omp = xgb_lib_dir / "libomp.dylib"
    if not dest_omp.exists():
        try:
            shutil.copy2(source_omp, dest_omp)
        except Exception:
            pass

    try:
        subprocess.run(
            ["install_name_tool", "-add_rpath", "@loader_path", str(xgb_dylib)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass

def load_system(csv_path):
    df=pd.read_csv(csv_path)
    X=_features_frame(df); y=df["Machine failure"].astype(int)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.20,random_state=42,stratify=y)
    positives=max(int(ytr.sum()),1); negatives=max(int((ytr==0).sum()),1)
    model_note=""
    _ensure_xgboost_omp()
    try:
        from xgboost import XGBClassifier
        ml=XGBClassifier(n_estimators=260,max_depth=4,learning_rate=.045,subsample=.9,colsample_bytree=.9,
                         objective="binary:logistic",eval_metric="logloss",scale_pos_weight=negatives/positives,
                         random_state=42,n_jobs=4)
        ml.fit(Xtr.to_numpy(),ytr)
        model_note="XGBoost trained locally from the bundled AI4I 2020 dataset."
    except Exception as e:
        from sklearn.ensemble import HistGradientBoostingClassifier
        ml=HistGradientBoostingClassifier(max_iter=180,learning_rate=.08,max_leaf_nodes=15,l2_regularization=1.0,random_state=42)
        sw=compute_sample_weight("balanced",ytr)
        ml.fit(Xtr,ytr,sample_weight=sw)
        model_note=f"XGBoost unavailable; HistGradientBoosting fallback trained instead ({type(e).__name__})."
    # DNN-like 3-hidden-layer MLP approximation; original Keras artifact was not supplied.
    dl=MLPClassifier(hidden_layer_sizes=(32,16,8),activation="relu",solver="adam",alpha=.001,
                     batch_size=32,learning_rate_init=.001,max_iter=90,early_stopping=True,
                     validation_fraction=.15,n_iter_no_change=12,random_state=42)
    sw=compute_sample_weight("balanced",ytr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dl.fit(Xtr,ytr,sample_weight=sw)
    auc=float(roc_auc_score(yte,ml.predict_proba(Xte.to_numpy())[:,1]))
    return {"ml":ml,"dl":dl,"columns":FEATURES,"holdout_auc":auc,"model_note":model_note,
            "dataset_rows":len(df),"failure_count":int(y.sum()),"df":df}

def assess_machine(system,payload):
    errors,warns=validate_input(payload)
    if errors: raise ValueError("Input validation failed: "+" ".join(errors))
    features=engineer(payload)
    row=pd.DataFrame([[features[c] for c in FEATURES]],columns=FEATURES)
    ml=float(system["ml"].predict_proba(row)[0,1])
    dl=float(system["dl"].predict_proba(row)[0,1])
    diagnosis,rule,expert,recommendation,window,trace=expert_rules(payload)
    risk=float(np.clip(.5*ml+.3*dl+.2*expert,0,1))
    level="CRITICAL" if risk>=.75 else "HIGH" if risk>=.5 else "MEDIUM" if risk>=.25 else "LOW"
    priority={"CRITICAL":"P1","HIGH":"P2","MEDIUM":"P3","LOW":"P4"}[level]
    # Local permutation impacts: absolute change in model probability after replacing one feature with its training median.
    impacts=[]
    base=ml
    medians=system["df"]
    med_payload={"Type_M":0,"Type_L":0}
    for col in FEATURES:
        changed=row.copy()
        if col in ("Type_M","Type_L"):
            changed[col]=0 if row.iloc[0][col] else row.iloc[0][col]
        else:
            vals=_features_frame(medians)[col]
            changed[col]=float(vals.median())
        new=float(system["ml"].predict_proba(changed)[0,1])
        impacts.append((col,base-new))
    impacts=sorted(impacts,key=lambda z:abs(z[1]),reverse=True)[:7]
    counterfactual="Try reducing tool wear and/or torque, then reassess; the smallest safe operating change depends on equipment limits. This is a scenario prompt, not an instruction to override rated limits."
    if diagnosis=="OSF": counterfactual="Inspect tool condition and reduce the wear–torque load combination within manufacturer limits, then reassess."
    elif diagnosis=="HDF": counterfactual="Check cooling and operating conditions; a larger temperature gap or corrected spindle speed may reduce the rule-triggered risk."
    elif diagnosis=="PWF": counterfactual="Verify power transmission and bring torque/speed into the machine's rated operating band."
    elif diagnosis.startswith("TWF"): counterfactual="Plan tool inspection/replacement before wear progresses beyond the preventive interval."
    return {"risk_score":risk,"risk_level":level,"ml_probability":ml,"dl_probability":dl,"expert_score":expert,
            "diagnosis":diagnosis,"rule_fired":rule,"recommendation":recommendation,"time_window":window,
            "priority":priority,"rule_trace":trace,"feature_impacts":[(k,float(v)) for k,v in impacts],
            "counterfactual":counterfactual,"warnings":warns,"features":features,
            "timestamp":pd.Timestamp.now().isoformat()}

def assess_fleet(system,df):
    required=["Type","Air temperature [K]","Process temperature [K]","Rotational speed [rpm]","Torque [Nm]","Tool wear [min]"]
    missing=[c for c in required if c not in df.columns]
    if missing: raise ValueError("CSV is missing required columns: "+", ".join(missing))
    records=[]
    for idx,r in df.iterrows():
        p={"machine_type":str(r["Type"]),"air_temp":float(r["Air temperature [K]"]),"process_temp":float(r["Process temperature [K]"]),
           "speed":float(r["Rotational speed [rpm]"]),"torque":float(r["Torque [Nm]"]),"wear":float(r["Tool wear [min]"]),
           "evidence":bool(r.get("Machine failure",0))}
        out=assess_machine(system,p)
        records.append({"UDI":r.get("UDI",idx+1),"Type":p["machine_type"],"risk_score":out["risk_score"],
                        "risk_level":out["risk_level"],"diagnosis":out["diagnosis"],"priority":out["priority"],
                        "ML_probability":out["ml_probability"],"DL_probability":out["dl_probability"]})
    return pd.DataFrame(records)
