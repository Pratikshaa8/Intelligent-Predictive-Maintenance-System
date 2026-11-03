import io, json, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from src.pipeline import load_system, assess_machine, assess_fleet, validate_input
from src.report import build_pdf_report

st.set_page_config(page_title="PM-CDSS | Predictive Maintenance", page_icon="⚙️", layout="wide", initial_sidebar_state="expanded")

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:linear-gradient(135deg,#0b1220 0%,#101b2d 52%,#0b1424 100%); color:#e7eef8;}
header[data-testid="stHeader"] {background:transparent;}
section[data-testid="stSidebar"] {background:#0c1728;border-right:1px solid #22334a;}
.block-container {padding-top:1.5rem; max-width:1450px;}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif; color:#f3f7fc;}
.hero {padding:26px 30px;border:1px solid #29415d;border-radius:20px;background:linear-gradient(110deg,rgba(22,43,67,.98),rgba(15,29,47,.92));margin-bottom:20px;}
.hero-kicker {color:#69d6c5;font-size:12px;font-weight:700;letter-spacing:2px;text-transform:uppercase;}
.hero h1 {font-size:34px;margin:7px 0 5px;}
.hero p {color:#a9bad0;margin:0;font-size:15px;}
.metric {background:linear-gradient(145deg,#17283e,#122136);border:1px solid #29415d;border-radius:16px;padding:17px 18px;min-height:112px;}
.metric .label {color:#93a9c1;font-size:12px;text-transform:uppercase;letter-spacing:1px;font-weight:700;}
.metric .value {font-family:'Space Grotesk';font-size:28px;font-weight:700;color:#f5f8fc;margin-top:8px;}
.metric .sub {color:#8da4bd;font-size:12px;margin-top:3px;}
.panel {background:#111f32;border:1px solid #263c55;border-radius:16px;padding:18px 20px;margin-bottom:12px;}
.small-muted {color:#91a6bd;font-size:13px;}
div[data-testid="stDataFrame"] {border:1px solid #263c55;border-radius:12px;overflow:hidden;}
.stButton>button, .stDownloadButton>button {border-radius:10px;border:1px solid #2b6c77;background:#1d6870;color:white;font-weight:700;padding:.55rem 1rem;}
.stButton>button:hover, .stDownloadButton>button:hover {background:#27838a;color:white;border-color:#42b9b3;}
div[data-baseweb="select"]>div, div[data-baseweb="input"]>div {background:#13243a;border-color:#2b4058;}
hr {border-color:#263b53;}
</style>
""", unsafe_allow_html=True)

def metric(label, value, sub=""):
    st.markdown(f'<div class="metric"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)

def risk_color(level):
    return {"CRITICAL":"#f06464","HIGH":"#f0a34a","MEDIUM":"#e8ca62","LOW":"#48c7a5"}.get(level,"#91a6bd")

@st.cache_resource(show_spinner="Preparing the predictive models from the AI4I dataset…")
def get_system():
    return load_system(ROOT/"data"/"ai4i2020.csv")

@st.cache_data
def get_dataset():
    return pd.read_csv(ROOT/"data"/"ai4i2020.csv")

st.markdown("""
<div class="hero">
  <div class="hero-kicker">Computational Intelligence · Decision Support</div>
  <h1>Predictive Maintenance <span style="color:#62d5c6">Command Center</span></h1>
  <p>Turn machine sensor readings into explainable, prioritised maintenance decisions.</p>
</div>
""", unsafe_allow_html=True)

try:
    system = get_system()
    data = get_dataset()
except Exception as e:
    st.error(f"Could not initialise the project dataset or models: {e}")
    st.stop()

with st.sidebar:
    st.markdown("## ⚙️ PM-CDSS")
    st.caption("Intelligent machine health monitoring")
    st.markdown("---")
    st.markdown("### Machine sensor input")
    preset = st.selectbox("Load an example", ["Custom input","Healthy machine","Overstrain example","Heat dissipation example","Tool wear risk example"])
    presets = {
        "Healthy machine": dict(machine_type="M",air_temp=298.1,process_temp=308.6,speed=1551,torque=42.8,wear=15,evidence=False),
        "Overstrain example": dict(machine_type="L",air_temp=298.9,process_temp=309.0,speed=1410,torque=65.7,wear=191,evidence=True),
        "Heat dissipation example": dict(machine_type="M",air_temp=300.8,process_temp=309.0,speed=1342,torque=62.4,wear=113,evidence=True),
        "Tool wear risk example": dict(machine_type="M",air_temp=298.1,process_temp=308.0,speed=1600,torque=45,wear=215,evidence=False),
    }
    p = presets.get(preset, presets["Healthy machine"])
    machine_type = st.selectbox("Product type", ["L","M","H"], index=["L","M","H"].index(p["machine_type"]))
    air_temp = st.number_input("Air temperature (K)", 250.0, 350.0, float(p["air_temp"]), .1)
    process_temp = st.number_input("Process temperature (K)", 250.0, 360.0, float(p["process_temp"]), .1)
    speed = st.number_input("Rotational speed (rpm)", 0, 5000, int(p["speed"]), 1)
    torque = st.number_input("Torque (Nm)", 0.0, 150.0, float(p["torque"]), .1)
    wear = st.number_input("Tool wear (min)", 0, 500, int(p["wear"]), 1)
    evidence = st.checkbox("Failure evidence already observed (for R5)", value=bool(p["evidence"]))
    run_one = st.button("▶  Assess machine", use_container_width=True)
    st.markdown("---")
    st.caption("Dataset: AI4I 2020 · 10,000 records")
    st.caption("Human approval is required before maintenance action.")

tabs = st.tabs(["◈ Overview","⌁ Assessment","✳ Explainability","▦ Fleet analytics","▤ Reports & system"])

# Overview
with tabs[0]:
    failures = int(data["Machine failure"].sum())
    c1,c2,c3,c4 = st.columns(4)
    with c1: metric("Machines in dataset", f"{len(data):,}", "AI4I 2020 benchmark")
    with c2: metric("Recorded failures", f"{failures:,}", f"{failures/len(data):.1%} failure prevalence")
    with c3: metric("No-failure records", f"{len(data)-failures:,}", "Class imbalance handled in training")
    with c4: metric("Primary model", "XGBoost", "Selected by ROC-AUC in project report")
    st.markdown("### System at a glance")
    left,right = st.columns([1.2,1])
    with left:
        st.markdown('<div class="panel"><h3>Decision pipeline</h3><p class="small-muted">Sensor data → validation → feature engineering → expert rules → ML + DL → risk fusion → explanation → maintenance action</p></div>',unsafe_allow_html=True)
        stages = pd.DataFrame({"Stage":["Input validation","Feature engineering","Expert rules","XGBoost + DNN","Decision fusion","Explainability"],"Progress":[1,1,1,1,1,1]})
        fig=px.bar(stages,x="Progress",y="Stage",orientation="h",range_x=[0,1],color="Stage",title="Integrated pipeline modules")
        fig.update_layout(template="plotly_dark",showlegend=False,height=300,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",xaxis_visible=False)
        st.plotly_chart(fig,use_container_width=True)
    with right:
        counts=data["Type"].value_counts().reindex(["L","M","H"]).fillna(0).reset_index()
        counts.columns=["Product type","Machines"]
        fig=px.pie(counts,names="Product type",values="Machines",hole=.62,title="Product type distribution")
        fig.update_layout(template="plotly_dark",height=300,paper_bgcolor="rgba(0,0,0,0)",legend_title_text="")
        st.plotly_chart(fig,use_container_width=True)
    st.markdown("### Reported project benchmark")
    bm=pd.DataFrame([
        ["XGBoost",.989,.849,.824,.836,.988],
        ["Random Forest",.991,.962,.750,.843,.968],
        ["Decision Tree",.975,.587,.897,.709,.961],
        ["SVM",.926,.294,.853,.438,.968],
        ["Logistic Regression",.840,.154,.824,.259,.911],
        ["Deep Neural Network",.892,.234,.961,.377,.977],
    ],columns=["Model","Accuracy","Precision","Recall","F1-score","ROC-AUC"])
    st.dataframe(bm.style.format({c:"{:.3f}" for c in bm.columns[1:]}),use_container_width=True,hide_index=True)
    st.caption("These are values transcribed from Task 10, not metrics recalculated by this app.")

# Assess button result kept in session
if "latest" not in st.session_state: st.session_state.latest=None
if run_one:
    payload=dict(machine_type=machine_type,air_temp=air_temp,process_temp=process_temp,speed=speed,torque=torque,wear=wear,evidence=evidence)
    try:
        st.session_state.latest=assess_machine(system,payload)
        st.session_state.latest_input=payload
    except Exception as e:
        st.session_state.latest=None
        st.error(str(e))

with tabs[1]:
    st.markdown("### Individual machine assessment")
    st.markdown('<p class="small-muted">Set sensor values in the left panel and select <b>Assess machine</b>. Example presets are available.</p>',unsafe_allow_html=True)
    result=st.session_state.latest
    if result:
        color=risk_color(result["risk_level"])
        a,b,c,d=st.columns(4)
        with a: metric("Fused risk",f'{result["risk_score"]:.1%}',result["risk_level"])
        with b: metric("ML probability",f'{result["ml_probability"]:.1%}',"XGBoost primary model")
        with c: metric("DL probability",f'{result["dl_probability"]:.1%}',"MLP neural-network approximation")
        with d: metric("Expert diagnosis",result["diagnosis"],result["rule_fired"])
        st.markdown(f'<div class="panel" style="border-left:5px solid {color}"><h2 style="color:{color};margin:0">{result["risk_level"]} RISK · {result["diagnosis"]}</h2><p>{result["recommendation"]}</p><p class="small-muted">Priority: {result["priority"]} · Suggested window: {result["time_window"]}</p></div>',unsafe_allow_html=True)
        st.markdown("#### Risk composition")
        comp=pd.DataFrame({"Component":["XGBoost · 50%","DNN · 30%","Expert rules · 20%"],"Score":[result["ml_probability"],result["dl_probability"],result["expert_score"]]})
        fig=px.bar(comp,x="Score",y="Component",orientation="h",range_x=[0,1],text=comp["Score"].map(lambda x:f"{x:.0%}"))
        fig.update_layout(template="plotly_dark",height=220,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",xaxis_tickformat=".0%")
        st.plotly_chart(fig,use_container_width=True)
        with st.expander("Expert-system rule trace",expanded=True):
            for line in result["rule_trace"]: st.write("• "+line)
        st.info("Decision support only. A qualified maintenance engineer must verify the machine and approve any action.")
    else:
        st.markdown('<div class="panel"><h3>Ready for assessment</h3><p class="small-muted">Your sensor inputs will be processed through the complete PM-CDSS pipeline. Choose a preset or enter your own values in the sidebar.</p></div>',unsafe_allow_html=True)

with tabs[2]:
    st.markdown("### Explainability & trust")
    result=st.session_state.latest
    if result:
        st.markdown("#### Local feature impact")
        imp=pd.DataFrame(result["feature_impacts"],columns=["Feature","Impact"])
        fig=px.bar(imp.sort_values("Impact"),x="Impact",y="Feature",orientation="h",color="Impact",color_continuous_scale="Teal")
        fig.update_layout(template="plotly_dark",height=330,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",coloraxis_showscale=False)
        st.plotly_chart(fig,use_container_width=True)
        st.markdown("#### What-if counterfactual")
        st.write(result["counterfactual"])
        st.markdown("#### Why did the expert system decide this?")
        st.write(f"**{result['diagnosis']}** was produced by **{result['rule_fired']}**. Rule evaluation follows the priority order R1 → R2 → R3 → R4 → R5 → R6.")
    else:
        st.info("Run an assessment to see machine-specific explanations.")
    st.markdown("#### Global explainability findings from Task 10")
    xai=pd.DataFrame([["Wear × Torque",2.004,30.5],["Power deviation",1.349,20.5],["Rotational speed",1.330,20.3],["Temperature difference",1.133,17.3],["Torque",.600,9.1]],columns=["Feature","Mean |SHAP|","Share (%)"])
    st.dataframe(xai,use_container_width=True,hide_index=True)
    q1,q2,q3=st.columns(3)
    with q1: metric("Global trust score","83.6/100","Reported in Task 8")
    with q2: metric("SHAP / permutation rank corr.","0.82","Reported in Task 10")
    with q3: metric("Noise-test decision changes","1.1%","Reported in Task 10")
    st.caption("Local impacts in this reconstructed app are perturbation-based approximations, not exact SHAP values from the original saved model artefacts.")

with tabs[3]:
    st.markdown("### Fleet analytics")
    uploaded=st.file_uploader("Upload a CSV with AI4I sensor columns",type=["csv"],key="fleet_upload")
    fleet_data= data.head(500).copy() if uploaded is None else pd.read_csv(uploaded)
    st.caption("Showing a 500-row sample of the bundled benchmark by default. Upload your own CSV for fleet scoring.")
    if st.button("Score fleet",key="score_fleet"):
        try:
            scored=assess_fleet(system,fleet_data)
            st.session_state.fleet=scored
        except Exception as e: st.error(str(e))
    if "fleet" in st.session_state:
        scored=st.session_state.fleet
        a,b,c,d=st.columns(4)
        with a: metric("Machines scored",f'{len(scored):,}')
        with b: metric("Critical",int((scored["risk_level"]=="CRITICAL").sum()))
        with c: metric("High",int((scored["risk_level"]=="HIGH").sum()))
        with d: metric("Medium / Low",int(scored["risk_level"].isin(["MEDIUM","LOW"]).sum()))
        col1,col2=st.columns(2)
        with col1:
            fig=px.histogram(scored,x="risk_level",category_orders={"risk_level":["CRITICAL","HIGH","MEDIUM","LOW"]},title="Risk level distribution",color="risk_level")
            fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig,use_container_width=True)
        with col2:
            fig=px.histogram(scored,x="risk_score",nbins=25,title="Fused risk score distribution")
            fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",xaxis_tickformat=".0%")
            st.plotly_chart(fig,use_container_width=True)
        st.markdown("#### Highest-priority machines")
        st.dataframe(scored.sort_values("risk_score",ascending=False).head(15),use_container_width=True,hide_index=True)
        st.download_button("⬇ Download scored fleet CSV",scored.to_csv(index=False).encode(),file_name="pm_cdss_fleet_assessment.csv",mime="text/csv")
    else:
        st.info("Select **Score fleet** to run assessments for the current CSV.")

with tabs[4]:
    st.markdown("### Maintenance report")
    result=st.session_state.latest
    if result:
        pdf=build_pdf_report(result,st.session_state.latest_input)
        st.download_button("⬇ Download machine maintenance PDF",pdf,file_name="PM_CDSS_Maintenance_Report.pdf",mime="application/pdf")
        st.download_button("⬇ Download prediction JSON",json.dumps(result,indent=2,default=str).encode(),file_name="PM_CDSS_Prediction.json",mime="application/json")
    else:
        st.info("Run an individual machine assessment to generate a PDF and JSON report.")
    st.markdown("### System configuration")
    st.markdown("""
    - **Primary ML:** XGBoost classifier (class imbalance handled with `scale_pos_weight`).
    - **DL component:** 3-hidden-layer neural network approximation (32, 16, 8 units) trained from the supplied CSV.
    - **Expert system:** priority-ordered rules R1–R6 with forward-chaining trace.
    - **Fusion:** 0.5 × ML + 0.3 × DL + 0.2 × expert score; weights are renormalised if a model is unavailable.
    - **Risk bands:** Critical ≥ 0.75; High ≥ 0.50; Medium ≥ 0.25; Low < 0.25.
    - **Validation:** hard checks block invalid values; soft range checks display warnings.
    """)
    st.warning("The original trained model files (best_model.pkl, dnn_model.h5 and dnn_scaler.pkl) were not among the uploads. This app trains reproducible replacement models from ai4i2020.csv at first launch. The DNN is an sklearn MLP approximation, not the original Keras network. Reported Task 8–10 metrics are retained as reference values and are not claimed as this rebuild's measured performance.")
    st.markdown("### Responsible AI notes")
    st.write("The benchmark is synthetic and failure labels are imbalanced (about 3.4%). Product-type false-alarm differences and data circularity need monitoring on real plant data. Predictions are decision support, not autonomous machine-control instructions.")
    st.markdown("### Model training diagnostics")
    if "fallback" in system.get("model_note", "").lower():
        st.warning(system["model_note"])
    else:
        st.success(system["model_note"])
    st.caption("PM-CDSS · Local prototype · Built from the uploaded Computational Intelligence Laboratory reports and AI4I 2020 dataset.")
