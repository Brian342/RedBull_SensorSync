import pandas as pd
import numpy as np
import streamlit as st
from streamlit_folium import st_folium
import folium
from matplotlib import pyplot as plt

# modification of streamlit page
st.set_page_config(page_title="WareHouse Readings", layout="wide", initial_sidebar_state="expanded")

CUSTOM_CSS = r"""
    <style>
:root[data-theme="light"] {
  --bg: #0f172a;
  --card: rgba(255,255,255,0.06);
  --text: #0b1220;
  --accent1: linear-gradient(90deg,#7c3aed, #06b6d4);
}
:root[data-theme="dark"] {
  --bg: #070812;
  --card: rgba(255,255,255,0.04);
  --text: #dbeafe;
  --accent1: linear-gradient(90deg,#06b6d4, #7c3aed);
}

/* Apply glass card effect to streamlit elements */
main .block-container {
  background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.00));
  padding: 1.6rem 2rem;
}
section[data-testid="stSidebar"] .css-1d391kg {
  background: transparent;
}
.css-1d391kg, .css-1d391kg .stButton button {
  border-radius: 14px;
}

/* Title style */
.header {
  display:flex; align-items:center; gap:12px;
}
.logo-circle {
  width:56px;height:56px;border-radius:12px;
  background: var(--accent1);
  display:flex;align-items:center;justify-content:center;color:white;font-weight:700;
  box-shadow: 0 8px 30px rgba(99,102,241,0.15);
}

/* Card style used in columns */
.card {
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  border: 1px solid rgba(255,255,255,0.04);
  padding: 16px;
  border-radius: 12px;
}

/* small pills */
.pill {
  display:inline-block;padding:6px 10px;border-radius:999px;font-size:12px;background:rgba(255,255,255,0.03);
}

/* bot bubble */
.bot {
  background: linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  padding: 10px 12px;border-radius:12px;margin:6px 0;
}
footer {
    color: white !important;
    background-color: #0e1117;
    padding: 8px;
    font-size: 16px;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
# with st.sidebar:
    # st.markdown(
    #     "<div style='display:flex;align-items:center;gap:10px'><div class='logo-circle'>WA</div><div><h3 style='margin:0'>WakeApp</h3><div style='font-size:12px;color:gray'>Opencv · YOLO · Torch</div></div></div>",
    #     unsafe_allow_html=True)
    # st.markdown("-------")
    # st.caption("Built w/ OpenCv + YOLO + Torch. prototype")


tabs = st.tabs(["Overview", "Exploratory Analysis", "Dashboard"])


with tabs[0]:
    st.markdown(
        """
            <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:20px'>
            <div>
                <h1 style='margin:0;background:linear-gradient(90deg,#06b6d4,#7c3aed);-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
                    WareHouse Sensor Readings
                </h1>
                <div style='color:gray;font-size:15px;'>AI-powered, monitoring and Analysis — Derives Hided information from raw sensor Data obtained from sensors in a Company</div>
            </div>
            </div>
            """,
        unsafe_allow_html=True
    )

    st.markdown("<div style='margin-bottom:10px'></div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([6.75, 5.55, 6.69])

    with c1:
        st.markdown(
            "<div class='card'><h4 style='margin:0'>Why need this system?</h4>"
            "<div style='color:gray;margin-top:6px'>Implementing an industrial sensor monitoring system transforms a company’s " \
            "operations by shifting from costly, reactive 'run-to-failure' models to Predictive Maintenance (PdM), " \
            "which utilizes real-time data like vibration and pressure to catch mechanical issues weeks before they escalate. " \
            "This proactive approach eliminates expensive unplanned downtime by allowing repairs to be scheduled during off-peak hours, " \
            "while simultaneously enhancing workplace safety through automated logic controls that shut down equipment in dangerous 'red-zone' conditions. " \
            "Furthermore, by keeping machinery running within optimal parameters, companies significantly extend asset lifespans and delay massive capital expenditures, " \
            "all while utilizing 'black box' Root Cause Analysis to prevent repeat failures and optimizing resource efficiency by targeting labor and energy usage only where it is truly needed.</div></div>",
            unsafe_allow_html=True
        )

        key_stats = [
            """
            * Total Asset Loss: Historically, without sensor monitoring, 5% to 10% of heavy industrial machinery experiences a "fatal" catastrophic breakdown annually that requires total replacement..
            * The Cost of a "Crash": Unplanned system crashes result in an average of 12–24 hours of downtime; for high-output companies, this translates to a loss of 20,000 - 200,000 dollars per hour depending on the industry.
            * The "Age" Multiplier: Machine risk increases exponentially after 15 years of service; sensor data shows that machines in this age bracket are 3.5x more likely to experience voltage spikes that fry internal circuits.
            """
        ]

    with c2:
        st.markdown(
            f"""
            <div class='card' style='text-align:left'>
            <h5 style='margin:0'>Key Stats and risk</h5>
            <div style='color:gray;margin-top:6px'>
            {' '.join(key_stats)}
            </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        crash_type = [
            """
            * Kinetic & Mechanical Crashes: These involve physical failures in moving parts like conveyors and AGVs, often caused by jams or snaps that immediately halt the physical movement of goods.
            * Electrical & Control System Crashes: These are "invisible" failures, such as brownouts or communication blackouts, where power fluctuations or signal loss paralyze the facility's "brain" and logic.
            * Hydraulic & Pneumatic Crashes: These occur in lifting and sorting equipment when pressure loss or seal blowouts cause sudden, gravity-driven failures of heavy loads.
            * Rack & Structural Collapses: The most catastrophic warehouse event, these are domino-effect failures usually triggered by unreported structural "injuries" that lead to total racking failure.
            """
        ]
    with c3:
        st.markdown(
            f"""
            <div class='card' style='text-align:left'>
            <h8 style='margin:0'>common crash types</h8>
            <div style='color:gray;margin-top:6px'>
            {' '.join(crash_type)}
            </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown("<br>", unsafe_allow_html=True)

with tabs[1]:
    st.header(
        "Exploratory Analysis")

    col1, col2 = st.columns([2, 1.9])
    with col1:
        st.markdown(
            """
            <div style="
                background-color: #ffffff;
                padding: 1.2rem;
                border-radius: 15px;
                box-shadow: 0px 4px 10px rgba(0,0,0,0.1);

            <h3 style="margin-top:0;color:white;">Route Map</h3>
            <p style="color:gray">Your map and directions will appear here.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
             <div style="
             background-color: #ffffff;
             padding: 2.5rem;
             border-radius: 15px;
             text-align: center;
             box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
             ">
             <h3 style="color:black">Status</h3>
             <div id="state-box" style="
             font-size: 2rem;
             font-weight: bold;
             margin-top: 20px;
             padding: 20px;
             border-radius: 12px;
             background-color: #e8f0fe;
             color: #1a73e8;
             ">
             Awake/Asleep
             </div>
             </div>
             """,
            unsafe_allow_html=True,
        )

with tabs[2]:
    st.header("Dashboard")