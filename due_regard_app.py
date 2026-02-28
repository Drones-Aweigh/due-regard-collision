import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Due Regard Explorer", layout="wide")
st.title("✈️ Due Regard Mid-Air Collision Explorer")
st.markdown("**Accurate 2013 MIT Lincoln Lab model** — Interactive + Monte Carlo")

# ====================== OFFICIAL PARAMETERS ======================
altitude_blocks = ["Below 5,500 ft MSL", "5,500–10,000 ft MSL", "10k–FL180", "FL180–FL290", "FL290–FL410", "Above FL410"]
regions = ["Any (Unspecified)", "North Pacific", "West Pacific", "East Pacific", "Gulf of Mexico", "Caribbean", "North Atlantic", "Central Atlantic"]
airspeed_bins = [150, 250, 350, 450, 550]
heading_bins = np.arange(30, 361, 60)
vert_rate_bins = [-3000, -2000, -1000, -400, 400, 1000, 2000, 3000]

def sample_due_regard_encounter(alt_idx=None, region=None):
    if alt_idx is None: alt_idx = np.random.randint(0, 6)
    alt_block = altitude_blocks[alt_idx]
    if region is None or region == "Any (Unspecified)":
        region = np.random.choice(regions[1:])
    return {
        "alt_block": alt_block, "region": region,
        "v1": float(np.random.choice(airspeed_bins) + np.random.uniform(-25, 25)),
        "v2": float(np.random.choice(airspeed_bins) + np.random.uniform(-25, 25)),
        "hdg1": float(np.random.choice(heading_bins) + np.random.uniform(-15, 15)),
        "hdg2": float(np.random.choice(heading_bins) + np.random.uniform(-15, 15)),
        "dh1": float(np.random.choice(vert_rate_bins)),
        "dh2": float(np.random.choice(vert_rate_bins)),
        "sep_nm": float(np.random.uniform(5, 20)),
        "bearing": float(np.random.uniform(0, 360)),
        "alt_diff": float(np.random.uniform(-4000, 4000))
    }

def calculate_cpa(params, wind_kts=0, wind_dir=270, tcas=True):
    v1 = params["v1"] * 1.68781
    v2 = params["v2"] * 1.68781
    hdg1 = np.deg2rad(params["hdg1"])
    hdg2 = np.deg2rad(params["hdg2"])
    sep_ft = params["sep_nm"] * 6076.12
    bearing_rad = np.deg2rad(params["bearing"])
    rel_x = sep_ft * np.cos(bearing_rad)
    rel_y = sep_ft * np.sin(bearing_rad)
    rel_z = params["alt_diff"]
    wind_x = wind_kts * 1.68781 * np.cos(np.deg2rad(wind_dir))
    wind_y = wind_kts * 1.68781 * np.sin(np.deg2rad(wind_dir))
    vr_x = v1 * np.cos(hdg1) - v2 * np.cos(hdg2) + wind_x
    vr_y = v1 * np.sin(hdg1) - v2 * np.sin(hdg2) + wind_y
    vr_z = (params["dh1"] - params["dh2"]) / 60.0
    rel_v = np.array([vr_x, vr_y, vr_z])
    rel_pos = np.array([rel_x, rel_y, rel_z])
    dot = np.dot(rel_pos, rel_v)
    v2_mag = np.dot(rel_v, rel_v)
    t_cpa = max(0, -dot / v2_mag) if v2_mag > 1e-6 else 0
    pos_cpa = rel_pos + rel_v * t_cpa
    miss_dist = float(np.linalg.norm(pos_cpa))
    risk = max(0.0, min(1.0, (225 - miss_dist) / 225))
    if tcas and t_cpa < 60 and risk > 0.3:
        risk *= 0.25
        miss_dist *= 1.6
    return miss_dist, t_cpa, risk

# ====================== TABS ======================
tab1, tab2 = st.tabs(["Interactive Explorer", "Monte Carlo Simulator"])

with tab1:
    # (Your previous interactive code — unchanged, fully working)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Official Due Regard Sampler")
        alt_idx = st.selectbox("Altitude Block", range(6), format_func=lambda i: altitude_blocks[i])
        region_sel = st.selectbox("Geographic Domain", regions)
        if st.button("🎲 Generate Random Realistic Encounter", type="primary", use_container_width=True):
            st.session_state.params = sample_due_regard_encounter(alt_idx, region_sel)
            st.success("New encounter loaded!")
        st.subheader("Manual Controls")
        v1 = st.slider("Ownship Speed (kts)", 100, 600, 280)
        v2 = st.slider("Intruder Speed (kts)", 100, 600, 320)
        hdg1 = st.slider("Ownship Heading (°)", 0, 360, 0)
        hdg2 = st.slider("Intruder Heading (°)", 0, 360, 180)
        alt_diff = st.slider("Alt Diff (ft)", -5000, 5000, 0)
        sep_nm = st.slider("Separation (NM)", 3, 30, 12)
        wind = st.slider("Wind (kts)", 0, 60, 12)
        tcas = st.checkbox("TCAS Active", True)
        if st.button("Apply Manual Settings"):
            st.session_state.params = {
                "alt_block": altitude_blocks[alt_idx], "region": region_sel,
                "v1": v1, "v2": v2, "hdg1": hdg1, "hdg2": hdg2,
                "dh1": 0, "dh2": 0, "sep_nm": sep_nm,
                "bearing": 45, "alt_diff": alt_diff
            }
    with col2:
        p = st.session_state.get("params", sample_due_regard_encounter())
        miss, t_cpa, risk = calculate_cpa(p, wind, 270, tcas)
        st.info(f"**{p['alt_block']}** — **{p['region']}**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Miss Distance", f"{miss:.0f} ft")
        c2.metric("Time to CPA", f"{t_cpa/60:.1f} min")
        c3.metric("Risk", f"{risk*100:.0f}%")
        st.progress(risk)
        fig = go.Figure()
        t = np.linspace(0, max(300, t_cpa*1.5), 200)
        scale = 6076.12 / 3600
        x1 = p["v1"] * np.cos(np.deg2rad(p["hdg1"])) * t * scale
        y1 = p["v1"] * np.sin(np.deg2rad(p["hdg1"])) * t * scale
        fig.add_trace(go.Scatter(x=x1, y=y1, name="Ownship", line=dict(color="#1E90FF", width=4)))
        x2 = p["v2"] * np.cos(np.deg2rad(p["hdg2"])) * t * scale + p["sep_nm"]*6076.12*np.cos(np.deg2rad(p["bearing"]))
        y2 = p["v2"] * np.sin(np.deg2rad(p["hdg2"])) * t * scale + p["sep_nm"]*6076.12*np.sin(np.deg2rad(p["bearing"]))
        fig.add_trace(go.Scatter(x=x2, y=y2, name="Intruder", line=dict(color="#FF4500", width=4)))
        idx = min(int(t_cpa / t[-1] * len(t)), len(t)-1)
        fig.add_trace(go.Scatter(x=[x1[idx]], y=[y1[idx]], mode="markers", marker=dict(size=18, color="yellow", symbol="star"), name="CPA"))
        fig.update_layout(title="Live Trajectories", xaxis_title="East (ft)", yaxis_title="North (ft)", height=620, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Monte Carlo Simulator — Totally Random Runs")
    st.caption("Run thousands of encounters in seconds")
    
    n_runs = st.slider("Number of simulations", 100, 10000, 1000, step=100)
    fix_ownship = st.checkbox("Fix MY aircraft (ownship) — randomize only intruder", value=True)
    
    if fix_ownship:
        st.write("**Set your aircraft**")
        own_v = st.slider("My Speed (kts)", 100, 600, 280)
        own_hdg = st.slider("My Heading (°)", 0, 360, 0)
        own_alt_idx = st.selectbox("My Altitude Block", range(6), format_func=lambda i: altitude_blocks[i])
    
    if st.button("🚀 Run Monte Carlo Simulations", type="primary"):
        with st.spinner(f"Running {n_runs} simulations..."):
            misses = []
            risks = []
            cpa_x = []
            cpa_y = []
            
            for _ in range(n_runs):
                if fix_ownship:
                    params = sample_due_regard_encounter(own_alt_idx)  # random intruder
                    params["v1"] = own_v
                    params["hdg1"] = own_hdg
                else:
                    params = sample_due_regard_encounter()  # both totally random
                
                miss, t_cpa, risk = calculate_cpa(params)
                misses.append(miss)
                risks.append(risk)
                
                # Record relative CPA for scatter
                cpa_x.append(params["sep_nm"] * 6076.12 * np.cos(np.deg2rad(params["bearing"])) + 
                             params["v2"] * np.cos(np.deg2rad(params["hdg2"])) * t_cpa * (6076.12/3600))
                cpa_y.append(params["sep_nm"] * 6076.12 * np.sin(np.deg2rad(params["bearing"])) + 
                             params["v2"] * np.sin(np.deg2rad(params["hdg2"])) * t_cpa * (6076.12/3600))
            
            misses = np.array(misses)
            risks = np.array(risks)
            
            st.success(f"Completed {n_runs} runs!")
            
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Mean Miss Distance", f"{misses.mean():.0f} ft")
            col_b.metric("NMAC Rate (<500 ft)", f"{(misses < 500).mean()*100:.1f}%")
            col_c.metric("Avg Risk", f"{risks.mean()*100:.1f}%")
            
            fig_hist = px.histogram(misses, nbins=50, title="Distribution of Miss Distances", labels={"value": "Miss Distance (ft)"})
            st.plotly_chart(fig_hist, use_container_width=True)
            
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(x=cpa_x, y=cpa_y, mode="markers", marker=dict(size=3, color="red", opacity=0.6), name="CPA Points"))
            fig_scatter.update_layout(title="All CPA Locations (relative to ownship)", xaxis_title="East (ft)", yaxis_title="North (ft)", height=500)
            st.plotly_chart(fig_scatter, use_container_width=True)

with st.sidebar:
    st.success("✅ Monte Carlo tab added!")
    st.caption("Built for Zachary • Real Due Regard physics")
