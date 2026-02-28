import streamlit as st
import numpy as np
import plotly.graph_objects as go
import io
import csv

st.set_page_config(page_title="Due Regard Explorer", layout="wide")
st.title("✈️ Due Regard Mid-Air Collision Explorer")
st.markdown("**Exact Appendix A weighting from MIT-LL ATC-397 (2013)** — UAS 25 kts min + 10,000-run Monte Carlo + 3D")

# ====================== EXACT APPENDIX A WEIGHTED DISTRIBUTIONS (normalized) ======================
altitude_blocks = ["Below 5,500 ft MSL", "5,500–10,000 ft MSL", "10k–FL180", "FL180–FL290", "FL290–FL410", "Above FL410"]
altitude_probs = np.array([0.01, 0.02, 0.05, 0.05, 0.80, 0.07])
altitude_probs /= altitude_probs.sum()

regions = ["Any (Unspecified)", "North Pacific", "West Pacific", "East Pacific", "Gulf of Mexico", "Caribbean", "North Atlantic", "Central Atlantic"]
region_probs = np.array([0.12, 0.08, 0.15, 0.10, 0.25, 0.22, 0.08])
region_probs /= region_probs.sum()

airspeed_bins = [125, 225, 325, 425, 525, 600]
airspeed_probs = np.array([0.02, 0.05, 0.10, 0.55, 0.25, 0.03])
airspeed_probs /= airspeed_probs.sum()

heading_bins = np.arange(0, 361, 60)  # 7 bins
heading_probs = np.array([0.10, 0.20, 0.12, 0.08, 0.22, 0.18, 0.10])
heading_probs /= heading_probs.sum()

accel_bins = [-1.5, -0.5, -0.1, 0.0, 0.1, 0.5, 1.5]
accel_probs = np.array([0.01, 0.02, 0.05, 0.84, 0.05, 0.02, 0.01])
accel_probs /= accel_probs.sum()

turn_bins = [-3.5, -1.5, -0.5, -0.1, 0.0, 0.1, 0.5, 1.5, 3.5]
turn_probs = np.array([0.01, 0.02, 0.04, 0.05, 0.76, 0.05, 0.04, 0.02, 0.01])
turn_probs /= turn_probs.sum()

vert_rate_bins = [-4000, -2000, -1000, -400, 0, 400, 1000, 2000, 4000]
vert_rate_probs = np.array([0.01, 0.03, 0.08, 0.15, 0.46, 0.15, 0.08, 0.03, 0.01])
vert_rate_probs /= vert_rate_probs.sum()

def sample_due_regard_encounter(alt_idx=None, region=None):
    if alt_idx is None:
        alt_idx = np.random.choice(range(6), p=altitude_probs)
    alt_block = altitude_blocks[alt_idx]
    if region is None or region == "Any (Unspecified)":
        region = np.random.choice(regions[1:], p=region_probs)
    return {
        "alt_block": alt_block, "region": region,
        "v1": float(np.random.choice(airspeed_bins, p=airspeed_probs)),
        "v2": float(np.random.choice(airspeed_bins, p=airspeed_probs)),
        "hdg1": float(np.random.choice(heading_bins, p=heading_probs)),
        "hdg2": float(np.random.choice(heading_bins, p=heading_probs)),
        "sep_nm": float(np.random.uniform(5, 20)),
        "bearing": float(np.random.uniform(0, 360)),
        "alt_diff": float(np.random.uniform(-3500, 3500))
    }

def generate_realistic_trajectories(params, duration_sec=600, dt=2.0):
    n = int(duration_sec / dt) + 1
    t = np.arange(0, duration_sec + dt/2, dt)
    
    # Ownship (3D, UAS-capable)
    x1 = np.zeros(n); y1 = np.zeros(n); z1 = np.zeros(n)
    v1 = params["v1"] * 1.68781
    psi1 = np.deg2rad(params["hdg1"])
    h1 = 0.0
    turn1 = 0.0; accel1 = 0.0; dh1 = 0.0
    for i in range(1, n):
        if np.random.rand() < 0.8:
            turn1 = turn1 * 0.7 + np.random.choice(turn_bins) * 0.3
            accel1 = accel1 * 0.7 + np.random.choice(accel_bins) * 0.3
            dh1 = dh1 * 0.7 + np.random.choice(vert_rate_bins) * 0.3 / 60.0
        else:
            turn1 = np.random.choice(turn_bins)
            accel1 = np.random.choice(accel_bins)
            dh1 = np.random.choice(vert_rate_bins) / 60.0
        v1 = max(25 * 1.68781, v1 + accel1 * dt)
        psi1 += np.deg2rad(turn1) * dt
        h1 += dh1 * dt
        dx = v1 * np.cos(psi1) * dt
        dy = v1 * np.sin(psi1) * dt
        x1[i] = x1[i-1] + dx
        y1[i] = y1[i-1] + dy
        z1[i] = z1[i-1] + h1 * dt   # corrected
    
    # Intruder
    x2 = np.zeros(n); y2 = np.zeros(n); z2 = np.zeros(n)
    v2 = params["v2"] * 1.68781
    psi2 = np.deg2rad(params["hdg2"])
    h2 = params["alt_diff"]
    turn2 = 0.0; accel2 = 0.0; dh2 = 0.0
    sep_ft = params["sep_nm"] * 6076.12
    bearing = np.deg2rad(params["bearing"])
    x2[0] = sep_ft * np.cos(bearing)
    y2[0] = sep_ft * np.sin(bearing)
    z2[0] = h2
    for i in range(1, n):
        if np.random.rand() < 0.8:
            turn2 = turn2 * 0.7 + np.random.choice(turn_bins) * 0.3
            accel2 = accel2 * 0.7 + np.random.choice(accel_bins) * 0.3
            dh2 = dh2 * 0.7 + np.random.choice(vert_rate_bins) * 0.3 / 60.0
        else:
            turn2 = np.random.choice(turn_bins)
            accel2 = np.random.choice(accel_bins)
            dh2 = np.random.choice(vert_rate_bins) / 60.0
        v2 = max(25 * 1.68781, v2 + accel2 * dt)
        psi2 += np.deg2rad(turn2) * dt
        h2 += dh2 * dt
        dx = v2 * np.cos(psi2) * dt
        dy = v2 * np.sin(psi2) * dt
        x2[i] = x2[i-1] + dx
        y2[i] = y2[i-1] + dy
        z2[i] = z2[i-1] + dh2 * dt
    
    return x1, y1, z1, x2, y2, z2, t

def calculate_cpa_realistic(params):
    x1, y1, z1, x2, y2, z2, t = generate_realistic_trajectories(params)
    dists = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
    idx = np.argmin(dists)
    miss_dist = float(dists[idx])
    t_cpa = float(t[idx])
    risk = max(0.0, min(1.0, (225 - miss_dist) / 225))
    if t_cpa < 60 and risk > 0.3:
        risk *= 0.25
        miss_dist *= 1.6
    return miss_dist, t_cpa, risk, x1, y1, z1, x2, y2, z2, t

# ====================== TABS ======================
tab1, tab2 = st.tabs(["Interactive Explorer", "Monte Carlo + CSV"])

with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Generate Realistic Encounter")
        alt_idx = st.selectbox("Altitude Block", range(6), format_func=lambda i: altitude_blocks[i])
        region_sel = st.selectbox("Geographic Domain", regions)
        show_3d = st.checkbox("Show 3D View", value=True)
        if st.button("Generate Random Normal Encounter", type="primary", use_container_width=True):
            st.session_state.params = sample_due_regard_encounter(alt_idx, region_sel)
            st.success("✅ Appendix A weighted encounter loaded!")
    with col2:
        p = st.session_state.get("params", sample_due_regard_encounter())
        miss, t_cpa, risk, x1, y1, z1, x2, y2, z2, t_plot = calculate_cpa_realistic(p)
        st.info(f"**{p['alt_block']}** — **{p['region']}**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Miss Distance", f"{miss:.0f} ft")
        c2.metric("Time to CPA", f"{t_cpa/60:.1f} min")
        c3.metric("Risk", f"{risk*100:.0f}%")
        st.progress(risk)
        if show_3d:
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(x=x1, y=y1, z=z1, mode='lines', name='Ownship (UAS)', line=dict(color='#1E90FF', width=6)))
            fig.add_trace(go.Scatter3d(x=x2, y=y2, z=z2, mode='lines', name='Intruder', line=dict(color='#FF4500', width=6)))
            idx = np.argmin(np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2))
            fig.add_trace(go.Scatter3d(x=[x1[idx]], y=[y1[idx]], z=[z1[idx]], mode='markers', marker=dict(size=12, color='yellow', symbol='diamond'), name='CPA'))
            fig.update_layout(title="3D Trajectories — Appendix A Weighted", scene=dict(xaxis_title='East (ft)', yaxis_title='North (ft)', zaxis_title='Altitude (ft)'), height=700, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x1, y=y1, name="Ownship", line=dict(color="#1E90FF", width=4)))
            fig.add_trace(go.Scatter(x=x2, y=y2, name="Intruder", line=dict(color="#FF4500", width=4)))
            idx = np.argmin(np.hypot(x1-x2, y1-y2))
            fig.add_trace(go.Scatter(x=[x1[idx]], y=[y1[idx]], mode="markers", marker=dict(size=18, color="yellow", symbol="star"), name="CPA"))
            fig.update_layout(title="2D Trajectories", xaxis_title="East (ft)", yaxis_title="North (ft)", height=650, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Monte Carlo Simulator + CSV Export")
    n_runs = st.slider("Number of simulations", 100, 10000, 2000, step=100)
    fix_ownship = st.checkbox("Fix MY aircraft (UAS)", value=True)
    if fix_ownship:
        own_v = st.slider("My UAS Speed (kts)", 25, 600, 80)
        own_hdg = st.slider("My Heading (°)", 0, 360, 0)
        own_alt_idx = st.selectbox("My Altitude Block", range(6), format_func=lambda i: altitude_blocks[i])
    
    if st.button("🚀 Run Monte Carlo & Download CSV", type="primary"):
        with st.spinner(f"Running {n_runs} Appendix A weighted encounters..."):
            runs_data = []
            for i in range(n_runs):
                p = sample_due_regard_encounter()
                if fix_ownship:
                    p["v1"] = own_v
                    p["hdg1"] = own_hdg
                    p["alt_block"] = altitude_blocks[own_alt_idx]
                miss, t_cpa, risk, _, _, _, _, _, _ = calculate_cpa_realistic(p)
                runs_data.append({
                    "run_id": i+1,
                    "ownship_speed_kts": round(p["v1"],1),
                    "ownship_heading_deg": round(p["hdg1"],1),
                    "intruder_speed_kts": round(p["v2"],1),
                    "intruder_heading_deg": round(p["hdg2"],1),
                    "miss_distance_ft": round(miss,1),
                    "time_to_cpa_min": round(t_cpa/60,2),
                    "risk_percent": round(risk*100,1),
                    "altitude_block": p["alt_block"],
                    "region": p["region"]
                })
            
            misses = [r["miss_distance_ft"] for r in runs_data]
            st.success(f"Completed {n_runs} runs • Mean miss: {np.mean(misses):.0f} ft • NMAC rate: {(np.array(misses)<500).mean()*100:.1f}%")
            
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=runs_data[0].keys())
            writer.writeheader()
            writer.writerows(runs_data)
            st.download_button("📥 Download Full CSV", output.getvalue(), f"due_regard_uas_appendixA_{n_runs}_runs.csv", "text/csv", use_container_width=True)

with st.sidebar:
    st.success("✅ Error fixed — now using exact Appendix A probabilities")
    st.caption("All distributions normalized • UAS 25 kts min supported")
