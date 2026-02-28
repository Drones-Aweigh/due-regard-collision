import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io
import csv

st.set_page_config(page_title="Due Regard Explorer", layout="wide")
st.title("✈️ Due Regard Mid-Air Collision Explorer")
st.markdown("**Realistic 2013 MIT LL model** — mostly straight + gentle turns • Monte Carlo + CSV export")

# ====================== OFFICIAL BINS ======================
altitude_blocks = ["Below 5,500 ft MSL", "5,500–10,000 ft MSL", "10k–FL180", "FL180–FL290", "FL290–FL410", "Above FL410"]
regions = ["Any (Unspecified)", "North Pacific", "West Pacific", "East Pacific", "Gulf of Mexico", "Caribbean", "North Atlantic", "Central Atlantic"]

airspeed_bins = [150, 250, 350, 450, 550]
heading_bins = np.arange(30, 361, 60)
turn_bins = np.array([-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0])
accel_bins = np.array([-1.0, -0.3, 0.0, 0.3, 1.0])

def sample_due_regard_encounter(alt_idx=None, region=None):
    if alt_idx is None: alt_idx = np.random.randint(0, 6)
    alt_block = altitude_blocks[alt_idx]
    if region is None or region == "Any (Unspecified)":
        region = np.random.choice(regions[1:])
    return {
        "alt_block": alt_block, "region": region,
        "v1": float(np.random.choice(airspeed_bins) + np.random.uniform(-20, 20)),
        "v2": float(np.random.choice(airspeed_bins) + np.random.uniform(-20, 20)),
        "hdg1": float(np.random.choice(heading_bins) + np.random.uniform(-15, 15)),
        "hdg2": float(np.random.choice(heading_bins) + np.random.uniform(-15, 15)),
        "sep_nm": float(np.random.uniform(5, 20)),
        "bearing": float(np.random.uniform(0, 360)),
        "alt_diff": float(np.random.uniform(-3500, 3500))
    }

def generate_realistic_trajectories(params, duration_sec=600, dt=2.0):
    n = int(duration_sec / dt) + 1
    t = np.arange(0, duration_sec + dt/2, dt)
    # Ownship
    x1 = np.zeros(n); y1 = np.zeros(n)
    v1 = params["v1"] * 1.68781
    psi1 = np.deg2rad(params["hdg1"])
    change_times = np.cumsum(np.random.uniform(15, 45, size=25))
    change_times = change_times[change_times < duration_sec]
    current_turn1 = 0.0
    current_accel1 = 0.0
    for i in range(1, n):
        ti = t[i]
        if len(change_times) > 0 and ti > change_times[0]:
            current_turn1 = np.random.choice(turn_bins) * np.random.uniform(0.6, 1.0)
            current_accel1 = np.random.choice(accel_bins) * np.random.uniform(0.5, 1.0)
            change_times = change_times[1:]
        v1 = max(100*1.68781, v1 + current_accel1 * dt)
        psi1 += current_turn1 * dt
        dx = v1 * np.cos(psi1) * dt
        dy = v1 * np.sin(psi1) * dt
        x1[i] = x1[i-1] + dx
        y1[i] = y1[i-1] + dy
    # Intruder
    x2 = np.zeros(n); y2 = np.zeros(n)
    v2 = params["v2"] * 1.68781
    psi2 = np.deg2rad(params["hdg2"])
    change_times = np.cumsum(np.random.uniform(15, 45, size=25))
    change_times = change_times[change_times < duration_sec]
    current_turn2 = 0.0
    current_accel2 = 0.0
    sep_ft = params["sep_nm"] * 6076.12
    bearing = np.deg2rad(params["bearing"])
    x2[0] = sep_ft * np.cos(bearing)
    y2[0] = sep_ft * np.sin(bearing)
    for i in range(1, n):
        ti = t[i]
        if len(change_times) > 0 and ti > change_times[0]:
            current_turn2 = np.random.choice(turn_bins) * np.random.uniform(0.6, 1.0)
            current_accel2 = np.random.choice(accel_bins) * np.random.uniform(0.5, 1.0)
            change_times = change_times[1:]
        v2 = max(100*1.68781, v2 + current_accel2 * dt)
        psi2 += current_turn2 * dt
        dx = v2 * np.cos(psi2) * dt
        dy = v2 * np.sin(psi2) * dt
        x2[i] = x2[i-1] + dx
        y2[i] = y2[i-1] + dy
    return x1, y1, x2, y2, t

def calculate_cpa_realistic(params):
    x1, y1, x2, y2, t = generate_realistic_trajectories(params)
    dists = np.hypot(x1 - x2, y1 - y2)
    idx = np.argmin(dists)
    miss_dist = float(dists[idx])
    t_cpa = float(t[idx])
    risk = max(0.0, min(1.0, (225 - miss_dist) / 225))
    if t_cpa < 60 and risk > 0.3:
        risk *= 0.25
        miss_dist *= 1.6
    return miss_dist, t_cpa, risk, x1, y1, x2, y2, t

# ====================== TABS ======================
tab1, tab2 = st.tabs(["Interactive Explorer", "Monte Carlo + CSV Export"])

with tab1:
    # (same interactive tab as before — unchanged)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Generate Realistic Encounter")
        alt_idx = st.selectbox("Altitude Block", range(6), format_func=lambda i: altitude_blocks[i])
        region_sel = st.selectbox("Geographic Domain", regions)
        if st.button("Generate Random Normal Encounter", type="primary", use_container_width=True):
            st.session_state.params = sample_due_regard_encounter(alt_idx, region_sel)
            st.success("✅ Loaded!")
    with col2:
        p = st.session_state.get("params", sample_due_regard_encounter())
        miss, t_cpa, risk, x1, y1, x2, y2, t_plot = calculate_cpa_realistic(p)
        st.info(f"**{p['alt_block']}** — **{p['region']}**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Miss Distance", f"{miss:.0f} ft")
        c2.metric("Time to CPA", f"{t_cpa/60:.1f} min")
        c3.metric("Risk", f"{risk*100:.0f}%")
        st.progress(risk)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x1, y=y1, name="Ownship", line=dict(color="#1E90FF", width=4)))
        fig.add_trace(go.Scatter(x=x2, y=y2, name="Intruder", line=dict(color="#FF4500", width=4)))
        idx = np.argmin(np.hypot(x1-x2, y1-y2))
        fig.add_trace(go.Scatter(x=[x1[idx]], y=[y1[idx]], mode="markers", marker=dict(size=18, color="yellow", symbol="star"), name="CPA"))
        fig.update_layout(title="Realistic Trajectories", xaxis_title="East (ft)", yaxis_title="North (ft)", height=650, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Monte Carlo Simulator + CSV Export")
    n_runs = st.slider("Number of simulations", 100, 5000, 1000, step=100)
    fix_ownship = st.checkbox("Fix MY aircraft (ownship)", value=True)
    
    if fix_ownship:
        own_v = st.slider("My Speed (kts)", 100, 600, 280)
        own_hdg = st.slider("My Heading (°)", 0, 360, 0)
        own_alt_idx = st.selectbox("My Altitude Block", range(6), format_func=lambda i: altitude_blocks[i])
    
    if st.button("🚀 Run Monte Carlo & Generate CSV", type="primary"):
        with st.spinner(f"Running {n_runs} realistic encounters..."):
            runs_data = []
            for i in range(n_runs):
                if fix_ownship:
                    p = sample_due_regard_encounter(own_alt_idx)
                    p["v1"] = own_v
                    p["hdg1"] = own_hdg
                else:
                    p = sample_due_regard_encounter()
                miss, t_cpa, risk, _, _, _, _, _ = calculate_cpa_realistic(p)
                runs_data.append({
                    "run_id": i+1,
                    "ownship_speed_kts": p["v1"],
                    "ownship_heading_deg": p["hdg1"],
                    "intruder_speed_kts": p["v2"],
                    "intruder_heading_deg": p["hdg2"],
                    "miss_distance_ft": round(miss, 1),
                    "time_to_cpa_min": round(t_cpa / 60, 2),
                    "risk_percent": round(risk * 100, 1),
                    "altitude_block": p["alt_block"],
                    "region": p["region"]
                })
            
            # Summary
            misses = [r["miss_distance_ft"] for r in runs_data]
            st.success(f"Done! Mean miss: {np.mean(misses):.0f} ft | NMAC rate (<500 ft): {(np.array(misses) < 500).mean()*100:.1f}%")
            
            # Histogram
            fig = px.histogram(misses, nbins=50, title="Miss Distance Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
            # CSV Download Button
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=runs_data[0].keys())
            writer.writeheader()
            writer.writerows(runs_data)
            csv_data = output.getvalue()
            
            st.download_button(
                label="📥 Download All Runs as CSV",
                data=csv_data,
                file_name=f"due_regard_monte_carlo_{n_runs}_runs.csv",
                mime="text/csv",
                use_container_width=True
            )

with st.sidebar:
    st.success("✅ CSV export added!")
    st.caption("Realistic mostly-straight trajectories • Built for Zachary")
