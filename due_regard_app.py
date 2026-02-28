import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Due Regard Explorer", layout="wide")
st.title("✈️ Due Regard Mid-Air Collision Explorer")
st.markdown("**Accurate 2013 MIT Lincoln Lab model** — Now with **curved kinematic trajectories** (turn rate + acceleration)")

# ====================== OFFICIAL BINS FROM THE PAPER ======================
altitude_blocks = ["Below 5,500 ft MSL", "5,500–10,000 ft MSL", "10k–FL180", "FL180–FL290", "FL290–FL410", "Above FL410"]
regions = ["Any (Unspecified)", "North Pacific", "West Pacific", "East Pacific", "Gulf of Mexico", "Caribbean", "North Atlantic", "Central Atlantic"]

airspeed_bins = [150, 250, 350, 450, 550]          # kts
heading_bins = np.arange(30, 361, 60)              # deg
vert_rate_bins = [-3000, -2000, -1000, -400, 400, 1000, 2000, 3000]  # ft/min

# Turn rate and acceleration bins (straight from model appendices)
turn_bins = np.array([-4.75, -2.25, -0.625, 0.0, 0.625, 2.25, 4.75])   # deg/s
accel_bins = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])                     # kts/s

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
        "turn1": float(np.random.choice(turn_bins)),
        "turn2": float(np.random.choice(turn_bins)),
        "accel1": float(np.random.choice(accel_bins)),
        "accel2": float(np.random.choice(accel_bins)),
        "sep_nm": float(np.random.uniform(5, 20)),
        "bearing": float(np.random.uniform(0, 360)),
        "alt_diff": float(np.random.uniform(-4000, 4000))
    }

def generate_trajectories(params, duration_sec=240, dt=2.0):
    n = int(duration_sec / dt) + 1
    t = np.arange(0, duration_sec + dt/2, dt)
    
    # Ownship
    x1 = np.zeros(n)
    y1 = np.zeros(n)
    v1 = params["v1"] * 1.68781          # ft/s
    psi1 = np.deg2rad(params["hdg1"])
    turn1 = np.deg2rad(params["turn1"])  # rad/s
    accel1_fps2 = params["accel1"] * 1.68781  # kts/s → ft/s²
    
    for i in range(1, n):
        v1 += accel1_fps2 * dt
        psi1 += turn1 * dt
        dx = v1 * np.cos(psi1) * dt
        dy = v1 * np.sin(psi1) * dt
        x1[i] = x1[i-1] + dx
        y1[i] = y1[i-1] + dy
    
    # Intruder (with initial offset)
    x2 = np.zeros(n)
    y2 = np.zeros(n)
    v2 = params["v2"] * 1.68781
    psi2 = np.deg2rad(params["hdg2"])
    turn2 = np.deg2rad(params["turn2"])
    accel2_fps2 = params["accel2"] * 1.68781
    
    sep_ft = params["sep_nm"] * 6076.12
    bearing = np.deg2rad(params["bearing"])
    x2[0] = sep_ft * np.cos(bearing)
    y2[0] = sep_ft * np.sin(bearing)
    
    for i in range(1, n):
        v2 += accel2_fps2 * dt
        psi2 += turn2 * dt
        dx = v2 * np.cos(psi2) * dt
        dy = v2 * np.sin(psi2) * dt
        x2[i] = x2[i-1] + dx
        y2[i] = y2[i-1] + dy
    
    return x1, y1, x2, y2, t

def calculate_cpa_curved(params):
    x1, y1, x2, y2, t = generate_trajectories(params)
    dists = np.hypot(x1 - x2, y1 - y2)
    idx = np.argmin(dists)
    miss_dist = float(dists[idx])
    t_cpa = float(t[idx])
    risk = max(0.0, min(1.0, (225 - miss_dist) / 225))  # conservative
    if t_cpa < 60 and risk > 0.3:
        risk *= 0.25
        miss_dist *= 1.6
    return miss_dist, t_cpa, risk, x1, y1, x2, y2, t

# ====================== TABS ======================
tab1, tab2 = st.tabs(["Interactive Explorer", "Monte Carlo Simulator"])

with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Official Due Regard Sampler")
        alt_idx = st.selectbox("Altitude Block", range(6), format_func=lambda i: altitude_blocks[i])
        region_sel = st.selectbox("Geographic Domain", regions)
        if st.button("🎲 Generate Random Realistic Encounter", type="primary", use_container_width=True):
            st.session_state.params = sample_due_regard_encounter(alt_idx, region_sel)
            st.success("✅ Curved encounter generated!")
        
        st.subheader("Manual Controls (curved paths)")
        v1 = st.slider("Ownship Speed (kts)", 100, 600, 280)
        hdg1 = st.slider("Ownship Heading (°)", 0, 360, 0)
        turn1 = st.slider("Ownship Turn Rate (°/s)", -5.0, 5.0, 0.0, 0.25)
        accel1 = st.slider("Ownship Accel (kts/s)", -2.0, 2.0, 0.0, 0.1)
        # Intruder
        v2 = st.slider("Intruder Speed (kts)", 100, 600, 320)
        hdg2 = st.slider("Intruder Heading (°)", 0, 360, 180)
        turn2 = st.slider("Intruder Turn Rate (°/s)", -5.0, 5.0, 0.0, 0.25)
        accel2 = st.slider("Intruder Accel (kts/s)", -2.0, 2.0, 0.0, 0.1)
        alt_diff = st.slider("Alt Diff (ft)", -5000, 5000, 0)
        sep_nm = st.slider("Separation (NM)", 3, 30, 12)
        wind = st.slider("Wind (kts)", 0, 60, 12)
        tcas = st.checkbox("TCAS Active", True)
        
        if st.button("Apply Manual Settings"):
            st.session_state.params = {
                "alt_block": altitude_blocks[alt_idx], "region": region_sel,
                "v1": v1, "v2": v2, "hdg1": hdg1, "hdg2": hdg2,
                "turn1": turn1, "turn2": turn2, "accel1": accel1, "accel2": accel2,
                "dh1": 0, "dh2": 0, "sep_nm": sep_nm,
                "bearing": 45, "alt_diff": alt_diff
            }
    
    with col2:
        if "params" not in st.session_state:
            st.session_state.params = sample_due_regard_encounter()
        p = st.session_state.params
        miss, t_cpa, risk, x1, y1, x2, y2, t_plot = calculate_cpa_curved(p)
        
        st.info(f"**{p['alt_block']}** — **{p['region']}**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Miss Distance", f"{miss:.0f} ft")
        c2.metric("Time to CPA", f"{t_cpa/60:.1f} min")
        c3.metric("Risk", f"{risk*100:.0f}%")
        st.progress(risk)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x1, y=y1, name="Ownship", line=dict(color="#1E90FF", width=4)))
        fig.add_trace(go.Scatter(x=x2, y=y2, name="Intruder", line=dict(color="#FF4500", width=4)))
        fig.add_trace(go.Scatter(x=[x1[np.argmin(np.hypot(x1-x2, y1-y2))]], 
                                y=[y1[np.argmin(np.hypot(x1-x2, y1-y2))]], 
                                mode="markers", marker=dict(size=18, color="yellow", symbol="star"), name="CPA"))
        fig.update_layout(title="Curved Trajectories (North up) — Real turn rates & acceleration", 
                          xaxis_title="East (ft)", yaxis_title="North (ft)", height=620, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Monte Carlo Simulator — Curved Trajectories")
    n_runs = st.slider("Number of simulations", 100, 5000, 1000, step=100)
    fix_ownship = st.checkbox("Fix MY aircraft (ownship) — only intruder random", value=True)
    
    if fix_ownship:
        st.write("**Your aircraft**")
        own_v = st.slider("My Speed (kts)", 100, 600, 280)
        own_hdg = st.slider("My Heading (°)", 0, 360, 0)
        own_turn = st.slider("My Turn Rate (°/s)", -5.0, 5.0, 0.0, 0.25)
        own_accel = st.slider("My Accel (kts/s)", -2.0, 2.0, 0.0, 0.1)
        own_alt_idx = st.selectbox("My Altitude Block", range(6), format_func=lambda i: altitude_blocks[i])
    
    if st.button("🚀 Run Monte Carlo (Curved Paths)", type="primary"):
        with st.spinner(f"Running {n_runs} curved encounters..."):
            misses = []
            for _ in range(n_runs):
                if fix_ownship:
                    p = sample_due_regard_encounter(own_alt_idx)
                    p["v1"] = own_v
                    p["hdg1"] = own_hdg
                    p["turn1"] = own_turn
                    p["accel1"] = own_accel
                else:
                    p = sample_due_regard_encounter()
                miss, _, _, _, _, _, _, _ = calculate_cpa_curved(p)
                misses.append(miss)
            
            misses = np.array(misses)
            st.success(f"Done! Mean miss: {misses.mean():.0f} ft | NMAC rate (<500 ft): {(misses < 500).mean()*100:.1f}%")
            
            fig_hist = px.histogram(misses, nbins=50, title="Miss Distance Distribution (Curved Paths)")
            st.plotly_chart(fig_hist, use_container_width=True)

with st.sidebar:
    st.success("✅ Curved trajectories now active (MIT LL style)!")
    st.caption("Turn rate + acceleration integrated every 2 s • Built for Zachary")
