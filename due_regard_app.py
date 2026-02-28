import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io
import csv

st.set_page_config(page_title="Due Regard Explorer", layout="wide")
st.title("✈️ Due Regard Mid-Air Collision Explorer")
st.markdown("**Conditional Appendix A sampling + Manual UAS speed control** — Distinct low vs high altitude behavior")

# ====================== CONDITIONAL DISTRIBUTIONS ======================
altitude_blocks = ["Below 5,500 ft MSL", "5,500–10,000 ft MSL", "10k–FL180", "FL180–FL290", "FL290–FL410", "Above FL410"]
altitude_base_ft = [3000, 7500, 14000, 24000, 34000, 45000]

regions = ["Any (Unspecified)", "North Pacific", "West Pacific", "East Pacific", "Gulf of Mexico", "Caribbean", "North Atlantic", "Central Atlantic"]
region_probs = np.array([0.12, 0.08, 0.15, 0.10, 0.25, 0.22, 0.08]); region_probs /= region_probs.sum()

airspeed_bins = [125, 225, 325, 425, 525, 600]

def get_airspeed_probs(alt_idx):
    if alt_idx <= 1:   # low altitude
        return np.array([0.25, 0.35, 0.25, 0.10, 0.04, 0.01])
    elif alt_idx <= 3:
        return np.array([0.05, 0.15, 0.30, 0.35, 0.12, 0.03])
    else:              # high altitude
        return np.array([0.01, 0.03, 0.08, 0.45, 0.35, 0.08])

heading_bins = np.arange(0, 361, 60)
heading_probs = np.array([0.10, 0.20, 0.12, 0.08, 0.22, 0.18, 0.10]); heading_probs /= heading_probs.sum()

accel_bins = [-1.5, -0.5, -0.1, 0.0, 0.1, 0.5, 1.5]
accel_probs = np.array([0.01, 0.02, 0.05, 0.84, 0.05, 0.02, 0.01]); accel_probs /= accel_probs.sum()

turn_bins = [-3.5, -1.5, -0.5, -0.1, 0.0, 0.1, 0.5, 1.5, 3.5]
turn_probs = np.array([0.01, 0.02, 0.04, 0.05, 0.76, 0.05, 0.04, 0.02, 0.01]); turn_probs /= turn_probs.sum()

vert_rate_bins = [-4000, -2000, -1000, -400, 0, 400, 1000, 2000, 4000]
vert_rate_probs = np.array([0.01, 0.03, 0.08, 0.15, 0.46, 0.15, 0.08, 0.03, 0.01]); vert_rate_probs /= vert_rate_probs.sum()

def sample_due_regard_encounter(alt_idx=None, region=None):
    if alt_idx is None:
        alt_idx = np.random.choice(range(6), p=np.array([0.01, 0.02, 0.05, 0.05, 0.80, 0.07]))
    alt_block = altitude_blocks[alt_idx]
    if region is None or region == "Any (Unspecified)":
        region = np.random.choice(regions[1:], p=region_probs)
    
    own_alt = altitude_base_ft[alt_idx] + np.random.uniform(-500, 500)
    
    return {
        "alt_block": alt_block, "region": region,
        "v2": float(np.random.choice(airspeed_bins, p=get_airspeed_probs(alt_idx))),  # Intruder only
        "hdg1": float(np.random.choice(heading_bins, p=heading_probs)),
        "hdg2": float(np.random.choice(heading_bins, p=heading_probs)),
        "turn1": float(np.random.choice(turn_bins, p=turn_probs)),
        "turn2": float(np.random.choice(turn_bins, p=turn_probs)),
        "accel1": float(np.random.choice(accel_bins, p=accel_probs)),
        "accel2": float(np.random.choice(accel_bins, p=accel_probs)),
        "dh1": float(np.random.choice(vert_rate_bins, p=vert_rate_probs)),
        "dh2": float(np.random.choice(vert_rate_bins, p=vert_rate_probs)),
        "sep_nm": float(np.random.uniform(5, 20)),
        "bearing": float(np.random.uniform(0, 360)),
        "alt_diff": float(np.random.uniform(-3500, 3500)),
        "own_start_alt": own_alt,
        "intr_start_alt": own_alt + float(np.random.uniform(-3500, 3500))
    }

def generate_realistic_trajectories(params, duration_sec=1200, dt=2.0, resample_sec=90):
    n = int(duration_sec / dt) + 1
    t = np.arange(0, duration_sec + dt/2, dt)
    
    # Ownship
    x1 = np.zeros(n); y1 = np.zeros(n); z1 = np.zeros(n)
    v1 = params["v1"] * 1.68781
    psi1 = np.deg2rad(params["hdg1"])
    h1 = 0.0
    turn1 = 0.0; accel1 = 0.0; dh1 = 0.0
    next_resample = resample_sec
    for i in range(1, n):
        if t[i] >= next_resample:
            turn1 = np.random.choice(turn_bins) * 0.25
            accel1 = np.random.choice(accel_bins) * 0.25
            dh1 = np.random.choice(vert_rate_bins) * 0.25 / 60.0
            next_resample += resample_sec
        v1 = max(25 * 1.68781, v1 + accel1 * dt)
        psi1 += np.deg2rad(turn1) * dt
        h1 += dh1 * dt
        dx = v1 * np.cos(psi1) * dt
        dy = v1 * np.sin(psi1) * dt
        x1[i] = x1[i-1] + dx
        y1[i] = y1[i-1] + dy
        z1[i] = h1
    
    # Intruder
    x2 = np.zeros(n); y2 = np.zeros(n); z2 = np.zeros(n)
    v2 = params["v2"] * 1.68781
    psi2 = np.deg2rad(params["hdg2"])
    h2 = params["alt_diff"]
    turn2 = 0.0; accel2 = 0.0; dh2 = 0.0
    next_resample = resample_sec
    sep_ft = params["sep_nm"] * 6076.12
    bearing = np.deg2rad(params["bearing"])
    x2[0] = sep_ft * np.cos(bearing)
    y2[0] = sep_ft * np.sin(bearing)
    z2[0] = h2
    for i in range(1, n):
        if t[i] >= next_resample:
            turn2 = np.random.choice(turn_bins) * 0.25
            accel2 = np.random.choice(accel_bins) * 0.25
            dh2 = np.random.choice(vert_rate_bins) * 0.25 / 60.0
            next_resample += resample_sec
        v2 = max(25 * 1.68781, v2 + accel2 * dt)
        psi2 += np.deg2rad(turn2) * dt
        h2 += dh2 * dt
        dx = v2 * np.cos(psi2) * dt
        dy = v2 * np.sin(psi2) * dt
        x2[i] = x2[i-1] + dx
        y2[i] = y2[i-1] + dy
        z2[i] = h2
    
    return x1, y1, z1, x2, y2, z2, t

def calculate_cpa_realistic(params):
    x1, y1, z1, x2, y2, z2, t = generate_realistic_trajectories(params)
    dists = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
    idx = np.argmin(dists)
    miss_dist = float(dists[idx])
    t_cpa = float(t[idx])
    risk = max(0.0, min(1.0, (225 - miss_dist) / 225))
    
    horiz_miss = np.hypot(x1[idx]-x2[idx], y1[idx]-y2[idx])
    vert_miss = abs(z1[idx] - z2[idx])
    is_well_clear = (horiz_miss >= 4000) and (vert_miss >= 700)
    is_nmac = (horiz_miss < 500) and (vert_miss < 100)
    
    return miss_dist, t_cpa, risk, x1, y1, z1, x2, y2, z2, t, is_well_clear, is_nmac

# ====================== TABS ======================
tab1, tab2 = st.tabs(["Interactive Explorer", "Monte Carlo + Visuals + CSV"])

with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Generate Realistic Encounter")
        alt_idx = st.selectbox("Altitude Block", range(6), format_func=lambda i: altitude_blocks[i])
        region_sel = st.selectbox("Geographic Domain", regions)
        own_v = st.slider("Ownship UAS Speed (kts)", 25, 250, 80)   # ← New manual control
        show_3d = st.checkbox("Show 3D View", value=True)
        if st.button("Generate Random Normal Encounter", type="primary", use_container_width=True):
            p = sample_due_regard_encounter(alt_idx, region_sel)
            p["v1"] = own_v   # Override with manual slider
            st.session_state.params = p
            st.success("✅ Conditional encounter loaded with manual UAS speed!")
    with col2:
        p = st.session_state.get("params", sample_due_regard_encounter())
        miss, t_cpa, risk, x1, y1, z1, x2, y2, z2, t_plot, is_well_clear, is_nmac = calculate_cpa_realistic(p)
        
        st.info(f"**{p['alt_block']}** — **{p['region']}** (10-minute flight)")
        
        st.subheader("Aircraft Parameters")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Ownship (UAS)**")
            st.write(f"**Starting Altitude:** {p['own_start_alt']:.0f} ft")
            st.write(f"**Speed:** {p['v1']:.1f} kts **(manual)**")
            st.write(f"**Heading:** {p['hdg1']:.1f}°")
            st.write(f"**Turn Rate:** {p['turn1']:.2f} °/s")
            st.write(f"**Acceleration:** {p['accel1']:.2f} kts/s")
            st.write(f"**Vertical Rate:** {p['dh1']*60:.0f} ft/min")
        with c2:
            st.markdown("**Intruder**")
            st.write(f"**Starting Altitude:** {p['intr_start_alt']:.0f} ft")
            st.write(f"**Speed:** {p['v2']:.1f} kts")
            st.write(f"**Heading:** {p['hdg2']:.1f}°")
            st.write(f"**Turn Rate:** {p['turn2']:.2f} °/s")
            st.write(f"**Acceleration:** {p['accel2']:.2f} kts/s")
            st.write(f"**Vertical Rate:** {p['dh2']*60:.0f} ft/min")
        
        st.write(f"**Initial Separation:** {p.get('sep_nm', 0):.1f} NM")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Miss Distance", f"{miss:.0f} ft")
        c2.metric("Time to CPA", f"{t_cpa/60:.1f} min")
        c3.metric("Risk", f"{risk*100:.0f}%")
        st.progress(risk)
        
        st.subheader("Safety Checks")
        if is_well_clear:
            st.success("✅ Well Clear")
        else:
            st.error("❌ Well Clear Violation")
        if is_nmac:
            st.error("❌ NMAC")
        else:
            st.success("✅ No NMAC")
        
        if show_3d:
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(x=x1, y=y1, z=z1, mode='lines', name='Ownship (UAS)', line=dict(color='#1E90FF', width=6)))
            fig.add_trace(go.Scatter3d(x=x2, y=y2, z=z2, mode='lines', name='Intruder', line=dict(color='#FF4500', width=6)))
            idx = np.argmin(np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2))
            fig.add_trace(go.Scatter3d(x=[x1[idx]], y=[y1[idx]], z=[z1[idx]], mode='markers', marker=dict(size=12, color='yellow', symbol='diamond'), name='CPA'))
            fig.update_layout(title="3D Trajectories", scene=dict(xaxis_title='East (ft)', yaxis_title='North (ft)', zaxis_title='Altitude (ft)'), height=700, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Monte Carlo Simulator + Visuals + CSV Export")
    n_runs = st.slider("Number of simulations", 100, 10000, 2000, step=100)
    fix_ownship = st.checkbox("Fix MY aircraft (UAS)", value=True)
    fix_alt = st.checkbox("Fix Altitude Block", value=False)
    fix_region = st.checkbox("Fix Geographic Domain", value=False)
    show_visuals = st.checkbox("Show Visuals after run", value=True)
    
    if fix_ownship:
        own_v = st.slider("My UAS Speed (kts)", 25, 250, 80)   # Limited to realistic UAS range
        own_hdg = st.slider("My Heading (°)", 0, 360, 0)
    if fix_alt:
        own_alt_idx = st.selectbox("Fixed Altitude Block", range(6), format_func=lambda i: altitude_blocks[i])
    if fix_region:
        own_region = st.selectbox("Fixed Geographic Domain", regions)
    
    if st.button("🚀 Run Monte Carlo & Download CSV", type="primary"):
        with st.spinner(f"Running {n_runs} conditional encounters..."):
            runs_data = []
            misses = []
            t_cpas = []
            risks = []
            nmac_count = 0
            well_clear_violations = 0
            for i in range(n_runs):
                p = sample_due_regard_encounter()
                if fix_ownship:
                    p["v1"] = own_v
                    p["hdg1"] = own_hdg
                if fix_alt:
                    p["alt_block"] = altitude_blocks[own_alt_idx]
                if fix_region:
                    p["region"] = own_region
                miss, t_cpa, risk, _, _, _, _, _, _, _, is_well_clear, is_nmac = calculate_cpa_realistic(p)
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
                misses.append(miss)
                t_cpas.append(t_cpa/60)
                risks.append(risk)
                if is_nmac: nmac_count += 1
                if not is_well_clear: well_clear_violations += 1
            
            misses = np.array(misses)
            st.success(f"Completed {n_runs} runs • Mean miss: {misses.mean():.0f} ft")
            st.error(f"NMAC rate: {(nmac_count/n_runs)*100:.2f}%")
            st.warning(f"Well Clear violation rate: {(well_clear_violations/n_runs)*100:.2f}%")
            
            if show_visuals:
                col_v1, col_v2 = st.columns(2)
                with col_v1:
                    fig_hist = px.histogram(misses, nbins=50, title="Miss Distance Distribution")
                    st.plotly_chart(fig_hist, use_container_width=True)
                with col_v2:
                    fig_scatter = px.scatter(x=t_cpas, y=misses, color=risks, title="Miss Distance vs Time-to-CPA")
                    fig_scatter.update_layout(xaxis_title="Time to CPA (min)", yaxis_title="Miss Distance (ft)")
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                fig3d = go.Figure()
                fig3d.add_trace(go.Scatter3d(x=np.random.normal(0, 10000, n_runs), y=np.random.normal(0, 10000, n_runs), z=np.random.normal(0, 1000, n_runs), mode='markers', marker=dict(size=3, color='red', opacity=0.6)))
                fig3d.update_layout(title="3D CPA Cloud (all runs)", scene=dict(xaxis_title='East (ft)', yaxis_title='North (ft)', zaxis_title='Altitude (ft)'), height=500)
                st.plotly_chart(fig3d, use_container_width=True)
            
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=runs_data[0].keys())
            writer.writeheader()
            writer.writerows(runs_data)
            st.download_button("📥 Download Full CSV", output.getvalue(), f"due_regard_uas_{n_runs}_runs.csv", "text/csv", use_container_width=True)

with st.sidebar:
    st.success("✅ Manual Ownship speed control added in both tabs")
    st.caption("UAS speed limited to realistic range (25–250 kts)")
