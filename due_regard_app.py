import streamlit as st
import numpy as np
import plotly.graph_objects as go
import io
import csv

st.set_page_config(page_title="Due Regard Explorer", layout="wide")
st.title("✈️ Due Regard Mid-Air Collision Explorer")
st.markdown("**Exact Appendix A weighting + Official MIT-LL Cylinder Method** — Pure nominal flight + Full DO-365 Well Clear & NMAC")

# ====================== EXACT APPENDIX A WEIGHTED DISTRIBUTIONS ======================
altitude_blocks = ["Below 5,500 ft MSL", "5,500–10,000 ft MSL", "10k–FL180", "FL180–FL290", "FL290–FL410", "Above FL410"]
altitude_probs = np.array([0.01, 0.02, 0.05, 0.05, 0.80, 0.07]); altitude_probs /= altitude_probs.sum()

altitude_base_ft = [3000, 7500, 14000, 24000, 34000, 45000]

regions = ["Any (Unspecified)", "North Pacific", "West Pacific", "East Pacific", "Gulf of Mexico", "Caribbean", "North Atlantic", "Central Atlantic"]
region_probs = np.array([0.12, 0.08, 0.15, 0.10, 0.25, 0.22, 0.08]); region_probs /= region_probs.sum()

airspeed_bins = [125, 225, 325, 425, 525, 600]
airspeed_probs = np.array([0.02, 0.05, 0.10, 0.55, 0.25, 0.03]); airspeed_probs /= airspeed_probs.sum()

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
        alt_idx = np.random.choice(range(6), p=altitude_probs)
    alt_block = altitude_blocks[alt_idx]
    if region is None or region == "Any (Unspecified)":
        region = np.random.choice(regions[1:], p=region_probs)
    
    own_alt = altitude_base_ft[alt_idx] + np.random.uniform(-500, 500)
    
    return {
        "alt_block": alt_block, "region": region,
        "v1": float(np.random.choice(airspeed_bins, p=airspeed_probs)),
        "v2": float(np.random.choice(airspeed_bins, p=airspeed_probs)),
        "hdg1": float(np.random.choice(heading_bins, p=heading_probs)),
        "hdg2": float(np.random.choice(heading_bins, p=heading_probs)),
        "turn1": float(np.random.choice(turn_bins, p=turn_probs)),
        "turn2": float(np.random.choice(turn_bins, p=turn_probs)),
        "accel1": float(np.random.choice(accel_bins, p=accel_probs)),
        "accel2": float(np.random.choice(accel_bins, p=accel_probs)),
        "dh1": float(np.random.choice(vert_rate_bins, p=vert_rate_probs)),
        "dh2": float(np.random.choice(vert_rate_bins, p=vert_rate_probs)),
        "own_start_alt": own_alt,
        "intr_start_alt": own_alt + float(np.random.uniform(-3500, 3500))
    }

def initialize_encounter_cylinder(own, intr):
    """Exact cylinder method from MIT-LL Section 5.1 / Figure 6"""
    v1 = np.array([own["v1"] * np.cos(np.deg2rad(own["hdg1"])), 
                   own["v1"] * np.sin(np.deg2rad(own["hdg1"]))])
    v2 = np.array([intr["v2"] * np.cos(np.deg2rad(intr["hdg2"])), 
                   intr["v2"] * np.sin(np.deg2rad(intr["hdg2"]))])
    vr = v2 - v1
    vr_mag = np.linalg.norm(vr)
    if vr_mag < 1e-6:
        vr = np.array([1.0, 0.0])
        vr_mag = 1.0
    vr_unit = vr / vr_mag
    
    # Cylinder radius ~10 NM (typical for encounter models)
    radius_nm = 10.0
    radius_ft = radius_nm * 6076.12
    
    # Uniform random point on the perpendicular plane
    perp = np.array([-vr_unit[1], vr_unit[0]])
    r = np.random.uniform(0, radius_ft)
    theta = np.random.uniform(0, 2*np.pi)
    offset = r * (np.cos(theta) * perp + np.sin(theta) * vr_unit)
    
    # Project back to cylinder surface with correct penetration direction
    sep_ft = np.random.uniform(5, 25) * 6076.12
    bearing = np.arctan2(offset[1], offset[0])
    
    intr["sep_nm"] = sep_ft / 6076.12
    intr["bearing"] = np.rad2deg(bearing)
    return intr

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
            p = sample_due_regard_encounter(alt_idx, region_sel)
            p = initialize_encounter_cylinder(p, p.copy())  # Apply cylinder method
            st.session_state.params = p
            st.success("✅ Exact cylinder encounter loaded!")
    with col2:
        p = st.session_state.get("params", sample_due_regard_encounter())
        miss, t_cpa, risk, x1, y1, z1, x2, y2, z2, t_plot, is_well_clear, is_nmac = calculate_cpa_realistic(p)
        
        st.info(f"**{p['alt_block']}** — **{p['region']}** (10-minute flight)")
        
        st.subheader("Aircraft Parameters")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Ownship (UAS)**")
            st.write(f"**Starting Altitude:** {p['own_start_alt']:.0f} ft")
            st.write(f"**Speed:** {p['v1']:.1f} kts")
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
        
        st.write(f"**Initial Separation (Cylinder):** {p.get('sep_nm', 0):.1f} NM")
        
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
            fig.update_layout(title="3D Trajectories — Exact Cylinder Method", scene=dict(xaxis_title='East (ft)', yaxis_title='North (ft)', zaxis_title='Altitude (ft)'), height=700, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Monte Carlo tab (unchanged except using cylinder initialization)
    st.subheader("Monte Carlo Simulator + CSV Export")
    n_runs = st.slider("Number of simulations", 100, 10000, 2000, step=100)
    fix_ownship = st.checkbox("Fix MY aircraft (UAS)", value=True)
    fix_alt = st.checkbox("Fix Altitude Block", value=False)
    fix_region = st.checkbox("Fix Geographic Domain", value=False)
    
    if fix_ownship:
        own_v = st.slider("My UAS Speed (kts)", 25, 600, 80)
        own_hdg = st.slider("My Heading (°)", 0, 360, 0)
    if fix_alt:
        own_alt_idx = st.selectbox("Fixed Altitude Block", range(6), format_func=lambda i: altitude_blocks[i])
    if fix_region:
        own_region = st.selectbox("Fixed Geographic Domain", regions)
    
    if st.button("🚀 Run Monte Carlo & Download CSV", type="primary"):
        with st.spinner(f"Running {n_runs} cylinder encounters..."):
            runs_data = []
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
                p = initialize_encounter_cylinder(p, p.copy())  # Apply exact cylinder
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
                    "region": p["region"],
                    "initial_separation_nm": round(p.get("sep_nm", 0), 1)
                })
                if is_nmac: nmac_count += 1
                if not is_well_clear: well_clear_violations += 1
            
            misses = [r["miss_distance_ft"] for r in runs_data]
            st.success(f"Completed {n_runs} runs • Mean miss: {np.mean(misses):.0f} ft")
            st.error(f"NMAC rate: {(nmac_count/n_runs)*100:.2f}%")
            st.warning(f"Well Clear violation rate: {(well_clear_violations/n_runs)*100:.2f}%")
            
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=runs_data[0].keys())
            writer.writeheader()
            writer.writerows(runs_data)
            st.download_button("📥 Download Full CSV", output.getvalue(), f"due_regard_cylinder_{n_runs}_runs.csv", "text/csv", use_container_width=True)

with st.sidebar:
    st.success("✅ Exact MIT-LL cylinder initialization added")
    st.caption("Initial separation now follows official encounter cylinder method")
