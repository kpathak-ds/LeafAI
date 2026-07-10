# sensor_analytics.py - {t["sens_title"]} Module
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import io
import joblib
import os
import requests

# ---------------- BLYNK TOKEN ----------------
BLYNK_TOKEN = "v67lqpGxQf8zfnvepJKkzpBHvbXrWYnN"

# ---------------- FETCH DATA ----------------
def get_blynk_data(pin):
    try:
        url = f"https://blynk.cloud/external/api/get?token={BLYNK_TOKEN}&{pin}"
        response = requests.get(url, timeout=3)
        return float(response.text)
    except:
        return None

def get_blynk_string(pin):
    try:
        url = f"https://blynk.cloud/external/api/get?token={BLYNK_TOKEN}&{pin}"
        response = requests.get(url, timeout=3)
        return response.text
    except:
        return None
def init_sensor_states():
    defaults = {
        "sensor_moisture": 45.0,
        "sensor_ph": 6.5,
        "sensor_temp": 25.0,
        "sensor_humidity": 60.0,
        "sensor_light": 10000.0,
        "sensor_soil_temp": 22.0,
        "sensor_analyzed": False
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def evaluate_status(val, optimal_range, moderate_range):
    if optimal_range[0] <= val <= optimal_range[1]:
        return "Optimal", "green", "var(--healthy)", "🟢"
    elif moderate_range[0] <= val <= moderate_range[1]:
        return "Moderate", "yellow", "var(--warning)", "🟡"
    else:
        return "Critical", "red", "var(--danger)", "🔴"

def draw_circular_gauge(score, status):
    stroke_dash = (score / 100) * 283
    if status == "Healthy":
        color = "#43A047"
    elif status == "Moderate":
        color = "#FB8C00"
    elif status == "Poor":
        color = "#F44336"
    else:
        color = "#E53935"
        
    gauge_html = f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 15px; background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 16px; box-shadow: var(--card-shadow);">
        <h4 style="margin: 0 0 10px 0; color: var(--text-primary); font-family: 'Space Grotesk', sans-serif; font-size: 1.1rem; font-weight: 700;">Overall Field Health Index</h4>
        <svg width="170" height="170" viewBox="0 0 100 100">
            <circle cx="50" cy="50" r="45" fill="none" stroke="var(--border-color)" stroke-width="7" />
            <circle cx="50" cy="50" r="45" fill="none" stroke="{color}" stroke-width="7"
                    stroke-dasharray="283" stroke-dashoffset="{283 - stroke_dash}"
                    stroke-linecap="round" transform="rotate(-90 50 50)"
                    style="transition: stroke-dashoffset 0.8s ease-out;" />
            <text x="50" y="47" text-anchor="middle" dominant-baseline="middle" font-family="'Space Grotesk', sans-serif" font-size="18" font-weight="700" fill="var(--text-primary)">{int(score)}%</text>
            <text x="50" y="66" text-anchor="middle" dominant-baseline="middle" font-family="'Plus Jakarta Sans', sans-serif" font-size="7.5" font-weight="700" fill="var(--text-secondary)">{status.upper()}</text>
        </svg>
    </div>
    """
    return gauge_html

def generate_ai_interpretation(moisture, ph, temp, ec, n, p, k, urea):
    clauses = []
    
    # Moisture
    if moisture < 35:
        clauses.append(f"Soil moisture is critical at {moisture:.1f}%, indicating significant water stress that impairs transpiration and nutrient uptake.")
    elif moisture > 75:
        clauses.append(f"Excessive soil moisture of {moisture:.1f}% presents a waterlogging risk, which can lead to anaerobic root zones.")
    else:
        clauses.append(f"Soil moisture levels are optimal ({moisture:.1f}%), encouraging efficient capillary absorption and soil microbial respiration.")
        
    # pH
    if ph < 5.8:
        clauses.append(f"Acidic soil conditions (pH {ph:.1f}) are present, which may bind vital phosphorus and decrease standard biological nitrogen fixation.")
    elif ph > 7.5:
        clauses.append(f"Alkaline soil conditions (pH {ph:.1f}) could trigger nutrient lockout, reducing iron, manganese, and boron bioavailability.")
    else:
        clauses.append(f"Soil pH is balanced in the neutral zone ({ph:.1f}), ensuring ideal chemical conditions for nutrient assimilation.")
        
    # NPK
    low_nutrients = []
    if n < 15:
        low_nutrients.append(f"Nitrogen ({n:.1f} mg/kg)")
    if p < 45:
        low_nutrients.append(f"Phosphorus ({p:.1f} mg/kg)")
    if k < 12:
        low_nutrients.append(f"Potassium ({k:.1f} mg/kg)")
        
    if low_nutrients:
        clauses.append("Macronutrient deficiencies in {} are limiting crop metabolism, which could slow down cell growth and leaf expansion if uncorrected.".format(", ".join(low_nutrients)))
    else:
        clauses.append("The crop macronutrient status (NPK) is fully sustained, reinforcing vegetative vigor, robust root structure, and osmotic regulation.")
        
    # Salinity (EC)
    if ec > 3.5:
        clauses.append(f"Soil salinity is elevated (EC: {ec:.2f} dS/m), increasing osmotic stress and hindering standard water flow into the root system.")
    elif ec < 1.0:
        clauses.append(f"Very low soluble salt levels (EC: {ec:.2f} dS/m) suggest overall nutrient depletion in the active rhizosphere.")
    else:
        clauses.append(f"Soluble salts are stable (EC: {ec:.2f} dS/m), maintaining a suitable osmotic gradient for root cells.")
        
    # Summary
    if len(low_nutrients) >= 2 or moisture < 35 or moisture > 75:
        clauses.append("Dynamic modeling shows the crop is under agronomic stress. Apply targeted adjustments immediately.")
    else:
        clauses.append("Overall, crop telemetry shows stable performance indices, reflecting a healthy soil ecosystem.")
        
    return " ".join(clauses)

def get_recommendations(moisture, ph, temp, ec, n, p, k, soil_temp):
    recs = []
    if moisture < 35:
        recs.append("💧 **Increase irrigation frequency:** Set up drip systems or increase watering frequency to achieve a target soil moisture of 45-60%.")
    elif moisture > 75:
        recs.append("🚿 **Reduce irrigation:** Halt watering and check soil drainage pathways to avoid root aeration failure.")
        
    if ph < 6.0:
        recs.append("🪵 **Apply Agricultural Lime:** Add calcium carbonate or dolomite lime to correct the soil acidity and unlock phosphorus.")
    elif ph > 7.5:
        recs.append("🍁 **Apply Elemental Sulfur:** Incorporate agricultural sulfur or organic peat moss to neutralize alkaline soil and restore iron uptake.")
        
    if n < 15:
        recs.append("🌱 **Apply Nitrogen fertilizer:** Feed the crop with nitrogen-rich amendments (e.g. Urea or organic fish emulsion).")
    elif n > 35:
        recs.append("⚠️ **Avoid Nitrogen additions:** High nitrogen promotes lush foliage but delays flowering and weakens plant structure.")
        
    if p < 45:
        recs.append("🦴 **Apply balanced NPK / Phosphate:** Supply bone meal, rock phosphate, or triple superphosphate to bolster root development.")
        
    if k < 12:
        recs.append("🍂 **Apply Potash (Potassium):** Apply sulfate of potash or potassium chloride to strengthen cell walls and boost disease immunity.")
        
    if temp > 35 or soil_temp > 32:
        recs.append("🌾 **Apply organic mulch:** Lay down straw or woodchips to insulate the soil against extreme solar heating and slow down water evaporation.")
    
    # Fallback recommendations if everything is perfect
    if not recs:
        recs.append("✅ **Maintain current protocols:** Telemetry parameters are well-balanced. Keep up standard watering and composting.")
        recs.append("🔍 **Regular monitoring:** Schedule next telemetry check in 7 days to track nutrient depletion.")
        
    return recs

def calculate_field_health(moisture, ph, ec, n, p, k):
    # Moisture score (ideal 45-65%)
    if 45 <= moisture <= 65:
        moisture_score = 100.0
    else:
        moisture_score = max(0.0, 100.0 - abs(moisture - 55.0) * 2.2)
        
    # pH score (ideal 6.0-7.2)
    if 6.0 <= ph <= 7.2:
        ph_score = 100.0
    else:
        ph_score = max(0.0, 100.0 - abs(ph - 6.6) * 25.0)
        
    # Nitrogen score (ideal 15-30)
    if 15 <= n <= 30:
        n_score = 100.0
    else:
        n_score = max(0.0, 100.0 - (15 - n) * 8.0) if n < 15 else max(0.0, 100.0 - (n - 30) * 4.0)
        
    # Phosphorus score (ideal 45-70)
    if 45 <= p <= 70:
        p_score = 100.0
    else:
        p_score = max(0.0, 100.0 - (45 - p) * 3.5)
        
    # Potassium score (ideal 12-25)
    if 12 <= k <= 25:
        k_score = 100.0
    else:
        k_score = max(0.0, 100.0 - (12 - k) * 10.0) if k < 12 else max(0.0, 100.0 - (k - 25) * 5.0)
        
    # EC score (ideal 1.0-3.5)
    if 1.0 <= ec <= 3.5:
        ec_score = 100.0
    else:
        ec_score = max(0.0, 100.0 - (1.0 - ec) * 100.0) if ec < 1.0 else max(0.0, 100.0 - (ec - 3.5) * 45.0)
        
    score = (moisture_score + ph_score + n_score + p_score + k_score + ec_score) / 6.0
    score = np.clip(score, 0, 100)
    
    if score >= 80:
        status = "Healthy"
    elif score >= 60:
        status = "Moderate"
    elif score >= 40:
        status = "Poor"
    else:
        status = "Critical"
        
    return score, status

def generate_pdf_report(moisture, ph, temp, humidity, light, soil_temp, ec, n, p, k, urea, score, status, interpretation, recommendations, predicted_crop):
    buffer = io.BytesIO()
    
    # Setup styling defaults
    plt.rcParams['font.family'] = 'sans-serif'
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5), dpi=150)
    plt.suptitle("LEAFHEALTH SYSTEM - SMART SENSOR ANALYTICS REPORT", fontsize=15, fontweight='bold', color='#1b5e20', y=0.97)
    
    # 1. Gauge / Health Score Panel
    ax = axes[0, 0]
    ax.axis('off')
    # Circular progress arc mock
    circle_bg = plt.Circle((0.5, 0.5), 0.35, color='#eeeeee', fill=False, linewidth=12)
    color_map = {"Healthy": "#43a047", "Moderate": "#ffa726", "Poor": "#ff7043", "Critical": "#e53935"}
    color = color_map.get(status, "#e53935")
    
    circle_fg = plt.Circle((0.5, 0.5), 0.35, color=color, fill=False, linewidth=12, linestyle='-')
    ax.add_patch(circle_bg)
    ax.add_patch(circle_fg)
    
    ax.text(0.5, 0.5, f"{int(score)}%", fontsize=34, fontweight='bold', ha='center', va='center', color='#111111')
    ax.text(0.5, 0.12, f"FIELD STATUS: {status.upper()}", fontsize=11, fontweight='bold', ha='center', va='center', color=color)
    ax.set_title("Overall Field Health Score", fontsize=11, fontweight='bold', color='#2e7d32', pad=10)
    
    # 2. Input Telemetry Summary Bar Chart
    ax = axes[0, 1]
    inputs = ['Moisture\n(%)', 'pH\n(x10)', 'Temp\n(°C)', 'Humidity\n(%)', 'Soil Temp\n(°C)']
    input_vals = [moisture, ph * 10, temp, humidity, soil_temp]
    bars = ax.bar(inputs, input_vals, color='#a5d6a7', edgecolor='#2e7d32', width=0.45)
    ax.set_ylim(0, 110)
    ax.set_title("Environmental & Soil Inputs", fontsize=11, fontweight='bold', color='#2e7d32')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}' if bar.get_x() != 0.8 else f'{height/10:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
        
    # 3. Calculated Nutrient Levels
    ax = axes[1, 0]
    nutrients = ['Nitrogen', 'Phosphorus', 'Potassium', 'Urea', 'EC (x10)']
    nutr_vals = [n, p, k, urea, ec * 10]
    colors = ['#81c784', '#a5d6a7', '#c8e6c9', '#e8f5e9', '#80deea']
    y_pos = np.arange(len(nutrients))
    ax.barh(y_pos, nutr_vals, color=colors, edgecolor='#555555', height=0.55)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(nutrients, fontsize=9)
    ax.set_title("Calculated Nutrients & Indicators", fontsize=11, fontweight='bold', color='#2e7d32')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i, val in enumerate(nutr_vals):
        display_val = f"{val:.1f}" if i != 4 else f"{val/10:.2f}"
        ax.text(val + 1, i, display_val, ha='left', va='center', fontsize=8, fontweight='bold')
        
    # 4. Text interpretation summary
    ax = axes[1, 1]
    ax.axis('off')
    ax.set_title("AI Diagnostics & Recommendations", fontsize=11, fontweight='bold', color='#2e7d32', pad=10)
    
    wrapped_interpretation = interpretation[:260] + "..." if len(interpretation) > 260 else interpretation
    text_content = f"AI Diagnosis:\n{wrapped_interpretation}\n\nOptimal Crop Prediction: {predicted_crop}\n\nKey Recommendations:\n"
    for r in recommendations[:3]:
        clean_rec = r.replace("**", "").replace("💧", "-").replace("🚿", "-").replace("🌱", "-").replace("🪵", "-").replace("🍁", "-").replace("🦴", "-").replace("🍂", "-").replace("☀️", "-").replace("🌾", "-").replace("✅", "-").replace("🔍", "-")
        text_content += f"{clean_rec}\n"
        
    ax.text(0.0, 0.95, text_content, transform=ax.transAxes, fontsize=8, va='top', wrap=True, color='#222222', linespacing=1.4)
    
    plt.tight_layout(pad=2.0)
    
    # Save to buffer using PdfPages
    with PdfPages(buffer) as pdf:
        pdf.savefig(fig)
        
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()

def render_sensor_analytics(t):
    init_sensor_states()
    
    # Header Panel
    st.markdown(f"""
    <div class="agri-card" style="margin-bottom: 24px; background: linear-gradient(135deg, rgba(46, 125, 50, 0.06) 0%, rgba(255, 255, 255, 0.8) 100%);">
        <h2 style="font-size: 2rem; font-weight: 700; color: var(--primary-green); margin: 0; font-family:'Space Grotesk',sans-serif;"><span class="material-symbols-outlined">sensors</span> {t["sens_title"]}</h2>
        <p style="font-size: 0.95rem; color: var(--text-secondary); margin-top: 6px;">{t["sens_desc"]}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Split layout into Input Panel & Results
    col_input, col_results = st.columns([1, 2], gap="large")
    
    with col_input:
        st.markdown(f'<p class="section-header"><span class="material-symbols-outlined">settings_input_component</span> {t["sens_input_panel"]}</p>', unsafe_allow_html=True)
        
        # Sub-card wrapper for form input
        with st.container(border=True):
            moisture = st.number_input("Soil Moisture (%)", min_value=0.0, max_value=100.0, value=st.session_state.sensor_moisture, step=1.0, format="%.1f", key="input_moisture")
            ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=st.session_state.sensor_ph, step=0.1, format="%.2f", key="input_ph")
            temp = st.number_input("Air Temperature (°C)", min_value=-20.0, max_value=60.0, value=st.session_state.sensor_temp, step=0.5, format="%.1f", key="input_temp")
            humidity = st.number_input("Air Humidity (%)", min_value=0.0, max_value=100.0, value=st.session_state.sensor_humidity, step=1.0, format="%.1f", key="input_humidity")
            light = st.number_input("Light Intensity (Lux)", min_value=0.0, max_value=150000.0, value=st.session_state.sensor_light, step=100.0, format="%.0f", key="input_light")
            soil_temp = st.number_input("Soil Temperature (°C)", min_value=-20.0, max_value=60.0, value=st.session_state.sensor_soil_temp, step=0.5, format="%.1f", key="input_soil_temp")
            
            # Action Buttons Row
            st.markdown("<div style='margin-bottom:10px;'></div>", unsafe_allow_html=True)
            if st.button(t["sens_fetch"], use_container_width=True):
                with st.spinner("Connecting to IoT Sensors..."):
                    # Using V1 to V6 as standard pins for the 6 metrics
                    b_moist = get_blynk_data("V1")
                    b_ph = get_blynk_data("V2")
                    b_temp = get_blynk_data("V3")
                    b_hum = get_blynk_data("V4")
                    b_light = get_blynk_data("V5")
                    b_soil_temp = get_blynk_data("V6")

                    if b_moist is not None:
                        st.session_state.sensor_moisture = float(b_moist)
                    if b_ph is not None:
                        st.session_state.sensor_ph = float(b_ph)
                    if b_temp is not None:
                        st.session_state.sensor_temp = float(b_temp)
                    if b_hum is not None:
                        st.session_state.sensor_humidity = float(b_hum)
                    if b_light is not None:
                        st.session_state.sensor_light = float(b_light)
                    if b_soil_temp is not None:
                        st.session_state.sensor_soil_temp = float(b_soil_temp)

                    if any(v is not None for v in [b_moist, b_ph, b_temp, b_hum, b_light, b_soil_temp]):
                        st.toast("Data fetched successfully from Blynk IoT!")
                        st.session_state.sensor_analyzed = True
                    else:
                        st.toast("Failed to connect to Blynk Cloud. Ensure device is online.", icon="⚠️")
                    st.rerun()

            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button(t["sens_reset"], use_container_width=True):
                    st.session_state.sensor_moisture = 45.0
                    st.session_state.sensor_ph = 6.5
                    st.session_state.sensor_temp = 25.0
                    st.session_state.sensor_humidity = 60.0
                    st.session_state.sensor_light = 10000.0
                    st.session_state.sensor_soil_temp = 22.0
                    st.session_state.sensor_analyzed = False
                    st.rerun()
            with btn_col2:
                if st.button(t["sens_analyze"], type="primary", use_container_width=True):
                    # Save current input values into session state
                    st.session_state.sensor_moisture = moisture
                    st.session_state.sensor_ph = ph
                    st.session_state.sensor_temp = temp
                    st.session_state.sensor_humidity = humidity
                    st.session_state.sensor_light = light
                    st.session_state.sensor_soil_temp = soil_temp
                    st.session_state.sensor_analyzed = True
                    st.rerun()
                    
    with col_results:
        st.markdown(f'<p class="section-header"><span class="material-symbols-outlined">analytics</span> {t["sens_report"]}</p>', unsafe_allow_html=True)
        
        if not st.session_state.sensor_analyzed:
            st.markdown(f"""
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 350px; border: 2px dashed var(--border-color); border-radius: 16px; padding: 20px; text-align: center;">
                    <span class="material-symbols-outlined" style="font-size: 3.5rem; color: var(--text-secondary); margin-bottom: 12px;">device_thermostat</span>
                    <h3 style="margin: 0; color: var(--text-primary); font-family: 'Space Grotesk', sans-serif;">{t["sens_wait"]}</h3>
                    <p style="margin: 8px 0 0 0; color: var(--text-secondary); font-size: 0.9rem; max-width: 380px;">{t['sens_wait_desc']}</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
        else:
            # 1. Grab inputs from session state
            moisture = st.session_state.sensor_moisture
            ph = st.session_state.sensor_ph
            temp = st.session_state.sensor_temp
            humidity = st.session_state.sensor_humidity
            light = st.session_state.sensor_light
            soil_temp = st.session_state.sensor_soil_temp
            
            # 2. Perform Calculations Exactly as requested:
            # - Soil EC
            soil_ec = (moisture * 0.01) + (ph * 0.5)
            # - Nitrogen
            nitrogen = moisture * 0.5
            # - Phosphorus
            phosphorus = (7 - abs(7 - ph)) * 10
            # - Potassium
            potassium = moisture * 0.4
            # - Urea
            urea = nitrogen * 2.17
            
            # Evaluate individual nutrient status levels & colors
            # (value, optimal_range, moderate_range)
            ec_status, ec_color, ec_css, ec_emoji = evaluate_status(soil_ec, (1.0, 3.5), (0.5, 4.5))
            n_status, n_color, n_css, n_emoji = evaluate_status(nitrogen, (15.0, 30.0), (10.0, 40.0))
            p_status, p_color, p_css, p_emoji = evaluate_status(phosphorus, (45.0, 70.0), (25.0, 45.0))
            k_status, k_color, k_css, k_emoji = evaluate_status(potassium, (12.0, 25.0), (8.0, 35.0))
            urea_status, urea_color, urea_css, urea_emoji = evaluate_status(urea, (32.55, 65.1), (21.7, 86.8))
            
            # Calculate health score & overall status
            score, health_status = calculate_field_health(moisture, ph, soil_ec, nitrogen, phosphorus, potassium)
            
            # Generate AI Interpretation & Recommendations
            interpretation = generate_ai_interpretation(moisture, ph, temp, soil_ec, nitrogen, phosphorus, potassium, urea)
            recommendations = get_recommendations(moisture, ph, temp, soil_ec, nitrogen, phosphorus, potassium, soil_temp)
            
            # Predict Optimal Crop Using Trained ML Model
            X_input = pd.DataFrame([[
                ph, soil_ec, phosphorus, potassium, urea, moisture, temp
            ]], columns=["pH", "Soil EC", "Phosphorus", "Potassium", "Urea", "Moisture", "Temperature"])
            
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "training.py", "plant_model.pkl")
            encoder_path = os.path.join(base_dir, "training.py", "plant_encoder.pkl")
            
            try:
                model = joblib.load(model_path)
                encoder = joblib.load(encoder_path)
                pred_encoded = model.predict(X_input)
                predicted_crop = encoder.inverse_transform(pred_encoded)[0]
            except Exception as e:
                predicted_crop = "Unknown (Model Error)"
            
            # ── SECTION 4 & 1 layout (Gauge alongside Sensor Inputs Summary)
            top_c1, top_c2 = st.columns([1, 1.3], gap="medium")
            with top_c1:
                # Render SVG circular gauge
                st.markdown(draw_circular_gauge(score, health_status), unsafe_allow_html=True)
                
                # Add Predicted Crop Card
                st.markdown(f"""
                <div class="agri-card" style="margin-top: 15px; padding: 15px; text-align: center; border: 1px solid var(--primary-green); background: rgba(67, 160, 71, 0.05);">
                    <div style="font-size: 0.85rem; color: var(--text-secondary); text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">{t["sens_ai_rec"]}</div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: var(--primary-green); margin-top: 5px;">🌱 {predicted_crop}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with top_c2:
                # Sensor Summary Card
                st.markdown(f"""
                <div class="agri-card" style="padding: 15px; height: 100%;">
                    <h4 style="margin: 0 0 12px 0; color: var(--text-primary); font-family:'Space Grotesk',sans-serif; font-size:1.1rem; font-weight:700;"><span class="material-symbols-outlined" style="font-size:1.3rem;">input</span> {t["sens_raw"]}</h4>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px 15px;">
                        <div>
                            <div style="font-size:0.75rem; color:var(--text-secondary); text-transform:uppercase;">Soil Moisture</div>
                            <div style="font-size:1.1rem; font-weight:700; color:var(--text-primary);">{moisture:.1f} %</div>
                        </div>
                        <div>
                            <div style="font-size:0.75rem; color:var(--text-secondary); text-transform:uppercase;">Soil pH</div>
                            <div style="font-size:1.1rem; font-weight:700; color:var(--text-primary);">{ph:.2f}</div>
                        </div>
                        <div>
                            <div style="font-size:0.75rem; color:var(--text-secondary); text-transform:uppercase;">Air Temperature</div>
                            <div style="font-size:1.1rem; font-weight:700; color:var(--text-primary);">{temp:.1f} °C</div>
                        </div>
                        <div>
                            <div style="font-size:0.75rem; color:var(--text-secondary); text-transform:uppercase;">Air Humidity</div>
                            <div style="font-size:1.1rem; font-weight:700; color:var(--text-primary);">{humidity:.1f} %</div>
                        </div>
                        <div>
                            <div style="font-size:0.75rem; color:var(--text-secondary); text-transform:uppercase;">Light Intensity</div>
                            <div style="font-size:1.1rem; font-weight:700; color:var(--text-primary);">{light:.0f} Lux</div>
                        </div>
                        <div>
                            <div style="font-size:0.75rem; color:var(--text-secondary); text-transform:uppercase;">Soil Temp</div>
                            <div style="font-size:1.1rem; font-weight:700; color:var(--text-primary);">{soil_temp:.1f} °C</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<div style='margin-bottom:15px;'></div>", unsafe_allow_html=True)
            
            # ── SECTION 2: Calculated Nutrient Analysis
            st.markdown(f'<h4 style="margin: 10px 0 10px 0; color: var(--text-primary); font-family:\'Space Grotesk\',sans-serif; font-size:1.1rem; font-weight:700;"><span class="material-symbols-outlined">science</span> {t["sens_derived"]}</h4>', unsafe_allow_html=True)
            
            nutr_cols = st.columns(5)
            nutr_list = [
                ("Soil EC", f"{soil_ec:.2f} dS/m", ec_status, ec_css, ec_emoji),
                ("Nitrogen", f"{nitrogen:.1f} mg/kg", n_status, n_css, n_emoji),
                ("Phosphorus", f"{phosphorus:.1f} mg/kg", p_status, p_css, p_emoji),
                ("Potassium", f"{potassium:.1f} mg/kg", k_status, k_css, k_emoji),
                ("Urea", f"{urea:.1f} mg/kg", urea_status, urea_css, urea_emoji),
            ]
            for col, (name, val_str, status_lbl, status_color, emoji) in zip(nutr_cols, nutr_list):
                with col:
                    st.markdown(f"""
                    <div class="agri-card" style="padding: 12px; text-align: center; border-color: {status_color};">
                        <div style="font-size: 0.72rem; color: var(--text-secondary); font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">{name}</div>
                        <div style="font-size: 1.15rem; font-weight: 800; color: var(--text-primary); margin: 6px 0;">{val_str}</div>
                        <div style="display: inline-block; font-size: 0.7rem; font-weight: 700; color: {status_color}; background: rgba(255,255,255,0.05); padding: 2px 8px; border-radius: 8px; border: 1px solid {status_color};">
                            {emoji} {status_lbl}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            st.markdown("<div style='margin-bottom:15px;'></div>", unsafe_allow_html=True)
            
            # ── SECTION 3 & 6: AI Interpretation & Risk Assessment
            interp_c, risk_c = st.columns([1.5, 1], gap="medium")
            
            with interp_c:
                st.markdown(f'<h4 style="margin: 0 0 10px 0; color: var(--text-primary); font-family:\'Space Grotesk\',sans-serif; font-size:1.1rem; font-weight:700;"><span class="material-symbols-outlined">psychology</span> {t["sens_ai_interp"]}</h4>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="agri-card" style="padding: 15px; font-size:0.9rem; line-height: 1.5; color: var(--text-primary); border-left: 4px solid var(--primary-green);">
                    {interpretation}
                </div>
                """, unsafe_allow_html=True)
                
            with risk_c:
                st.markdown(f'<h4 style="margin: 0 0 10px 0; color: var(--text-primary); font-family:\'Space Grotesk\',sans-serif; font-size:1.1rem; font-weight:700;"><span class="material-symbols-outlined">warning</span> {t["sens_risk"]}</h4>', unsafe_allow_html=True)
                
                # Dynamic Risk calculations
                n_def_risk = "High" if n_status == "Critical" else "Medium" if n_status == "Moderate" else "Low"
                water_stress_risk = "High" if moisture < 30 or moisture > 80 else "Medium" if moisture < 40 or moisture > 70 else "Low"
                root_health_risk = "High" if ph < 5.0 or ph > 9.0 or moisture > 80 else "Medium" if ph < 5.8 or ph > 7.8 or moisture > 70 else "Low"
                fertility_rating = "High" if score >= 80 else "Medium" if score >= 55 else "Low"
                
                def get_badge(level):
                    color = "var(--danger)" if level == "High" else "var(--warning)" if level == "Medium" else "var(--healthy)"
                    bg = "rgba(229, 57, 53, 0.08)" if level == "High" else "rgba(251, 140, 0, 0.08)" if level == "Medium" else "rgba(67, 160, 71, 0.08)"
                    return f'<span style="font-size: 0.72rem; font-weight: 700; color: {color}; background: {bg}; border: 1px solid {color}; padding: 2px 8px; border-radius: 8px; float: right;">{level}</span>'
                
                st.markdown(f"""
                <div class="agri-card" style="padding: 12px; font-size: 0.85rem; line-height: 1.8;">
                    <div style="margin-bottom: 6px;">
                        <span style="color:var(--text-secondary);">Nutrient Deficiency Risk:</span>
                        {get_badge(n_def_risk)}
                    </div>
                    <div style="margin-bottom: 6px;">
                        <span style="color:var(--text-secondary);">Water Stress Risk:</span>
                        {get_badge(water_stress_risk)}
                    </div>
                    <div style="margin-bottom: 6px;">
                        <span style="color:var(--text-secondary);">Root Health Risk:</span>
                        {get_badge(root_health_risk)}
                    </div>
                    <div>
                        <span style="color:var(--text-secondary);">Soil Fertility:</span>
                        {get_badge(fertility_rating)}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<div style='margin-bottom:15px;'></div>", unsafe_allow_html=True)
            
            # ── SECTION 5: Recommendations Panel
            st.markdown(f'<h4 style="margin: 0 0 10px 0; color: var(--text-primary); font-family:\'Space Grotesk\',sans-serif; font-size:1.1rem; font-weight:700;"><span class="material-symbols-outlined">assignment_turned_in</span> {t["sens_directives"]}</h4>', unsafe_allow_html=True)
            rec_html = ""
            for r in recommendations:
                rec_html += f"<li style='margin-bottom: 8px; font-size: 0.9rem; color: var(--text-primary); list-style-type: none; border-bottom: 1px solid var(--border-color); padding-bottom: 6px;'>{r}</li>"
            st.markdown(f"""
            <div class="agri-card" style="padding: 15px 20px;">
                <ul style="padding: 0; margin: 0;">
                    {rec_html}
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<div style='margin-bottom:15px;'></div>", unsafe_allow_html=True)
            
            # ── SECTION 7: Export Actions
            st.markdown(f'<h4 style="margin: 0 0 10px 0; color: var(--text-primary); font-family:\'Space Grotesk\',sans-serif; font-size:1.1rem; font-weight:700;"><span class="material-symbols-outlined">download</span> {t["sens_export"]}</h4>', unsafe_allow_html=True)
            
            exp_c1, exp_c2 = st.columns(2)
            
            # Prepare CSV
            csv_df = pd.DataFrame({
                "Telemetry/Derived Parameter": ["Soil Moisture", "Soil pH", "Air Temperature", "Air Humidity", "Light Intensity", "Soil Temperature", "Soil EC (Calculated)", "Nitrogen (Calculated)", "Phosphorus (Calculated)", "Potassium (Calculated)", "Urea (Calculated)", "Overall Health Score", "AI Predicted Crop"],
                "Value": [moisture, ph, temp, humidity, light, soil_temp, soil_ec, nitrogen, phosphorus, potassium, urea, score, predicted_crop],
                "Unit": ["%", "pH", "°C", "%", "Lux", "°C", "dS/m", "mg/kg", "mg/kg", "mg/kg", "mg/kg", "%", "N/A"],
                "Status": ["Input", "Input", "Input", "Input", "Input", "Input", ec_status, n_status, p_status, k_status, urea_status, health_status, "Prediction"]
            })
            csv_bytes = csv_df.to_csv(index=False).encode("utf-8")
            
            # Prepare PDF
            pdf_bytes = generate_pdf_report(
                moisture, ph, temp, humidity, light, soil_temp,
                soil_ec, nitrogen, phosphorus, potassium, urea,
                score, health_status, interpretation, recommendations, predicted_crop
            )
            
            with exp_c1:
                st.download_button(
                    label=t["sens_down_csv"],
                    data=csv_bytes,
                    file_name="sensor_analytics_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with exp_c2:
                st.download_button(
                    label=t["sens_down_pdf"],
                    data=pdf_bytes,
                    file_name="sensor_analytics_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
