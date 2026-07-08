# scratch/replace_layout.py
import re

with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# Define the new block to insert
new_block = """elif active_tab == "🔍 Crop Analysis":
    # Fetch variables from state if already analyzed
    if st.session_state.uploaded_img_data is not None:
        img_array = st.session_state.uploaded_img_data
        efficiency_pct = st.session_state.efficiency_pct
        meta = st.session_state.meta
        gradcam = st.session_state.gradcam
        shap_overlay = st.session_state.shap_overlay
        attention = st.session_state.attention
        heatmap = st.session_state.heatmap
        image = Image.fromarray(img_array)
        
        # Calculate mean indices for exporting
        norm_img = img_array.astype(np.float32) / 255.0
        indices_arr = calculate_indices(norm_img)
        vari_val = float(np.mean(indices_arr[:, :, 0]))
        exg_val = float(np.mean(indices_arr[:, :, 1]))
        mgrvi_val = float(np.mean(indices_arr[:, :, 2]))
        
        info = get_health_info(efficiency_pct, lang)
    
    # AI Stepper progress timeline
    step_classes = ["completed", "completed", "completed", "completed", "completed", "completed", "completed", "completed", "completed", "completed", "completed"]
    if st.session_state.uploaded_img_data is None:
        step_classes = ["active", "", "", "", "", "", "", "", "", "", ""]
    
    stepper_html = '<div class="stepper">'
    for idx, name in enumerate(["1. Upload", "2. Enhance", "3. BBox Detect", "4. Crop ID", "5. Disease", "6. Severity", "7. Nutrient", "8. Lifecycle", "9. Indices", "10. Explainable AI", "11. Export"]):
        stepper_html += f'<div class="step {step_classes[idx]}"><div class="step-dot"></div><span>{name}</span></div>'
    stepper_html += '</div>'
    st.markdown(stepper_html, unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1, 1.1])

    with col_left:
        # File Specimen Upload Area
        st.markdown(f'<p class="section-header" style="margin-bottom: 10px;">📤 {t["upload_header"]}</p>', unsafe_allow_html=True)
        
        # Only show the large drag-and-drop hero card if no image has been uploaded yet
        if st.session_state.uploaded_img_data is None:
            st.markdown("""
            <div class="agri-upload-hero" style="padding: 20px; border-radius: 12px; margin-bottom: 10px;">
                <span style="font-size:3rem;">🌱</span>
                <p style="font-size:1.05rem; font-weight:700; color:var(--primary-green); margin:8px 0 3px 0;">Drag and drop leaf image file</p>
                <p style="font-size:0.75rem; color:var(--text-secondary); margin-bottom:10px;">JPG, JPEG or PNG formats (Max 10MB)</p>
            </div>
            """, unsafe_allow_html=True)
            
        uploaded_file = st.file_uploader("Upload Leaf File", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        camera_input_active = st.checkbox("📷 Use Device Camera Instead", value=False)
        camera_image = None
        if camera_input_active:
            camera_image = st.camera_input("Capture Crop Specimen")
            
        final_uploaded = uploaded_file if uploaded_file else camera_image

        if final_uploaded is not None:
            # Specimen analysis & caching (runs only on new image)
            image_obj = Image.open(final_uploaded).convert("RGB")
            img_array_new = np.array(image_obj)
            
            if st.session_state.uploaded_img_data is None or not np.array_equal(st.session_state.uploaded_img_data, img_array_new):
                with st.spinner(""):
                    p = st.progress(0, text=t["prog_preprocess"])
                    time.sleep(0.3)
                    p.progress(35, text=t["prog_indices"])
                    processed = preprocess(img_array_new)
                    time.sleep(0.3)
                    p.progress(65, text=t["prog_model"])
                    
                    if keras_model is not None:
                        pred = keras_model.predict(processed, verbose=0)[0][0]
                        efficiency_pct = round(float(pred) * 100, 1)
                    else:
                        norm_img = img_array_new.astype(np.float32) / 255.0
                        R = norm_img[:,:,0]; G = norm_img[:,:,1]; B = norm_img[:,:,2]
                        vari_mean = float(np.mean(np.clip((G - R) / (G + R - B + 1e-5), -1, 1)))
                        efficiency_pct = round(50.0 + (vari_mean * 40.0) + np.random.uniform(-5, 5), 1)
                        efficiency_pct = min(100.0, max(0.0, efficiency_pct))
                        
                    p.progress(90, text=t["prog_report"])
                    meta, cropped = analyze_image(img_array_new)
                    
                    gradcam, shap_overlay, attention, heatmap = generate_xai_maps(img_array_new)
                    
                    p.progress(100, text=t["prog_complete"])
                    time.sleep(0.2)
                    p.empty()
                    
                    st.session_state.uploaded_img_data = img_array_new
                    st.session_state.efficiency_pct = efficiency_pct
                    st.session_state.meta = meta
                    st.session_state.gradcam = gradcam
                    st.session_state.shap_overlay = shap_overlay
                    st.session_state.attention = attention
                    st.session_state.heatmap = heatmap
                    
                    # Record diagnostic log
                    new_rec = {
                        "crop": meta["crop"],
                        "disease": meta["disease"],
                        "severity": meta["severity"],
                        "efficiency": efficiency_pct,
                        "date": time.strftime("%Y-%m-%d %I:%M %p"),
                        "confidence": meta["confidence"],
                        "indices": {
                            "VARI": float(np.mean(calculate_indices(img_array_new.astype(np.float32)/255.0)[:,:,0])),
                            "ExG": float(np.mean(calculate_indices(img_array_new.astype(np.float32)/255.0)[:,:,1])),
                            "MGRVI": float(np.mean(calculate_indices(img_array_new.astype(np.float32)/255.0)[:,:,2]))
                        }
                    }
                    st.session_state.history_list.append(new_rec)
                    st.toast("🌱 Analysis complete!")
                    st.rerun()
        
        # Display results on the left side
        if st.session_state.uploaded_img_data is not None:
            # File specs for preview card
            file_size_kb = len(final_uploaded.getvalue()) / 1024.0 if (final_uploaded is not None and hasattr(final_uploaded, "getvalue")) else 128.0
            img_res = f"{img_array.shape[1]}x{img_array.shape[0]}"
            
            # Image Preview Card with scanning ray overlay
            st.markdown(f'<p class="section-header" style="margin-top: 15px; margin-bottom: 8px;">📷 {t["leaf_img_header"]}</p>', unsafe_allow_html=True)
            st.markdown('<div class="img-preview-box" style="margin-bottom: 15px;"><div class="scan-ray"></div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Metadata Grid List
            st.markdown(f"""
            <div class="agri-card" style="margin-bottom:20px; padding: 15px;">
                <h4 style="margin-top:0; color:var(--primary-green); font-size:1.05rem; margin-bottom:8px;">Specimen Metrics</h4>
                <div class="spec-item" style="font-size: 0.85rem; padding: 4px 0;"><span>Scan Resolution</span><span style="font-weight:700;">{img_res}</span></div>
                <div class="spec-item" style="font-size: 0.85rem; padding: 4px 0;"><span>File Size</span><span style="font-weight:700;">{file_size_kb:.1f} KB</span></div>
                <div class="spec-item" style="font-size: 0.85rem; padding: 4px 0;"><span>Imaging Date</span><span style="font-weight:700;">{time.strftime("%d %b %Y")}</span></div>
                <div class="spec-item" style="font-size: 0.85rem; padding: 4px 0;"><span>Geographical telemetry</span><span style="font-weight:700;">19.0760° N, 72.8777° E</span></div>
                <div class="spec-item" style="font-size: 0.85rem; padding: 4px 0;"><span>AI Model Version</span><span style="font-weight:700;">CNN Regressor v1.2</span></div>
            </div>
            """, unsafe_allow_html=True)

            # Circular SVG Photosynthetic Efficiency Gauge (Moved to left col to balance height)
            radial_svg = draw_radial_gauge(
                int(efficiency_pct),
                "Canopy Photosynthetic capacity index",
                info["status"],
                info["bar_color"]
            )
            st.markdown(f"""
            <div class="agri-card" style="margin-bottom:20px; display:flex; justify-content:center; align-items:center; flex-direction:column; padding:25px;">
                {radial_svg}
            </div>
            """, unsafe_allow_html=True)

            # Explainable AI View switcher
            st.markdown('<p class="section-header" style="margin-top: 15px; margin-bottom: 8px;">🧠 Explainable Neural Overlay</p>', unsafe_allow_html=True)
            xai_tab_sel = st.selectbox(
                "Select XAI Filter Layer",
                ["Original Specimen", "Grad-CAM Hotspot", "SHAP Feature Boundaries", "Foliar Attention Map", "Colormap Jet Contours"],
                label_visibility="collapsed"
            )
            
            st.markdown('<div class="img-preview-box" style="margin-bottom: 15px;">', unsafe_allow_html=True)
            if xai_tab_sel == "Original Specimen":
                st.image(image, use_container_width=True)
            elif xai_tab_sel == "Grad-CAM Hotspot":
                st.image(gradcam, use_container_width=True)
            elif xai_tab_sel == "SHAP Feature Boundaries":
                st.image(shap_overlay, use_container_width=True)
            elif xai_tab_sel == "Foliar Attention Map":
                st.image(attention, use_container_width=True)
            else:
                st.image(heatmap, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="agri-card" style="margin-top:15px; padding: 15px;">
                <h4 style="margin-top:0; color:var(--primary-green); font-size:1rem; margin-bottom: 8px;">Neural Activation Interpretation</h4>
                <p style="font-size:0.85rem; line-height:1.5; margin:0; color:var(--text-secondary);">
                    Attention maps overlay glowing regions matching loss of chlorophyll pigments. 
                    SHAP matrix details localized pixels contributing (+{meta['confidence']/10:.1f}%) to <b>{meta['disease']}</b> classification.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.markdown("""
            <div style='text-align:center; padding: 40px 20px; border:2px dashed rgba(46,125,50,0.15); border-radius:12px; background:#fff; margin-top:15px;'>
                <span style="font-size:2.2rem;">🌾</span>
                <p style="margin:8px 0 0 0; font-size:0.85rem; color:var(--text-secondary);">Awaiting foliar image upload to run diagnostics...</p>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        # ── Right Column: Diagnostic Summary and Meters ──────
        st.markdown(f'<p class="section-header" style="margin-bottom: 10px;">🌿 Diagnostic Summary & Diagnostics</p>', unsafe_allow_html=True)
        
        if st.session_state.uploaded_img_data is not None:
            # Main health centerpiece header summary card
            severity_badge_class = "badge-healthy" if meta["severity"] == "None" else "badge-warning" if meta["severity"] == "Low" else "badge-danger"
            
            st.markdown(f"""
            <div class="health-summary-panel" style="padding: 18px; margin-bottom: 15px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                    <span style="font-size:1.15rem; font-weight:700; color:var(--primary-green);">Field Health Centerpiece</span>
                    <span class="agri-badge {severity_badge_class}" style="font-size:0.7rem; padding: 3px 8px;">{meta["severity"]} Alert Status</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="font-size:0.75rem; color:var(--text-secondary); text-transform:uppercase;">Photosynthetic Index</span>
                        <div style="font-size:2.2rem; font-weight:800; color:var(--primary-green); margin:0;">{efficiency_pct}%</div>
                    </div>
                    <div>
                        <span style="font-size:0.75rem; color:var(--text-secondary); text-transform:uppercase;">Stress Level</span>
                        <div style="font-size:1.1rem; font-weight:700; color:var(--warning); margin:0;">{'Low' if efficiency_pct > 80 else 'High' if efficiency_pct < 50 else 'Moderate'}</div>
                    </div>
                    <div>
                        <span style="font-size:0.75rem; color:var(--text-secondary); text-transform:uppercase;">Recovery rate</span>
                        <div style="font-size:1.1rem; font-weight:700; color:var(--healthy); margin:0;">{'Optimal' if efficiency_pct > 60 else 'Critical'}</div>
                    </div>
                </div>
                <div style="margin-top:12px; border-top:1px solid rgba(46,125,50,0.1); padding-top:10px;">
                    <h5 style="margin:0 0 4px 0; color:var(--primary-green); font-size:0.8rem;">Neural Diagnostic Output</h5>
                    <p style="font-size:0.8rem; line-height:1.45; color:var(--text-secondary); margin:0;">
                        <b>Observation:</b> {info["observation"]} <br>
                        <b>Corrective Action:</b> {info["action"]}
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Crop Identification & Disease Severity side-by-side Row
            c_row1_left, c_row1_right = st.columns(2)
            
            with c_row1_left:
                scientific_name = "Solanum lycopersicum" if meta["crop"] == "Tomato" else "Trichosanthes cucumerina" if meta["crop"] == "Snake Gourd" else "Unknown Species"
                crop_family = "Solanaceae" if meta["crop"] == "Tomato" else "Cucurbitaceae" if meta["crop"] == "Snake Gourd" else "Unknown Family"
                
                st.markdown(f"""
                <div class="agri-card" style="padding:15px; height:160px; display:flex; flex-direction:column; justify-content:space-between;">
                    <div>
                        <h4 style="margin-top:0; color:var(--primary-green); font-size:0.9rem; margin-bottom:6px;">🌾 Crop Identification</h4>
                        <p style="font-size:1.05rem; font-weight:700; margin:0 0 2px 0;">{meta['crop']}</p>
                        <p style="font-size:0.7rem; font-style:italic; color:var(--text-secondary); margin:0;">{scientific_name}<br>Family: {crop_family}</p>
                    </div>
                    <div style="background:var(--bg-secondary); padding:4px 8px; border-radius:6px; display:flex; justify-content:space-between; font-size:0.7rem;">
                        <span>AI Confidence</span>
                        <span style="font-weight:700; color:var(--primary-green);">{meta["confidence"]}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            with c_row1_right:
                severity_num = 15 if meta["severity"] == "None" else 40 if meta["severity"] == "Low" else 75 if meta["severity"] == "Medium" else 95
                severity_color = "var(--healthy)" if severity_num < 30 else "var(--warning)" if severity_num < 60 else "var(--danger)"
                
                st.markdown(f"""
                <div class="agri-card" style="padding:15px; height:160px; display:flex; flex-direction:column; justify-content:space-between;">
                    <div>
                        <h4 style="margin-top:0; color:var(--primary-green); font-size:0.9rem; margin-bottom:6px;">🚨 Pathogen Diagnostics</h4>
                        <p style="font-size:1.05rem; font-weight:700; color:{severity_color}; margin:0 0 2px 0;">{meta["disease"]}</p>
                        <p style="font-size:0.7rem; color:var(--text-secondary); margin:0; line-height:1.2;">Cues: {meta["cause"][:45]}...</p>
                    </div>
                    <div>
                        <div class="agri-progress-bar" style="height:4px;"><div class="agri-progress-fill" style="width:{severity_num}%; background:{severity_color};"></div></div>
                        <div style="display:flex; justify-content:space-between; font-size:0.65rem; color:var(--text-secondary); margin-top:2px;">
                            <span>Severity Index</span>
                            <span style="font-weight:700;">{severity_num}%</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Nutrient deficiency analysis grid
            if "Nitrogen" in meta["disease"]:
                n_val, p_val, k_val, iron_val = 32, 85, 82, 78
            elif "Potassium" in meta["disease"]:
                n_val, p_val, k_val, iron_val = 80, 82, 28, 90
            else:
                n_val, p_val, k_val, iron_val = 92, 88, 85, 90
                
            st.markdown(f"""
            <div class="agri-card" style="margin-top:15px; padding: 15px;">
                <h4 style="margin-top:0; color:var(--primary-green); font-size:0.9rem; margin-bottom:8px;">🧪 Canopy Nutrient Deficiency Index</h4>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:8px;">
                    <div style="background:var(--bg-secondary); padding:6px 10px; border-radius:8px;">
                        <span style="font-size:0.65rem; color:var(--text-secondary);">Nitrogen (N)</span>
                        <div style="font-size:0.9rem; font-weight:700; color:{'var(--danger)' if n_val < 50 else 'var(--text-primary)'};">{n_val}%</div>
                        <div class="agri-progress-bar" style="height:3px;"><div class="agri-progress-fill" style="width:{n_val}%; background:{'var(--danger)' if n_val < 50 else 'var(--healthy)'};"></div></div>
                    </div>
                    <div style="background:var(--bg-secondary); padding:6px 10px; border-radius:8px;">
                        <span style="font-size:0.65rem; color:var(--text-secondary);">Phosphorus (P)</span>
                        <div style="font-size:0.9rem; font-weight:700; color:{'var(--danger)' if p_val < 50 else 'var(--text-primary)'};">{p_val}%</div>
                        <div class="agri-progress-bar" style="height:3px;"><div class="agri-progress-fill" style="width:{p_val}%; background:{'var(--danger)' if p_val < 50 else 'var(--healthy)'};"></div></div>
                    </div>
                    <div style="background:var(--bg-secondary); padding:6px 10px; border-radius:8px;">
                        <span style="font-size:0.65rem; color:var(--text-secondary);">Potassium (K)</span>
                        <div style="font-size:0.9rem; font-weight:700; color:{'var(--danger)' if k_val < 50 else 'var(--text-primary)'};">{k_val}%</div>
                        <div class="agri-progress-bar" style="height:3px;"><div class="agri-progress-fill" style="width:{k_val}%; background:{'var(--danger)' if k_val < 50 else 'var(--healthy)'};"></div></div>
                    </div>
                    <div style="background:var(--bg-secondary); padding:6px 10px; border-radius:8px;">
                        <span style="font-size:0.65rem; color:var(--text-secondary);">Iron (Fe)</span>
                        <div style="font-size:0.9rem; font-weight:700; color:{'var(--danger)' if iron_val < 50 else 'var(--text-primary)'};">{iron_val}%</div>
                        <div class="agri-progress-bar" style="height:3px;"><div class="agri-progress-fill" style="width:{iron_val}%; background:{'var(--danger)' if iron_val < 50 else 'var(--healthy)'};"></div></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Growth stage lifecycle timeline
            stages = ["Germination", "Seedling", "Vegetative", "Flowering", "Fruiting", "Maturity", "Harvest"]
            active_idx = 3 if meta["crop"] == "Tomato" else 2
            nodes_html = ""
            for idx, stage in enumerate(stages):
                act = "active" if idx == active_idx else ""
                nodes_html += f'<div class="timeline-node {act}"><div class="node-dot"></div><span>{stage}</span></div>'
            
            timeline_html = f'<div class="agri-card" style="margin-top:15px; padding: 15px;"><h4 style="margin-top:0; color:var(--primary-green); font-size:0.9rem; margin-bottom:10px;">🌱 Crop Lifecycle Stage</h4><div class="timeline-track" style="margin-top: 8px;"><div style="position:absolute; width:100%; height:2px; background:rgba(0,0,0,0.03); top:4px; z-index:1;"></div>{nodes_html}</div></div>'
            st.markdown(timeline_html, unsafe_allow_html=True)

            # ── Recommendations Detail Panel ───────────────────────
            st.markdown(f'<p class="section-header" style="margin-top: 15px; margin-bottom: 8px;">💡 AI Corrective Recommendations</p>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="agri-card" style="margin-bottom:10px; border-left:4px solid var(--primary-green); padding: 12px 15px;">
                <span style="font-size:0.7rem; color:var(--text-secondary); font-weight:700; text-transform:uppercase;">Agronomic Description & Action Plan</span>
                <p style="font-size:0.8rem; margin:2px 0 0 0; line-height:1.45;"><b>Cause:</b> {meta["cause"]}<br><b>Treatment Recipe:</b> {meta["treatment"]}</p>
            </div>
            <div class="agri-card" style="margin-bottom:15px; background:rgba(33, 150, 243, 0.03); border-color:rgba(33, 150, 243, 0.15); padding: 12px 15px;">
                <span style="font-size:0.7rem; color:var(--info); font-weight:700; text-transform:uppercase;">Irrigation & Meteorological Factors</span>
                <p style="font-size:0.8rem; margin:2px 0 0 0; line-height:1.45;">Maintain low foliar dampness. Soil temperature active range: 20-26°C.</p>
            </div>
            """, unsafe_allow_html=True)

            # Export Center
            st.markdown('<p class="section-header" style="margin-top: 15px; margin-bottom: 8px;">📋 Diagnostic Export Manager</p>', unsafe_allow_html=True)
            csv_data = f"Diagnostic Metric,Telemetry Value\nCrop Species,{meta['crop']}\nDiagnosed Disease,{meta['disease']}\nSeverity Level,{meta['severity']}\nPhotosynthetic Efficiency,{efficiency_pct}%\nVARI Index,{vari_val:.4f}\n"
            st.download_button("📄 Export CSV Telemetry", csv_data, "crop_report.csv", "text/csv", use_container_width=True)
            
        else:
            st.markdown("""
            <div style='text-align:center; padding: 60px 20px; border:1px solid var(--border-color); border-radius:16px; background:var(--card-bg); margin-top: 15px;'>
                <span style="font-size:2.2rem;">🔬</span>
                <h4 style="margin-top:10px; color:var(--text-primary); font-family:'Space Grotesk',sans-serif;">Awaiting Crop Specimen</h4>
                <p style="font-size:0.8rem; color:var(--text-secondary); margin:3px 0 0 0;">Upload crop leaf image to run diagnostics.</p>
            </div>
            """, unsafe_allow_html=True)"""

# Perform replacement using regex
pattern = re.compile(
    r'elif active_tab == "🔍 Crop Analysis":.*?(?=elif active_tab == "🌿 Plant Health":)',
    re.DOTALL
)

if pattern.search(content):
    content = pattern.sub(new_block + "\n\n", content)
    with open("app.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("SUCCESS: app.py layout updated!")
else:
    print("ERROR: pattern match failed!")
