"""
DeepShield-X: Integrated Multimodal Deepfake Detection & Digital Identity Verification Engine
Team Capstone Project (SIMATS - B Slot)
Integrated Pipeline: Agent Search + Knowledge Reasoning + ML Classification + File Uploader
"""

import random
import time
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="DeepShield-X Dashboard",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ DeepShield-X: Multimodal Deepfake Detection & Identity Verification")
st.markdown(
    "*Integrated Capstone Platform combining Agent Search, First-Order Logic Reasoning, "
    "Machine Learning Classification, and Scalability Testing.*"
)

# ==========================================
# MODULE 1: AGENT ARCHITECTURE & SEARCH
# ==========================================
class MediaVerificationAgent:
    def __init__(self):
        self.states = [
            "Input_Ingestion",
            "Metadata_Check",
            "Feature_Extraction",
            "Heuristic_Evaluation",
            "Final_Verdict",
        ]
        self.actions = ["scan_pixels", "verify_sync", "check_certificate", "evaluate_risk"]

    def heuristic_search(self, input_size, search_method="A*"):
        """Simulates A* or Greedy Best-First Search for verification pipeline routing."""
        start_time = time.time()
        expanded_nodes = input_size * (4 if search_method == "A*" else 3)
        path_cost = round(input_size * 1.5 + (1.2 if search_method == "A*" else 0.8), 2)

        path = ["Input_Ingestion"]
        if input_size >= 10:
            path.append("Metadata_Check")
        path.extend(["Feature_Extraction", "Heuristic_Evaluation", "Final_Verdict"])

        execution_time = round((time.time() - start_time) + (input_size * 0.005), 4)
        memory_used = round(12.4 + (input_size * 0.2), 2)

        return {
            "method": search_method,
            "input_size": input_size,
            "expanded_nodes": expanded_nodes,
            "path_cost": path_cost,
            "execution_time": execution_time,
            "memory_mb": memory_used,
            "path": path,
        }


# ==========================================
# MODULE 2: KNOWLEDGE BASE & LOGICAL REASONING
# ==========================================
class DeepShieldKnowledgeBase:
    def __init__(self):
        self.facts = {
            "has_pixel_anomaly",
            "has_blinking_irregularity",
            "has_audio_visual_sync_lag",
            "has_metadata_mismatch",
            "has_high_compression_artifact",
            "has_unnatural_skin_texture",
            "has_warp_border_artifact",
            "has_inconsistent_lighting",
            "has_robotic_voice_tone",
            "has_background_noise_mismatch",
            "has_zero_eye_blink_rate",
            "has_frequency_spectrum_anomaly",
            "has_known_synthetic_signature",
            "has_unusual_head_pose_jitter",
            "has_mismatched_dental_structure",
            "has_low_resolution_face_swap",
            "has_temporal_flicker",
            "has_inconsistent_shadows",
            "has_biometric_hash_collision",
            "has_revoked_digital_certificate",
        }

        self.rules = [
            {"id": "R1", "if": ["has_pixel_anomaly", "has_unnatural_skin_texture"], "then": "potential_image_deepfake"},
            {"id": "R2", "if": ["has_warp_border_artifact", "has_low_resolution_face_swap"], "then": "confirmed_face_swap"},
            {"id": "R3", "if": ["has_audio_visual_sync_lag", "has_robotic_voice_tone"], "then": "potential_audio_deepfake"},
            {"id": "R4", "if": ["has_zero_eye_blink_rate", "has_unusual_head_pose_jitter"], "then": "potential_video_deepfake"},
            {"id": "R5", "if": ["confirmed_face_swap", "potential_video_deepfake"], "then": "high_confidence_deepfake"},
            {"id": "R6", "if": ["has_metadata_mismatch", "has_known_synthetic_signature"], "then": "verified_synthetic_origin"},
            {"id": "R7", "if": ["has_revoked_digital_certificate"], "then": "invalid_digital_identity"},
            {"id": "R8", "if": ["potential_image_deepfake", "has_inconsistent_lighting"], "then": "structured_visual_tampering"},
            {"id": "R9", "if": ["potential_audio_deepfake", "has_frequency_spectrum_anomaly"], "then": "synthetic_voice_clone"},
            {"id": "R10", "if": ["high_confidence_deepfake", "verified_synthetic_origin"], "then": "reject_identity_verification"},
        ]

    def forward_chain(self, evidence):
        inferred = set(evidence)
        trace = []
        added = True
        while added:
            added = False
            for rule in self.rules:
                if rule["then"] not in inferred:
                    if all(cond in inferred for cond in rule["if"]):
                        inferred.add(rule["then"])
                        trace.append(f"Rule {rule['id']} triggered: {rule['if']} -> {rule['then']}")
                        added = True
        return inferred, trace


# ==========================================
# MODULE 3: MACHINE LEARNING & SCALABILITY
# ==========================================
class DeepShieldClassifier:
    def __init__(self):
        # Features: [anomaly_ratio, derived_risk, size_norm, type_code, high_sev_flag]
        X_train = np.array([
            [0.00, 0.0, 0.2, 0.0, 0.0],
            [0.05, 0.0, 0.5, 0.0, 0.0],
            [0.10, 0.0, 0.3, 0.33, 0.0],
            [0.15, 0.1, 0.7, 0.66, 0.0],
            [0.35, 0.3, 0.4, 0.0, 1.0],
            [0.50, 0.5, 0.8, 0.33, 1.0],
            [0.70, 0.7, 0.2, 0.66, 1.0],
            [0.90, 0.9, 0.9, 1.0, 1.0],
        ])
        y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        self.model = DecisionTreeClassifier(random_state=42, max_depth=4)
        self.model.fit(X_train, y_train)

    def predict_sample(self, features):
        pred = int(self.model.predict([features])[0])
        proba = self.model.predict_proba([features])[0]
        return pred, float(max(proba))

    def evaluate_workload(self, scale_factor):
        rng = np.random.default_rng(42 + scale_factor)
        test_size = 50 * scale_factor
        X_test = rng.random((test_size, 5))
        y_test = rng.integers(0, 2, size=test_size)

        start_time = time.time()
        preds = self.model.predict(X_test)
        duration = round(time.time() - start_time, 4)

        acc = round(accuracy_score(y_test, preds), 3)
        f1 = round(f1_score(y_test, preds, zero_division=0), 3)

        return {
            "workload": f"{scale_factor}x",
            "samples": test_size,
            "execution_time_sec": duration,
            "accuracy": acc,
            "f1_score": f1,
        }


def extract_image_stats(uploaded_file):
    """Extracts basic non-random image properties if Pillow is available."""
    if not PIL_AVAILABLE or uploaded_file is None or not uploaded_file.type.startswith("image"):
        return None
    try:
        raw_bytes = uploaded_file.getvalue()
        with Image.open(BytesIO(raw_bytes)) as img:
            width, height = img.size
            mode = img.mode
            arr = np.asarray(img.convert("RGB"), dtype=np.float32)
            mean_luma = float(np.mean(arr))
            std_luma = float(np.std(arr))
        return {
            "resolution": f"{width}x{height}",
            "mode": mode,
            "mean_brightness": round(mean_luma, 1),
            "pixel_contrast_std": round(std_luma, 1),
        }
    except Exception:
        return None


# --- INITIALIZE SYSTEM COMPONENTS ---
agent = MediaVerificationAgent()
kb = DeepShieldKnowledgeBase()
classifier = DeepShieldClassifier()

# --- STREAMLIT UI LAYOUT ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Pipeline Execution",
    "🧠 Knowledge Reasoning",
    "🤖 ML & Scalability",
    "📊 Team & Evaluation Specs",
])

with tab1:
    st.subheader("End-to-End Media Ingestion & Verification")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 1. Upload Media Asset")
        media_type = st.selectbox(
            "Select Media Type",
            ["Image", "Video", "Audio Stream", "Digital Identity Document"],
        )

        uploaded_file = st.file_uploader(
            f"Upload your {media_type} file",
            type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "wav", "mp3", "pdf"],
        )

        st.markdown("#### 2. Configure Pipeline Parameters")
        input_size = st.slider("Search Input Size (Batch Items)", min_value=1, max_value=50, value=10)
        search_algo = st.radio("Search Algorithm", ["A*", "Greedy Best-First"])

        selected_evidence = st.multiselect(
            "Select Detected Evidence / Anomalies",
            sorted(kb.facts),
            default=[],
        )

    with col2:
        st.markdown("#### 3. Asset Preview & Execution Results")
        img_stats = None

        if uploaded_file is not None:
            file_details = {
                "FileName": uploaded_file.name,
                "FileType": uploaded_file.type,
                "FileSize": f"{uploaded_file.size / 1024:.2f} KB",
            }
            st.write("📁 **File Uploaded Successfully:**", file_details)

            if uploaded_file.type.startswith("image"):
                st.image(uploaded_file, caption="Uploaded Image Preview", use_container_width=True)
                img_stats = extract_image_stats(uploaded_file)
                if img_stats:
                    st.caption(
                        f"Resolution: {img_stats['resolution']} | Mode: {img_stats['mode']} | "
                        f"Mean Brightness: {img_stats['mean_brightness']} | Std Dev: {img_stats['pixel_contrast_std']}"
                    )
            elif uploaded_file.type.startswith("video"):
                st.video(uploaded_file)
            elif uploaded_file.type.startswith("audio"):
                st.audio(uploaded_file)
            else:
                st.info("📄 Document uploaded. Metadata and rule-based verification are ready.")
        else:
            st.info("ℹ️ Upload a media file on the left to preview it here, or run verification using selected anomalies.")

        if st.button("Run Full Pipeline Verification", type="primary"):
            search_res = agent.heuristic_search(input_size, search_algo)
            derived, trace = kb.forward_chain(selected_evidence)

            critical_conclusions = {
                "high_confidence_deepfake",
                "reject_identity_verification",
                "invalid_digital_identity",
                "confirmed_face_swap",
                "synthetic_voice_clone",
                "structured_visual_tampering",
            }
            triggered_critical = sorted(fact for fact in derived if fact in critical_conclusions)
            has_critical = len(triggered_critical) > 0

            file_size_kb = (uploaded_file.size / 1024.0) if uploaded_file is not None else 0.0
            size_norm = min(file_size_kb / 2048.0, 1.0)
            type_map = {
                "Image": 0.0,
                "Video": 0.33,
                "Audio Stream": 0.66,
                "Digital Identity Document": 1.0,
            }
            anomaly_ratio = len(selected_evidence) / max(len(kb.facts), 1)
            derived_risk = min(len(derived) / 10.0, 1.0)
            high_sev_flag = 1.0 if has_critical else 0.0

            features = [
                anomaly_ratio,
                derived_risk,
                size_norm,
                type_map.get(media_type, 0.0),
                high_sev_flag,
            ]
            pred_class, confidence = classifier.predict_sample(features)

            is_fake = has_critical or (pred_class == 1 and len(selected_evidence) >= 2)
            verdict = "🚨 DEEPFAKE DETECTED" if is_fake else "✅ AUTHENTIC ASSET"

            if is_fake:
                st.error(f"**Final Verdict:** {verdict}")
            else:
                st.success(f"**Final Verdict:** {verdict}")

            st.metric("Model Confidence Score", f"{confidence * 100:.2f}%")

            with st.expander("🧾 View Reasoning Summary"):
                st.write(f"**Selected Evidence ({len(selected_evidence)}):**", selected_evidence if selected_evidence else "None")
                st.write(f"**Inferred Conclusions ({len(derived)}):**", sorted(derived) if derived else "None")
                if triggered_critical:
                    st.write("**Critical Conclusions Triggered:**", triggered_critical)
                elif not selected_evidence:
                    st.write("**Reason:** No anomaly indicators selected and no critical tampering rules fired.")
                else:
                    st.write("**Reason:** Evidence did not trigger critical deepfake/tampering conclusions.")

            with st.expander("🔍 View Agent Search Metrics"):
                st.write(search_res)

            with st.expander("📜 View Logical Rule Inference Trace"):
                if trace:
                    for t in trace:
                        st.text(t)
                else:
                    st.text("No inference rules triggered from current evidence.")

with tab2:
    st.subheader("Module 2: Knowledge Base & Rule Verification")
    st.markdown(f"**Loaded Domain Facts:** `{len(kb.facts)}` (Required: >= 20)")
    st.markdown(f"**Loaded Logical Rules:** `{len(kb.rules)}` (Required: >= 10)")

    with st.expander("View All Formalized Logical Rules"):
        for rule in kb.rules:
            st.markdown(f"- **{rule['id']}**: IF `{rule['if']}` THEN **{rule['then']}**")

with tab3:
    st.subheader("Module 3: Machine Learning & 1x / 10x / 100x Scalability Benchmarks")
    st.markdown("Performance metrics across scaled operational workloads:")

    if st.button("Run Scalability Benchmark Suite"):
        results = [
            classifier.evaluate_workload(1),
            classifier.evaluate_workload(10),
            classifier.evaluate_workload(100),
        ]
        df_results = pd.DataFrame(results)
        st.dataframe(df_results, use_container_width=True)
        st.line_chart(df_results.set_index("workload")["execution_time_sec"])

with tab4:
    st.subheader("Capstone Team & Evaluation Compliance")
    st.info(
        """
        - **Project Title:** DeepShield-X - Multimodal Deepfake Detection and Digital Identity Verification Engine
        - **Module 1 (Agent & Search):** Mohammed Falah Miya (Reg. No: 1911260265)
        - **Module 2 (Knowledge & Reasoning):** Sreevathsan K (Reg. No: 1924260100)
        - **Module 3 (ML & Scalability):** Yuvanshankar M (Reg. No: 1912260450)
        """
    )
