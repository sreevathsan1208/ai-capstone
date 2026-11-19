"""
DeepShield-X: Integrated Multimodal Deepfake Detection & Digital Identity Verification Engine
Team Capstone Project (SIMATS - B Slot)
Integrated Pipeline: Agent Search + Knowledge Reasoning + ML Classification + Multi-Format Uploader
"""

import hashlib
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
        # Features: [anomaly_ratio, derived_risk, entropy_norm, type_code, high_sev_flag]
        X_train = np.array([
            [0.00, 0.00, 0.20, 0.00, 0.0],
            [0.00, 0.00, 0.85, 0.33, 0.0],
            [0.05, 0.00, 0.50, 0.00, 0.0],
            [0.10, 0.00, 0.60, 0.33, 0.0],
            [0.15, 0.10, 0.70, 0.66, 0.0],
            [0.35, 0.30, 0.50, 0.00, 1.0],
            [0.50, 0.50, 0.80, 0.33, 1.0],
            [0.70, 0.70, 0.65, 0.66, 1.0],
            [0.90, 0.90, 0.90, 1.00, 1.0],
        ])
        y_train = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])

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


# ==========================================
# FORMAT-AGNOSTIC ASSET INSPECTOR
# ==========================================
def compute_byte_entropy_norm(raw_bytes):
    """Computes normalized Shannon byte entropy (0.0 to 1.0) for any uploaded file."""
    if not raw_bytes:
        return 0.0
    counts = np.bincount(np.frombuffer(raw_bytes, dtype=np.uint8), minlength=256)
    probs = counts[counts > 0] / len(raw_bytes)
    entropy = -float(np.sum(probs * np.log2(probs)))
    return round(min(entropy / 8.0, 1.0), 4)


def inspect_uploaded_asset(uploaded_file, media_type):
    """Extracts safe metadata and format-specific signals without failing on any format."""
    if uploaded_file is None:
        return None

    raw_bytes = uploaded_file.getvalue()
    sha256_short = hashlib.sha256(raw_bytes).hexdigest()[:12]
    entropy_norm = compute_byte_entropy_norm(raw_bytes)

    info = {
        "file_name": uploaded_file.name,
        "mime_type": uploaded_file.type or "application/octet-stream",
        "size_kb": round(len(raw_bytes) / 1024.0, 2),
        "sha256_prefix": sha256_short,
        "entropy_norm": entropy_norm,
        "format_details": {},
    }

    ext = uploaded_file.name.rsplit(".", 1)[-1].lower() if "." in uploaded_file.name else ""

    if media_type == "Image" or ext in {"jpg", "jpeg", "png", "webp"}:
        if PIL_AVAILABLE:
            try:
                with Image.open(BytesIO(raw_bytes)) as img:
                    w, h = img.size
                    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
                    info["format_details"] = {
                        "resolution": f"{w}x{h}",
                        "color_mode": img.mode,
                        "mean_brightness": round(float(np.mean(arr)), 1),
                        "contrast_std": round(float(np.std(arr)), 1),
                    }
            except Exception as e:
                info["format_details"] = {"image_parse_note": str(e)}

    elif media_type == "Video" or ext in {"mp4", "avi", "mov", "mkv", "webm"}:
        info["format_details"] = {
            "container_ext": ext or "unknown",
            "stream_bytes": len(raw_bytes),
        }

    elif media_type == "Audio Stream" or ext in {"wav", "mp3", "ogg", "m4a"}:
        info["format_details"] = {
            "audio_ext": ext or "unknown",
            "sample_bytes": len(raw_bytes),
        }

    else:
        text_preview = None
        if ext in {"txt", "csv", "json", "log", "md"}:
            try:
                text_preview = raw_bytes.decode("utf-8", errors="replace")[:400]
            except Exception:
                text_preview = None
        info["format_details"] = {
            "doc_ext": ext or "unknown",
            "is_pdf": ext == "pdf",
            "text_preview": text_preview,
        }

    return info


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
    st.subheader("End-to-End Media & Identity Verification")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 1. Upload Asset (Any Supported Format)")
        media_type = st.selectbox(
            "Select Asset Category",
            ["Image", "Video", "Audio Stream", "Digital Identity Document"],
        )

        uploaded_file = st.file_uploader(
            f"Upload your {media_type} file",
            type=[
                "jpg", "jpeg", "png", "webp",
                "mp4", "avi", "mov", "mkv", "webm",
                "wav", "mp3", "ogg", "m4a",
                "pdf", "txt", "csv", "json", "md",
            ],
        )

        st.markdown("#### 2. Configure Pipeline Parameters")
        input_size = st.slider("Search Input Size (Batch Items)", min_value=1, max_value=50, value=10)
        search_algo = st.radio("Search Algorithm", ["A*", "Greedy Best-First"])

        selected_evidence = st.multiselect(
            "Select Detected Evidence / Anomalies",
            sorted(kb.facts),
            default=[],
        )

        run_clicked = st.button("Run Full Pipeline Verification", type="primary")

    with col2:
        st.markdown("#### 3. Asset Preview & Execution Results")
        asset_info = inspect_uploaded_asset(uploaded_file, media_type)

        if uploaded_file is not None and asset_info is not None:
            st.write(
                "📁 **File Loaded:**",
                {
                    "FileName": asset_info["file_name"],
                    "Format": asset_info["mime_type"],
                    "Size": f"{asset_info['size_kb']} KB",
                    "SHA-256 Prefix": asset_info["sha256_prefix"],
                    "Byte Entropy (Norm)": asset_info["entropy_norm"],
                },
            )

            ext = uploaded_file.name.rsplit(".", 1)[-1].lower() if "." in uploaded_file.name else ""
            if ext in {"jpg", "jpeg", "png", "webp"} or (uploaded_file.type or "").startswith("image"):
                st.image(uploaded_file, caption="Uploaded Image Preview", use_container_width=True)
            elif ext in {"mp4", "avi", "mov", "mkv", "webm"} or (uploaded_file.type or "").startswith("video"):
                st.video(uploaded_file)
            elif ext in {"wav", "mp3", "ogg", "m4a"} or (uploaded_file.type or "").startswith("audio"):
                st.audio(uploaded_file)
            else:
                preview_txt = asset_info["format_details"].get("text_preview")
                if preview_txt:
                    st.text_area("Document Preview (First 400 chars)", preview_txt, height=120, disabled=True)
                else:
                    st.info("📄 Identity / binary document loaded. Metadata and rule-based verification ready.")
        else:
            st.info("ℹ️ Upload any supported file on the left to preview it, or run verification with selected anomalies.")

        if run_clicked:
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

            type_map = {
                "Image": 0.0,
                "Video": 0.33,
                "Audio Stream": 0.66,
                "Digital Identity Document": 1.0,
            }
            anomaly_ratio = len(selected_evidence) / max(len(kb.facts), 1)
            derived_risk = min(len(derived) / 10.0, 1.0)
            entropy_norm = asset_info["entropy_norm"] if asset_info else 0.0
            high_sev_flag = 1.0 if has_critical else 0.0

            features = [
                anomaly_ratio,
                derived_risk,
                entropy_norm,
                type_map.get(media_type, 0.0),
                high_sev_flag,
            ]
            pred_class, confidence = classifier.predict_sample(features)

            # Authentic by default unless critical rules fire or strong evidence + ML flag align
            is_fake = has_critical or (pred_class == 1 and len(selected_evidence) >= 2 and len(trace) >= 1)

            st.session_state["last_verification"] = {
                "is_fake": is_fake,
                "verdict": "🚨 DEEPFAKE / TAMPERING DETECTED" if is_fake else "✅ AUTHENTIC ASSET",
                "confidence": confidence,
                "selected_evidence": list(selected_evidence),
                "derived": sorted(derived),
                "triggered_critical": triggered_critical,
                "search_res": search_res,
                "trace": trace,
                "asset_info": asset_info,
            }

        if "last_verification" in st.session_state:
            res = st.session_state["last_verification"]
            if res["is_fake"]:
                st.error(f"**Final Verdict:** {res['verdict']}")
            else:
                st.success(f"**Final Verdict:** {res['verdict']}")

            st.metric("Model Confidence Score", f"{res['confidence'] * 100:.2f}%")

            with st.expander("🧾 View Reasoning Summary"):
                st.write(
                    f"**Selected Evidence ({len(res['selected_evidence'])}):**",
                    res["selected_evidence"] if res["selected_evidence"] else "None",
                )
                st.write(
                    f"**Inferred Conclusions ({len(res['derived'])}):**",
                    res["derived"] if res["derived"] else "None",
                )
                if res["triggered_critical"]:
                    st.write("**Critical Conclusions Triggered:**", res["triggered_critical"])
                elif not res["selected_evidence"]:
                    st.write("**Reason:** No anomaly indicators selected and no critical rules triggered.")
                else:
                    st.write("**Reason:** Selected indicators did not trigger critical tampering rules.")
                if res["asset_info"] and res["asset_info"]["format_details"]:
                    st.write("**Format Metadata:**", res["asset_info"]["format_details"])

            with st.expander("🔍 View Agent Search Metrics"):
                st.write(res["search_res"])

            with st.expander("📜 View Logical Rule Inference Trace"):
                if res["trace"]:
                    for t in res["trace"]:
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
