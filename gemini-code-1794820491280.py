# 1. Make classifier deterministic and tied to structured features
class DeepShieldClassifier:
    def __init__(self):
        # Features: [anomaly_ratio, derived_risk_count, size_kb_norm, type_code, high_sev_flag]
        X_train = np.array([
            [0.00, 0.0, 0.2, 0.0, 0.0],
            [0.05, 0.0, 0.5, 0.0, 0.0],
            [0.10, 0.0, 0.3, 0.3, 0.0],
            [0.15, 0.1, 0.7, 0.6, 0.0],
            [0.35, 0.3, 0.4, 0.0, 1.0],
            [0.50, 0.5, 0.8, 0.3, 1.0],
            [0.70, 0.7, 0.2, 0.6, 1.0],
            [0.90, 0.9, 0.9, 1.0, 1.0],
        ])
        y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        self.model = DecisionTreeClassifier(random_state=42, max_depth=4)
        self.model.fit(X_train, y_train)

    def predict_sample(self, features):
        pred = int(self.model.predict([features])[0])
        proba = self.model.predict_proba([features])[0]
        return pred, float(max(proba))


# 2. In Tab 1: start with default=[] and compute deterministic features
selected_evidence = st.multiselect(
    "Select Detected Evidence / Anomalies",
    list(kb.facts),
    default=[]
)

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
    has_critical = any(fact in derived for fact in critical_conclusions)

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

    st.success(f"**Final Verdict:** {verdict}")
    st.metric("Model Confidence Score", f"{confidence * 100:.2f}%")

    with st.expander("🔍 View Agent Search Metrics"):
        st.write(search_res)

    with st.expander("📜 View Logical Rule Inference Trace"):
        if trace:
            for t in trace:
                st.text(t)
        else:
            st.text("No inference rules triggered from current evidence.")
