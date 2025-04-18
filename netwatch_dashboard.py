# netwatch_dashboard.py (Revamped)
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
import os
import shap
import matplotlib.pyplot as plt
import json
import joblib
import tempfile
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from streamlit_javascript import st_javascript

# ==========================
# Page Config
# ==========================
st.set_page_config(page_title="NetWatch – Smart WiFi Health", layout="wide")

# ==========================
# Sidebar Navigation
# ==========================
st.sidebar.title("🛰️ NetWatch Navigation")
tabs = st.sidebar.radio("Go to:", [
    "Welcome", 
    "Experience Predictor", 
    "Outage Detector", 
    "Connection Tester", 
    "Time Series Explorer",
    "Behavior Clusters", 
    "Incident Log", 
    "Technical Analysis",
    "WiFi Simulation",
    "Real vs Simulated", 
    "Dataset Stories",
    "About"
], key="main_tabs")


# ==========================
# Load data and models
# ==========================
@st.cache_data
def load_data():
    df = pd.read_csv("Generated_data/netwatch_wifi_cleaned.csv")
    np.random.seed(42)
    df["rssi"] += np.random.normal(0, 5, len(df))
    df["latency_ms"] += np.random.exponential(5, len(df))
    df["jitter_ms"] += np.random.normal(0, 3, len(df))
    df["latency_ms"] += np.random.choice([0, 100, 300], size=len(df), p=[0.98, 0.01, 0.01])
    return df

data = load_data()

@st.cache_resource
def load_model(path):
    return joblib.load(path)

# SHAP-safe helper function (unhashable model fix)
def get_shap_values(_model, sample):
    explainer = shap.TreeExplainer(_model)
    return explainer.shap_values(sample)

# SHAP render helper
def render_shap_explainer(model, data_sample):
    try:
        st.markdown("SHAP summary for class 1 (outage):")
        plt.clf()
        shap_values = get_shap_values(model, data_sample)
        shap.summary_plot(
            shap_values[1] if isinstance(shap_values, list) else shap_values,
            data_sample,
            plot_type="bar",
            show=False
        )
        fig = plt.gcf()
        st.pyplot(fig)
        st.write(data_sample.head())
    except Exception as e:
        st.error(f"Could not display SHAP plot: {e}")

def download_model_from_gdrive(file_id, destination_path="models/experience_score_model.pkl"):
    """Download a file from Google Drive using file_id."""
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    response.raise_for_status()

    with open(destination_path, "wb") as f:
        f.write(response.content)
    
    return destination_path

@st.cache_resource
def download_experience_model():
    # Your shared Google Drive file ID
    file_id = "1v8rzLByAJpChO2fN1HxFOOBDS16NcH6y"
    gdrive_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(gdrive_url)
    if response.status_code != 200:
        raise Exception("Failed to download model from Google Drive.")
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(response.content)
        return joblib.load(tmp.name)

# Use your actual file ID here
EXPERIENCE_MODEL_FILE_ID = "1v8rzLByAJpChO2fN1HxFOOBDS16NcH6y"
experience_model = download_model_from_gdrive(EXPERIENCE_MODEL_FILE_ID)

# ==========================
# TABS
# ==========================

if tabs == "Welcome":
    st.title("👋 Welcome to NetWatch")
    st.markdown("""
    Hello there! I'm Keerthi — and this is **NetWatch**, your all-in-one WiFi health dashboard.

    It's fun, it's smart, and it tells you **exactly what’s going on** with your smart home WiFi setup.

    From spotting outages to simulating futuristic network layouts — you're gonna love what's ahead 👇
    """)

    st.markdown("---")
    st.subheader("🎥 What Is This Dashboard?")
    st.markdown("""
    NetWatch helps you **track**, **predict**, and **understand** your WiFi's behavior — from everyday performance to critical crashes.

    Whether you're a curious nerd, a smart home enthusiast, or just tired of glitchy internet during Zoom calls, this dashboard is for you.

    → Built with real-world and simulated data  
    → Powered by machine learning models  
    → Wrapped in Keerthi-style vibes 🧠✨
    """)

    st.markdown("---")
    st.subheader("💡 What You’ll Find Inside (How It Works)")
    st.markdown("""
    **📡 Experience Predictor**  
    → Ever wonder *how smooth* your WiFi feels? This model takes signal values like RSSI, latency, jitter, and packet loss, and gives you a score.  
    - 0.9+ = buttery smooth  
    - < 0.6 = buffering, freezing, and frustration 😩

    **⚠️ Outage Detector**  
    → The drama queen of the dashboard. Uses the same stats, but instead of “feel,” it tells you: “Will this thing crash?”  
    If RSSI is trash, latency is laggy, or packet loss is maxed out — this will yell 🚨 *OUTAGE LIKELY*.

    **📈 Time Series Explorer**  
    → Want to see your WiFi’s mood swings over time? This shows RSSI changes for different homes/devices.  
    Great for spotting drop-offs, spikes, or ghosting gadgets.

    **🧠 Behavior Clusters**  
    → It's like high school cliques but for your devices.  
    We run PCA + clustering to group smart devices that behave similarly — Hue bulbs with Hue bulbs, WeMos with WeMos. 🧠💡

    **📋 Incident Log**  
    → When multiple devices rage-quit together, something’s up.  
    If 3+ devices drop within 15 minutes, we log it as a major incident. Think of it like group therapy for your gadgets 😤📉.

    **🧪 Technical Analysis**  
    → The control room. For those who love stats, graphs, confusion matrices, and SHAP plots.  
    It’s where all the nerdy magic happens.

    **🛰 WiFi Simulation**  
    → A virtual floorplan where you can play with dots.  
    Pretend you live in a smart apartment and drag gadgets around. Signal strengths change, and the vibes follow.

    **📊 Dataset Stories**  
    → Peek into the real and simulated data powering NetWatch — from malware traffic to smart home chatter.

    **🧭 About**  
    → Who made this (me 👋), why, and what's next.
    """)

    st.markdown("---")
    st.success("💬 Tip: Click any tab on the left to begin!")


elif tabs == "Experience Predictor":
    st.header("📡 WiFi Experience Score Predictor")
    st.markdown("Move the sliders below to simulate real-world WiFi signal conditions.")

    with st.expander("💡 Try This!"):
        st.markdown("""
        - Set **RSSI** to `-80`  
        - **Packet Loss** to `0.8`  
        - See how badly your experience score tanks! 😅  
        """)

    col1, col2 = st.columns(2)
    with col1:
        rssi = st.slider("📶 RSSI (Signal Strength)", -100, -20, -52)
        latency = st.slider("🕒 Latency (ms)", 0, 300, 76)
    with col2:
        jitter = st.slider("📊 Jitter (ms)", 0, 300, 146)
        packet_loss = st.slider("🥔 Packet Loss", 0.00, 1.00, 0.13)

    #model = load_model("models/experience_score_model.pkl")
    model = experience_model

    features = pd.DataFrame([[rssi, latency, jitter, packet_loss]],
                            columns=["rssi", "latency_ms", "jitter_ms", "packet_loss"])
    score = model.predict(features)[0]

    if score >= 0.85:
        emoji, note = "🌟", "Flawless! You could livestream a space launch 🚀"
        color = "green"
    elif score >= 0.6:
        emoji, note = "😐", "Decent but not perfect. Some buffering might sneak in."
        color = "orange"
    else:
        emoji, note = "🚨", "Yikes. Expect slowdowns, maybe even full drops 😬"
        color = "red"

    st.markdown(f"<h3 style='color:{color}; font-weight:bold;'>{emoji} Experience Score: {score:.3f}</h3>", unsafe_allow_html=True)
    st.markdown(f"<span style='color:{color}'>{note}</span>", unsafe_allow_html=True)

    with st.expander("ℹ️ What Do These Features Mean?"):
        st.markdown("""
        - **RSSI** → Signal strength. Lower (like -90) = weaker signal.
        - **Latency (ms)** → Time it takes data to travel. High = laggy.
        - **Jitter (ms)** → Fluctuation in latency. High = unstable connection.
        - **Packet Loss** → % of data packets lost. 1.0 = disaster.
        - **is_connected** → 1 if the device is online, 0 if not.
        - **Experience Score** → A number from 0 to 1 showing how smooth your WiFi is.
        - **Predicted Outage** → Model says 1 = outage likely, 0 = you're safe.
        """)

elif tabs == "Connection Tester":
    st.header("🧪 Test My Connection")
    st.markdown("Let's quickly measure your network latency and predict your WiFi experience in real time.")
    st.info("⏱️ This will ping a reliable server from your browser and estimate how fast your internet responds.")

    latency_script = """
    const start = performance.now();
    fetch("https://api.github.com", { mode: "cors" }).then(() => {
        const end = performance.now();
        Streamlit.setComponentValue(end - start);
    });
    """
    from streamlit_javascript import st_javascript
    latency_ms = st_javascript(latency_script, key="latency_test_final")

    st.caption("🔍 Measuring latency using GitHub API (fast, safe, and public)")

    if latency_ms and latency_ms > 0:
        st.success(f"📶 Estimated Latency: `{round(latency_ms)} ms`")

        # Example signal values (you can later customize or make them dynamic)
        rssi = -55
        jitter = 15
        packet_loss = 0.01

        # Run through your experience model
        #model = load_model("models/experience_score_model.pkl")
        model = experience_model

        input_df = pd.DataFrame([[rssi, latency_ms, jitter, packet_loss]],
                                columns=["rssi", "latency_ms", "jitter_ms", "packet_loss"])
        score = model.predict(input_df)[0]

        # Display result
        if score >= 0.85:
            emoji, note, color = "🌟", "Flawless connection. You could livestream a rocket launch 🚀", "green"
        elif score >= 0.6:
            emoji, note, color = "😐", "Decent connection — good for browsing or calls", "orange"
        else:
            emoji, note, color = "🚨", "Yikes. Expect buffering, drops, or lag", "red"

        st.markdown(f"<h3 style='color:{color}'>{emoji} Experience Score: {score:.2f}</h3>", unsafe_allow_html=True)
        st.markdown(f"<span style='color:{color}'>{note}</span>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Could not measure latency. This may not work on localhost. Try it on your deployed dashboard.")


elif tabs == "Outage Detector":
    st.header("⚠️ Predicting WiFi Outages")
    st.markdown("Based on signal stats — will your WiFi crash and burn? Let’s see.")

    with st.expander("💡 Try This!"):
        st.markdown("""
        - **RSSI** = `-90`, **Latency** = `280`, **Packet Loss** = `1.0`  
        - A full WiFi meltdown 💥
        """)

    col1, col2 = st.columns(2)
    with col1:
        rssi = st.slider("📶 RSSI", -100, -20, -90)
        latency = st.slider("🕒 Latency (ms)", 0, 300, 280)
    with col2:
        jitter = st.slider("📊 Jitter (ms)", 0, 300, 50)
        packet_loss = st.slider("🥔 Packet Loss", 0.00, 1.00, 1.0)

    test_data = pd.DataFrame([[rssi, latency, jitter, packet_loss]],
        columns=["rssi", "latency_ms", "jitter_ms", "packet_loss"])

    model_bundle = load_model("models/outage_detector_model.pkl")
    clf_model = model_bundle["model"]
    threshold = st.slider("🎯 Outage Sensitivity Threshold", 0.0, 1.0, model_bundle["threshold"], step=0.05)

    st.caption("🧠 This prediction is based on your chosen threshold. Lower it to catch more risks, raise it for stricter detection.")


    prob = clf_model.predict_proba(test_data)[0][1]
    st.metric("📉 Outage Probability", f"{prob:.2%}")

    prediction = int(prob > threshold)

    if rssi <= -85 and latency >= 250 and packet_loss == 1.0 and prediction == 0:
        st.warning("⚠️ This looks like a full meltdown… but the model didn't predict an outage. Might need retraining!")

    if prediction == 1:
        st.error("🚨 Outage Likely! Get ready for connection drops.")
    else:
        st.success("✅ You're in the clear! No outage predicted.")

    with st.expander("ℹ️ What Do These Features Mean?"):
        st.markdown("""
        - **RSSI** → Signal strength. Lower (like -90) = weaker signal.
        - **Latency (ms)** → Time it takes data to travel. High = laggy.
        - **Jitter (ms)** → Fluctuation in latency. High = unstable connection.
        - **Packet Loss** → % of data packets lost. 1.0 = disaster.
        - **is_connected** → 1 if the device is online, 0 if not.
        - **Experience Score** → A number from 0 to 1 showing how smooth your WiFi is.
        - **Predicted Outage** → Model says 1 = outage likely, 0 = you're safe.
        """)

    st.markdown("**🤖 Why did the model say “outage”?**<br>SHAP shows how much each feature pushed the model toward or away from predicting a WiFi crash.", unsafe_allow_html=True)

    # Inside Outage Detector (SHAP EXPANDER SECTION)
    with st.expander("🧠 SHAP Explainer – Why did the model predict an outage?"):
        sample_data = data[["rssi", "latency_ms", "jitter_ms", "packet_loss"]].dropna().sample(100, random_state=42)
        render_shap_explainer(clf_model, sample_data)

elif tabs == "Time Series Explorer":
    st.header("📈 Signal Strength Over Time")
    st.markdown("Visualize how RSSI changed over time for different homes.")

    selected_home = st.selectbox("Select a Home", sorted(data["home_id"].unique()))
    df_filtered = data[data["home_id"] == selected_home]
    fig = px.line(df_filtered, x="timestamp", y="rssi", color="device_id",
                  title=f"Signal Strength – Home {selected_home}")
    fig.update_traces(mode="lines+markers", hovertemplate="Time: %{x}<br>RSSI: %{y}")
    fig.add_vline(x="2025-04-11 12:00:00", line_dash="dot", line_color="red")

    st.plotly_chart(fig, use_container_width=True)

elif tabs == "Behavior Clusters":
    st.header("🧠 Device Behavior Clusters")
    st.markdown("This scatterplot shows how similar devices behave in your home.")

    cluster_df = pd.read_csv("output/iot_clustered_pca.csv")
    fig = px.scatter(
        cluster_df,
        x="PCA1", y="PCA2",
        color="device",
        size="mean",
        hover_data=["mean", "std", "cluster"],
        title="📊 IoT Device Behavior: PCA Projection",
        width=1000, height=600
    )
    fig.update_traces(marker=dict(opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
    st.plotly_chart(fig, use_container_width=True)
    
    # 🔧 Set marker size manually (uniform size, not tied to data)
    fig.update_traces(marker=dict(size=12))  # or size=15 for big juicy dots
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Devices with similar traffic patterns are grouped together.")

elif tabs == "Incident Log":
    import datetime

    st.header("📋 Detected Incidents (15-min multi-device drops)")
    st.markdown("These logs show when multiple devices lost connection together – likely a WiFi outage.")

    log_data = pd.read_csv("logs/incident_log.csv")
    log_data["timestamp"] = pd.to_datetime(log_data["timestamp"])

    # === Filters ===
    all_severities = ["High", "Medium", "Low"]

    selected_severity = st.multiselect(
        "🔍 Filter by Severity",
        options=all_severities,
        default=["High", "Medium", "Low"]
    )

    unique_severities = sorted(log_data["severity"].dropna().unique())
    #selected_severity = st.multiselect("🔍 Filter by Severity", unique_severities, default=unique_severities)

    min_date = log_data["timestamp"].min().date()
    max_date = log_data["timestamp"].max().date()
    date_range = st.date_input("🗓️ Filter by Date", (min_date, max_date), min_value=min_date, max_value=max_date)

    home_ids = sorted(log_data["home_id"].unique())
    selected_home = st.selectbox("🏠 Filter by Home ID", options=["All"] + [str(h) for h in home_ids])

    # === Apply Filters ===
    filtered = log_data[
        (log_data["severity"].isin(selected_severity)) &
        (log_data["timestamp"].dt.date >= date_range[0]) &
        (log_data["timestamp"].dt.date <= date_range[1])
    ]
    if selected_home != "All":
        filtered = filtered[filtered["home_id"] == int(selected_home)]

    filtered = filtered.sort_values("timestamp", ascending=False)

    severity_counts = filtered["severity"].value_counts()
    for sev in all_severities:
        count = severity_counts.get(sev, 0)
        st.markdown(f"- **{sev}**: {count} entries")

    # === Export CSV ===
    st.download_button("📥 Download Filtered Logs as CSV", data=filtered.to_csv(index=False), file_name="incident_log_filtered.csv", mime="text/csv")

    # === View Toggle ===
    view_mode = st.radio("🖥️ View Mode", ["Stylized HTML", "Raw Table"], horizontal=True)

    # === Styled Severity Cell ===
    def styled_severity(sev):
        emoji = "🔥" if sev == "High" else "⚠️" if sev == "Medium" else "✅"
        color = "red" if sev == "High" else "orange" if sev == "Medium" else "green"
        return f"<span style='color:{color}; font-weight:bold'>{emoji} {sev}</span>"

    # === Display Table ===
    st.markdown("### 🔥 Filtered Incident Log")

    if filtered.empty:
        st.info("No incidents match your filters.")
    elif view_mode == "Raw Table":
        st.dataframe(filtered)
    else:
        # Build Full HTML Table
        rows_html = ""
        for _, row in filtered.iterrows():
            rows_html += f"<tr style='border-bottom:1px solid #eee;'>"
            rows_html += f"<td>{row['home_id']}</td>"
            rows_html += f"<td>{row['timestamp']}</td>"
            rows_html += f"<td>{row['affected_devices']}</td>"
            rows_html += f"<td>{row['alert_type']}</td>"
            rows_html += f"<td>{styled_severity(row['severity'])}</td>"
            rows_html += "</tr>"

        full_table = (
            "<table style='width:100%; border-collapse: collapse; margin-top:1em'>"
            "<thead><tr style='text-align:left; border-bottom:2px solid #ccc;'>"
            "<th>Home ID</th><th>Timestamp</th><th>Affected Devices</th><th>Alert Type</th><th>Severity</th>"
            "</tr></thead><tbody>"
            f"{rows_html}"
            "</tbody></table>"
        )

        #st.markdown(full_table, unsafe_allow_html=True)
        st.markdown(f"""
        <div style='overflow-x:auto'>
        {full_table}
        </div>
        """, unsafe_allow_html=True)



    # === 📊 Analytics Charts ===
    st.markdown("### 📊 Incident Analytics")

    col1, col2 = st.columns(2)

    with col1:
        top_homes = filtered["home_id"].value_counts().reset_index()
        top_homes.columns = ["home_id", "incident_count"]
        st.bar_chart(top_homes.set_index("home_id"))

    with col2:
        filtered["date"] = filtered["timestamp"].dt.date
        daily_counts = filtered["date"].value_counts().sort_index()
        st.line_chart(daily_counts)

    # === 🧠 Insight ===
    if not filtered.empty:
        worst_home = top_homes.iloc[0]["home_id"]
        total = int(top_homes.iloc[0]["incident_count"])
        st.markdown(f"**🧠 Home ID `{worst_home}` had the most outages — `{total}` times in your filters.**")

elif tabs == "Technical Analysis":

    st.header("🧪 Deep Dive – Technical Analysis")
    st.markdown("Welcome to the data science brain of NetWatch — this is where we break down what's really powering those predictions.")

    bundle = load_model("models/outage_detector_model.pkl")
    clf = bundle["model"]

    tab_titles = [
        "📊 Feature Importance",
        "🧠 SHAP Explainer",
        "📈 Signal Distributions",
        "🔍 Feature Correlations",
        "🦠 CTU Malware Detection",
        "📡 IoT Frame Lengths",
        "🕵️‍♀️ WLS Logs"
    ]
    selected_tab = st.tabs(tab_titles)

    with selected_tab[0]:
        st.subheader("📊 Feature Importance (Outage Model)")
        st.markdown("""
        **🧠 Why it matters:** This shows which features the model thinks are most important when deciding if your WiFi will fail.

        Features like RSSI and latency often dominate because they directly reflect your network's signal quality and response time.
        Higher importance means the model leans heavily on that feature when predicting.
        """)
        importances = clf.feature_importances_
        features = ["rssi", "latency_ms", "jitter_ms", "packet_loss"]
        feat_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values("Importance")
        fig, ax = plt.subplots(figsize=(4.5, 2.2))
        bars = ax.barh(feat_df.Feature, feat_df.Importance, color="skyblue")
        ax.set_title("Feature Importance", fontsize=11)
        ax.set_xlabel("Importance")
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.005, bar.get_y() + bar.get_height()/2, f"{width:.2f}", va='center', fontsize=7)
        st.pyplot(fig)

    with selected_tab[1]:
        st.subheader("🧠 SHAP Explainer")
        st.markdown("""
        **🔍 Think of this like opening the model's brain.** This plot shows which features pushed it toward predicting an outage.

        🔴 = pushed toward outage
        🔵 = pushed toward safety
        """)
        sample_data = data[["rssi", "latency_ms", "jitter_ms", "packet_loss"]].dropna().sample(100, random_state=42)
        render_shap_explainer(clf, sample_data)

    with selected_tab[2]:
        st.subheader("📈 Simulated Signal Distributions")
        st.markdown("""
        - **RSSI** above -70 = strong signal. Below -85 = danger zone 🚨
        - **Latency** under 100ms is good for video calls and gaming 🎮
        """)
        fig1 = px.histogram(data, x="rssi", nbins=40, title="RSSI Distribution")
        fig1.add_vline(x=-70, line_dash="dot", line_color="green")
        fig1.add_vline(x=-85, line_dash="dot", line_color="red")
        st.plotly_chart(fig1)

        fig2 = px.histogram(data, x="latency_ms", nbins=40, title="Latency Distribution")
        fig2.add_vline(x=100, line_dash="dot", line_color="red")
        st.plotly_chart(fig2)

    with selected_tab[3]:
        st.subheader("🔍 Feature Correlations")
        st.markdown("""
        A heatmap of how different network stats relate.

        - 🔵 = Positive correlation (rise together)
        - 🔴 = Negative correlation (one rises, other falls)
        """)
        fig_corr = px.imshow(data.corr(numeric_only=True), color_continuous_scale="RdBu", zmin=-1, zmax=1)
        st.plotly_chart(fig_corr)

    with selected_tab[4]:
        st.subheader("🦠 CTU Malware Classifier")
        st.markdown("""
        Trained on real network traffic — detects malware by analyzing packet patterns.

        📋 **Classifier Metrics**
        - **Accuracy:** 99.92%
        - **Malware Precision:** 97.9%
        - **Malware Recall:** 94%
        - **F1 Score:** 95.91%
        """)
        from sklearn.metrics import confusion_matrix, classification_report
        import seaborn as sns

        clf_ctu = joblib.load("models/ctu_malware_model.pkl")
        ctu_df = pd.read_csv("Real_data/ctu_malware_dataset/capture20110810.binetflow.txt", skipinitialspace=True)
        ctu_df = ctu_df.dropna(subset=["TotBytes", "TotPkts", "Dur", "Label"])
        ctu_df["is_malicious"] = ~ctu_df["Label"].str.contains("Background", case=False)

        ctu_df["SrcBytes"] = pd.to_numeric(ctu_df["SrcBytes"], errors='coerce')
        ctu_df["DstBytes"] = ctu_df["TotBytes"] - ctu_df["SrcBytes"]
        ctu_df["Proto"] = ctu_df["Proto"].astype('category').cat.codes
        ctu_df.dropna(subset=["SrcBytes", "DstBytes", "Proto"], inplace=True)

        X_ctu = ctu_df[["TotBytes", "TotPkts", "Dur", "SrcBytes", "DstBytes", "Proto"]]
        y_ctu = ctu_df["is_malicious"]
        y_pred = clf_ctu.predict(X_ctu)

        cm = confusion_matrix(y_ctu, y_pred)
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig_cm)

    with selected_tab[5]:
        st.subheader("📡 IoT Frame Lengths")
        st.markdown("""
        Devices behave differently — this shows how packet sizes differ across devices and clusters.
        """)
        try:
            iot_df = pd.read_csv("output/iot_clustered.csv")
            fig1 = px.histogram(iot_df, x="frame_len", color="device", nbins=50, title="Frame Lengths by Device")
            st.plotly_chart(fig1)
            fig2 = px.histogram(iot_df, x="frame_len", color="cluster", nbins=50, title="Clusters by Frame Length")
            st.plotly_chart(fig2)
        except:
            st.warning("IoT data not available.")

    with selected_tab[6]:
        st.subheader("🕵️‍♀️ WLS Logs")
        st.markdown("""
        Logging events from Windows — great for spotting frequent or suspicious activity.
        """)
        try:
            import json
            logs = []
            with open("Real_data/wls_day-01", "r") as f:
                for i, line in enumerate(f):
                    if i >= 20000: break
                    try:
                        logs.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
            df_wls = pd.DataFrame(logs)
            base_time = pd.Timestamp("2024-01-01 00:00:00")
            df_wls['timestamp'] = [base_time + pd.Timedelta(seconds=i*5) for i in range(len(df_wls))]
            df_wls.set_index('timestamp', inplace=True)

            hourly_counts = df_wls['ProcessName'].resample('1H').count()
            fig_wls = px.line(hourly_counts, title="WLS Event Frequency Over Time")
            st.plotly_chart(fig_wls)

            st.markdown("### 🔎 Top Processes")
            top_processes = df_wls['ProcessName'].value_counts().nlargest(10).reset_index()
            top_processes.columns = ['ProcessName', 'Count']
            fig_proc = px.bar(top_processes, x='ProcessName', y='Count', title="Top 10 Processes")
            st.plotly_chart(fig_proc)

            st.markdown("### 💻 Top Hosts")
            top_hosts = df_wls['LogHost'].value_counts().nlargest(10).reset_index()
            top_hosts.columns = ['LogHost', 'EventCount']
            fig_hosts = px.bar(top_hosts, x='LogHost', y='EventCount', title="Top 10 Logging Hosts")
            st.plotly_chart(fig_hosts)

        except Exception as e:
            st.warning(f"Could not parse WLS logs: {e}")

    st.markdown("---")
    st.markdown("Made with ☕, SHAP values, and a bit of sass ✨")

elif tabs == "WiFi Simulation":
    st.header("🛰️ WiFi Coverage Playground")
    st.markdown("Imagine a house floorplan where your devices are randomly scattered...")

    sim_df = pd.DataFrame({
        "x": np.random.randint(0, 100, 20),
        "y": np.random.randint(0, 100, 20),
        "rssi": np.random.uniform(-60, -30, 20),
        "device": [f"Device {i+1}" for i in range(20)]
    })
    fig = px.scatter(sim_df, x="x", y="y", color="rssi", text="device",
                     color_continuous_scale="RdYlGn_r", title="🎯 WiFi Device Heatmap", width=800, height=600)
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Each dot is a device. Color = signal strength. Click and drag to explore!")

elif tabs == "Real vs Simulated":
    st.header("📊 Real vs Simulated Data")
    st.markdown("Compare signal characteristics, outage patterns, and device behavior between the synthetic data and multiple real-world datasets.")

    st.info("🧠 Each real-world dataset reflects a different type of network behavior. Try comparing each!")

    dataset_choice = st.selectbox("Choose a Real Dataset to Compare:", [
        "CTU Malware", "IoT Sentinel", "Los Alamos Logs"
    ])

    simulated_df = data.copy()
    simulated_df["source"] = "Simulated"

    try:
        if dataset_choice == "CTU Malware":
            real_df = pd.read_csv("Real_data/ctu_malware_dataset/capture20110810.binetflow.txt", skipinitialspace=True)
            real_df = real_df.rename(columns={"TotBytes": "rssi", "TotPkts": "latency_ms", "Dur": "jitter_ms"})
            real_df = real_df[real_df["rssi"] < 1e7]
            real_df["packet_loss"] = np.random.rand(len(real_df)) * 0.1
            real_df = real_df[["rssi", "latency_ms", "jitter_ms", "packet_loss"]]
            caption_note = "Malware network traffic often exhibits extreme behavior (spikes, burstiness)."

        elif dataset_choice == "IoT Sentinel":
            df1 = pd.read_csv("Real_data/tp_link_setup1.csv")
            df2 = pd.read_csv("Real_data/hue_setup1.csv")
            df3 = pd.read_csv("Real_data/wemo_setup1.csv")
            real_df = pd.concat([df1, df2, df3])
            real_df = real_df.rename(columns={"frame_len": "rssi"})
            real_df["latency_ms"] = np.random.randint(20, 150, len(real_df))
            real_df["jitter_ms"] = np.random.randint(0, 70, len(real_df))
            real_df["packet_loss"] = np.random.rand(len(real_df)) * 0.05
            real_df = real_df[["rssi", "latency_ms", "jitter_ms", "packet_loss"]]
            caption_note = "IoT traffic is lightweight but varies by device – expect clustered behaviors."

        elif dataset_choice == "Los Alamos Logs":
            lines = []
            with open("Real_data/wls_day-01", "r") as f:
                for i, line in enumerate(f):
                    if i >= 50000:
                        break
                    try:
                        lines.append(json.loads(line.strip()))
                    except:
                        continue
            df = pd.DataFrame(lines)
            df = df.dropna(subset=["ProcessName", "LogHost"])
            np.random.seed(42)
            real_df = pd.DataFrame({
                "rssi": np.random.normal(-60, 15, len(df)),
                "latency_ms": np.random.exponential(200, len(df)),
                "jitter_ms": np.random.normal(15, 5, len(df)),
                "packet_loss": np.random.rand(len(df)) * 0.1
            })
            caption_note = "Enterprise logs show consistent baseline but sudden event spikes."

        real_df["source"] = "Real"
        real_df["dataset"] = dataset_choice
        combined = pd.concat([real_df, simulated_df], ignore_index=True)

        if st.checkbox("🩺 Filter out extreme outliers (>99th percentile)"):
            for col in ["rssi", "latency_ms", "jitter_ms", "packet_loss"]:
                q = combined[col].quantile(0.99)
                combined = combined[combined[col] <= q]

        selected_features = st.multiselect(
            "📊 Select Features to Compare",
            ["rssi", "latency_ms", "jitter_ms", "packet_loss"],
            default=["rssi", "latency_ms", "jitter_ms", "packet_loss"]
        )


        for feat in selected_features:
            st.markdown(f"#### {feat.replace('_', ' ').title()} Distribution")
            fig = px.histogram(
                combined, x=feat, color="source",
                marginal="box", nbins=50, barmode="overlay",
                log_y=True,
                color_discrete_map={"Simulated": "#636EFA", "Real": "#EF553B"}
            )
            fig.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"🔎 **Insight:** {caption_note}")

        st.markdown("---")
        st.subheader("🧠 Cluster Comparison (if available)")
        try:
            sim_cluster = pd.read_csv("output/iot_clustered_pca.csv")
            real_cluster_path = "output/iot_real_clustered_pca.csv"
            if os.path.exists(real_cluster_path):
                real_cluster = pd.read_csv(real_cluster_path)
                sim_cluster["type"] = "Simulated"
                real_cluster["type"] = "Real"
                cluster_data = pd.concat([sim_cluster, real_cluster])
                fig = px.scatter(cluster_data, x="PCA1", y="PCA2", color="type", symbol="cluster")
                st.plotly_chart(fig)
            else:
                st.info("Real IoT cluster file not found. Skipping cluster comparison.")
        except Exception as e:
            st.warning(f"Error loading cluster data: {e}")

    except Exception as e:
        st.error(f"Failed to load real vs simulated comparison: {e}")

elif tabs == "Dataset Stories":
    st.header("📚 Dataset Stories")
    st.markdown("Discover the unique traits and storytelling power of each dataset used in NetWatch.")

    story_tab = st.tabs(["CTU Malware", "IoT Sentinel", "Los Alamos Logs", "Simulated Data"])

    with story_tab[0]:
        st.subheader("💀 CTU Malware")
        st.markdown("This dataset captures **real malware traffic** in a controlled network. Spikes in byte transfer, bursty flows, and erratic behaviors are typical. Useful to model cyber-attacks and anomaly detection.")
        df = pd.read_csv("Real_data/ctu_malware_dataset/capture20110810.binetflow.txt", skipinitialspace=True)
        df = df.dropna(subset=["TotBytes", "Dur", "Label"])
        df = df[df["TotBytes"] < 2.5e6]
        #fig = px.scatter(df, x="Dur", y="TotBytes", color=df["Label"].str.contains("Botnet"), title="💥 Traffic Bursts During Botnet Attacks")
        df["is_botnet"] = df["Label"].str.contains("Botnet")
        fig = px.scatter(
            df, x="Dur", y="TotBytes",
            color="is_botnet",
            color_discrete_map={True: "crimson", False: "steelblue"},
            title="💥 Traffic Bursts During Botnet Attacks"
        )
        fig.update_traces(marker=dict(size=6, opacity=0.7))

        st.plotly_chart(fig, use_container_width=True)
        st.caption("🧠 Behavior: Spiky, erratic flows. Good for anomaly detection and modeling cyber attacks.")

        st.markdown("""
        **🧠 What to notice:**
        - Botnet flows are bursty and send massive data quickly.
        - Normal traffic is calmer and more stable.
        - Helps spot malicious patterns using time and volume.
        """)


    with story_tab[1]:
        st.subheader("📡 IoT Sentinel")
        st.markdown("Captured from smart devices like TP-Link plugs, Hue lights, and WeMo switches. Traffic is light, frequent, and device-specific — great for understanding clustered behaviors.")
        df1 = pd.read_csv("Real_data/tp_link_setup1.csv")
        df2 = pd.read_csv("Real_data/hue_setup1.csv")
        df3 = pd.read_csv("Real_data/wemo_setup1.csv")
        df1["device"] = "TP-Link"
        df2["device"] = "Hue"
        df3["device"] = "WeMo"
        df_all = pd.concat([df1, df2, df3])
        df_all = df_all.rename(columns={"frame_len": "rssi"})
        #fig = px.histogram(df_all, x="rssi", color="device", barmode="group", title="📊 Frame Size by IoT Device")
        fig = px.histogram(
            df_all, x="rssi", color="device", nbins=40,
            barmode="overlay", opacity=0.7,
            title="📊 Frame Size by IoT Device"
        )

        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔬 Behavior: Clustered around device types, shows consistent but varied signal loads.")

        st.markdown("""
        **🔍 What to notice:**
        - TP-Link has frequent short bursts.
        - Hue bulbs send consistent, tiny signals.
        - WeMo varies a bit — all reflect **device behavior patterns**.
        """)


    with story_tab[2]:
        st.subheader("🏢 Los Alamos Logs")
        st.markdown("These are anonymized **enterprise logs** from system events. Patterns show long stable periods and rare bursts — good for modeling **baseline vs anomaly** in secure environments.")
        logs = []
        with open("Real_data/wls_day-01", "r") as f:
            for i, line in enumerate(f):
                if i >= 10000:
                    break
                try:
                    logs.append(json.loads(line.strip()))
                except:
                    continue
        df_logs = pd.DataFrame(logs)
        df_logs = df_logs[df_logs["ProcessName"].notna()]
        counts = df_logs["ProcessName"].value_counts().reset_index()
        counts.columns = ["Process", "Count"]
        #fig = px.bar(counts.head(15), x="Process", y="Count", title="📈 Top 15 Logged Processes")
        fig = px.bar(
            counts.head(15), x="Process", y="Count",
            color="Count", color_continuous_scale="Plasma",
            title="📈 Top 15 Logged Processes"
        )

        st.plotly_chart(fig, use_container_width=True)
        st.caption("📊 Behavior: Stable baseline with spikes — helpful for time-based anomaly detection.")

        st.markdown("""
        **🔎 What to notice:**
        - A few processes dominate — maybe schedulers or antivirus tasks.
        - Great for spotting **unexpected spikes** or **noisy apps**.
        """)


    with story_tab[3]:
        st.subheader("🧪 Simulated Data")
        st.markdown("This synthetic dataset is engineered to act like real WiFi. It’s clean, balanced, and full of controlled chaos — like random latency spikes and signal drops — so we can test how WiFi behaves under stress. Perfect for training models to detect slowdowns, signal degradation, or total chaos.")
        #fig = px.scatter(data.sample(2000), x="rssi", y="latency_ms", color="packet_loss",
        #                 color_continuous_scale="Blues", title="🔬 Simulated Signal Quality Snapshot")
        
        fig = px.scatter(
            data.sample(2000),
            x="rssi", y="latency_ms",
            color="packet_loss",
            hover_data=["jitter_ms"],
            color_continuous_scale="Turbo",
            title="🔬 Simulated Signal Quality Snapshot"
        )
        fig.update_traces(marker=dict(size=6, opacity=0.7))

        st.plotly_chart(fig, use_container_width=True)
        st.caption("🧬 Behavior: Clean but customizable. Designed to mimic real-world data with injected anomalies.")

        st.markdown("""
        **🧠 What to notice:**
        - Most signals live between **-70 to -40 dBm** — lower = worse.
        - Latency shoots up when signal weakens.
        - Darker dots = more packet loss = higher chance of WiFi pain.
        """)

elif tabs == "About":
    st.header("ℹ️ About This Project")
    st.markdown("""
    👩‍💻 **Built by:** Keerthi Chowdary Gogineni  
    🧠 **Goal:** Predict and explain WiFi health using ML  
    💡 **Data:** Simulated + Real IoT & Malware Logs  
    🎨 **Vibe:** Smart, friendly, and just a little dramatic
    """)
