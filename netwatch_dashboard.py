import streamlit as st
import pandas as pd
import plotly.express as px
import os

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="NetWatch Dashboard",
    layout="wide",
    page_icon="📡",
)

# ========== SIDEBAR ==========
st.sidebar.title("📡 NetWatch Navigation")
tab = st.sidebar.radio("Go to:", ["Welcome", "WiFi Health (Generated Data)", "Real Data Models", "IoT Clusters", "Incident Logs", "About"])

# ========== TAB: WELCOME ==========
if tab == "Welcome":
    st.title("📶 NetWatch – Smart WiFi Health & Disruption Prediction System")
    st.markdown("""
    Welcome to the NetWatch dashboard!  
    Explore how your home WiFi and smart devices behave — detect outages, predict disruptions, and analyze IoT behavior.
    
    Use the sidebar to navigate across various tabs.
    """)

# ========== TAB: WiFi Health ==========
elif tab == "WiFi Health (Generated Data)":
    st.header("📊 WiFi Experience Score & Outage Prediction")
    # Load example data (you can replace with actual paths)
    gen_data_path = "Generated_data/netwatch_wifi_cleaned.csv"
    if os.path.exists(gen_data_path):
        df = pd.read_csv(gen_data_path)
        st.dataframe(df.head())
        st.line_chart(df[["timestamp", "experience_score"]].set_index("timestamp"))
    else:
        st.warning("WiFi data not found!")

# ========== TAB: Real Data Models ==========
elif tab == "Real Data Models":
    st.header("🦠 Malware Detection from CTU Dataset")
    st.markdown("Sample results or charts from your real data classification model go here.")
    # You can load predictions, confusion matrix, accuracy here

# ========== TAB: IoT Clusters ==========
elif tab == "IoT Clusters":
    st.header("📡 IoT Device Behavior Clustering")
    cluster_data = "output/iot_clustered.csv"
    if os.path.exists(cluster_data):
        df = pd.read_csv(cluster_data)
        fig = px.scatter(df, x="PCA1", y="PCA2", color="cluster", hover_data=["device"])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Clustered IoT data not found!")

# ========== TAB: Incident Logs ==========
elif tab == "Incident Logs":
    st.header("📄 Incident Alerts Log")
    log_path = "logs/incident_log.csv"
    if os.path.exists(log_path):
        logs = pd.read_csv(log_path)
        st.dataframe(logs)
    else:
        st.warning("No incident logs available.")

# ========== TAB: About ==========
elif tab == "About":
    st.markdown("Made with ❤️ by Keerthi Chowdary Gogineni")
    st.markdown("This project uses generated + real datasets to monitor and predict WiFi & IoT behavior.")
