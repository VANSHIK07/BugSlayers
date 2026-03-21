import streamlit as st
import plotly.express as px
from data import get_air_quality
import pandas as pd
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="City Environment Monitor", page_icon="🌍")

st.title("🌍 Satellite Environmental Intelligence Platform")
st.subheader("Smart City Environmental Dashboard")

# City selector
city = st.selectbox("Select a City", ["Delhi", "Mumbai", "Bengaluru", "Chennai"])

if st.button("Analyse City"):
    with st.spinner("Fetching satellite data..."):
        df = get_air_quality(city)
    
    st.success(f"Data loaded for {city}!")
    
    # Show metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg PM2.5", f"{df['pm2_5'].mean():.1f} µg/m³")
    col2.metric("Avg NO₂", f"{df['nitrogen_dioxide'].mean():.1f} µg/m³")
    col3.metric("Avg PM10", f"{df['pm10'].mean():.1f} µg/m³")
    
    # Chart
    fig = px.line(df, x="time", y="pm2_5", title=f"PM2.5 Levels in {city} - Last 7 Days")
    st.plotly_chart(fig)
    
    # Simple risk level
    avg = df['pm2_5'].mean()
    if avg > 60:
        st.error("🔴 HIGH POLLUTION RISK — Immediate action needed")
    elif avg > 30:
        st.warning("🟡 MODERATE RISK — Monitor closely")
    else:
        st.success("🟢 LOW RISK — Air quality acceptable")
        

# Action Plan Section
    st.subheader("📋 Recommended Action Plan")
    
    if avg > 60:
        st.error("🔴 HIGH RISK CITY")
        st.write("""
        ### Immediate Actions (0-7 days):
        - 🚗 Implement odd-even vehicle restrictions
        - 🏭 Suspend high-emission industries temporarily
        - 😷 Issue public health advisory for vulnerable groups
        - 🚌 Increase public transport frequency
        
        ### Short Term (1-3 months):
        - 🌳 Emergency tree plantation drive
        - ⚡ Accelerate EV adoption incentives
        - 🏗️ Strict dust control on construction sites
        """)
    elif avg > 30:
        st.warning("🟡 MODERATE RISK CITY")
        st.write("""
        ### Recommended Actions:
        - 📊 Increase air quality monitoring stations
        - 🚲 Build dedicated cycling lanes
        - 🌿 Expand urban green cover by 15%
        - 🏭 Monthly industrial emission audits
        """)
    else:
        st.success("🟢 LOW RISK CITY")
        st.write("""
        ### Maintenance Actions:
        - ✅ Continue current pollution controls
        - 📈 Monthly monitoring and reporting
        - 🌳 Maintain existing green cover
        """)