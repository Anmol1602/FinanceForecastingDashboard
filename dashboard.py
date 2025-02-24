import streamlit as st
import dashboard_main
import historical_analysis
import about  # Import AFTER set_page_config()
import live_market

# Set page configuration (MUST be the first Streamlit command)
st.set_page_config(
    page_title="Financial Forecasting Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Historical Analysis", "About","Live Market"], index=0)

# Route to the selected page
if page == "Dashboard":
    dashboard_main.main()
elif page == "Historical Analysis":
    historical_analysis.main()
elif page == "Live Market":
    live_market.live_market()
elif page == "About":
    about.main()
