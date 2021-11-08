import streamlit as st
from multiapp import MultiApp
from apps import home, dashboard, notification #import modules

app = MultiApp()

# Config function
st.set_page_config(layout="wide", page_title='Smart Crowd Monitoring System')

hide_menu_style = """
<style>
#MainMenu {visibility:hidden; }
footer {visibility:hidden;}
</style>
"""

st.markdown(hide_menu_style,unsafe_allow_html=True)

st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #4D4D4D;">
  <div class="navbar-brand" target="_blank" style="color: #FFFFFF; font-size: 300%; margin-left: 30px;">Smart Crowd Monitoring System</div>
</nav>
""", unsafe_allow_html=True)

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Dashboard", dashboard.app)
app.add_app("Notification", notification.app)
# The main app
app.run()