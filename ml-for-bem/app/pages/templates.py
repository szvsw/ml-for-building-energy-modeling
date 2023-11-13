import json

import numpy as np
import pandas as pd
import plotly as px
import streamlit as st

from app.app_utils import (
    filter_templates,
    load_lib,
    template_categories,
    load_space,
    template_climate_zones,
)

st.set_page_config(
    "UBEM.io",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.title("UBEM.io (v2)")

st.divider()
templates, schedules = load_lib()
l, r = st.columns(2, gap="medium")
with l:
    czs = st.multiselect("Climate Zone", template_climate_zones(templates))
with r:
    cats = st.multiselect("Category", template_categories(templates))
templates = filter_templates(templates, czs, cats)
st.dataframe(templates)

st.divider()


space_config = load_space()

for param, param_def in space_config.items():
    if param_def["mode"] == "Continuous":
        st.slider(param, float(param_def["min"]), float(param_def["max"]))
    elif param_def["mode"] == "Onehot":
        st.selectbox(param, list(range(param_def["option_count"])))
