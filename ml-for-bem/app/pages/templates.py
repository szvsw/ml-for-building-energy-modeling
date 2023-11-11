import streamlit as st
import numpy as np
import pandas as pd
import json
import plotly as px

from app.app_utils import (
    load_lib,
    template_climate_zones,
    template_categories,
    filter_templates,
)

st.set_page_config(
    "UBEM.io",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_data
def load_space(path=None):
    if path is None:
        path = "data/lightning/full_climate_zone/v7/train/space_definition.json"

    with open(path, "r") as f:
        space_config = json.load(f)

    return space_config


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
