import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data
def load_lib() -> tuple[pd.DataFrame, np.ndarray]:
    templates = pd.read_hdf("app/ref_templates.hdf", key="features")
    schedules = np.load("app/ref_templates_schedules.npy")
    templates = templates[templates.columns[::-1]]
    return templates, schedules


@st.cache_data
def template_climate_zones(template_df: pd.DataFrame) -> list[str]:
    return sorted(template_df.ClimateZone.unique().tolist())


@st.cache_data
def template_categories(template_df: pd.DataFrame) -> list[str]:
    return sorted(template_df.Category.unique().tolist())


@st.cache_data
def filter_templates(
    template_df: pd.DataFrame, czs: list[str], cats: list[str]
) -> pd.DataFrame:
    if len(czs) > 0:
        template_df = template_df[template_df.ClimateZone.isin(czs)]
    if len(cats) > 0:
        template_df = template_df[template_df.Category.isin(cats)]
    return template_df
