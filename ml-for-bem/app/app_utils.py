import numpy as np
from ladybug.epw import EPW
import plotly.express as px
import pandas as pd
import streamlit as st
import json


@st.cache_data
def load_schedule(ix: int) -> np.ndarray:
    schedules = np.load("data/schedules.npy")
    return schedules[ix]


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


@st.cache_data
def load_space(path=None):
    if path is None:
        path = "app/space_definition.json"

    with open(path, "r") as f:
        space_config = json.load(f)

    return space_config


def make_controls_columns(
    space_config,
    col_count: int = 3,
    skip_params: list = [
        "width",
        "height",
        "perim_depth",
        "core_depth",
        "roof_2_footprint",
        "ground_2_footprint",
        "orientation",
        "wwr",
    ],
):
    columns = [[] for _ in range(col_count)]
    counter = 0
    for param, param_def in space_config.items():
        if param in skip_params or "shading" in param:
            continue
        columns[counter].append((param, param_def))

        counter = (counter + 1) % len(columns)
    return columns


def render_controls(columns, template_config, key_formatter, defaults=None):
    cols = st.columns(len(columns))
    for i, col_group in enumerate(columns):
        with cols[i]:
            for param, param_def in col_group:
                if param_def["mode"] == "Continuous":
                    template_config[param] = (
                        st.number_input(
                            param,
                            float(param_def["min"]),
                            float(param_def["max"]),
                            step=0.00001,
                            format="%.5f",
                            key=key_formatter(param),
                        )
                        if defaults is None
                        else st.number_input(
                            param,
                            float(param_def["min"]),
                            float(param_def["max"]),
                            step=0.00001,
                            format="%.5f",
                            value=defaults[param],
                            key=key_formatter(param),
                        )
                    )
                elif param_def["mode"] == "Onehot":
                    mass_labels = [
                        "Steelframe",
                        "Woodframe",
                        "Brick",
                        "Concrete",
                    ]
                    econ_labels = [
                        "No Economizer",
                        "Differential Enthalpy",
                    ]
                    hrv_labels = [
                        "No HRV",
                        "Sensible Recovery",
                        "Sensible + Latent Recovery",
                    ]
                    ventilation_labels = [
                        "No Mechanical Ventilation",
                        "Always On Mechanical Ventilation",
                        "Demand Control Mechanical Ventilation",
                    ]
                    labels = mass_labels
                    if "econ" in param.lower():
                        labels = econ_labels
                    elif "recovery" in param.lower():
                        labels = hrv_labels
                    elif "ventilation" in param.lower():
                        labels = ventilation_labels
                    elif "mass" in param.lower():
                        labels = mass_labels
                    else:
                        raise ValueError(f"Could not find labels for param: {param}")
                    assert (
                        len(labels) == param_def["option_count"]
                    ), f"Labels and option count do not match for param: {param}"
                    template_config[param] = st.selectbox(
                        param,
                        list(range(param_def["option_count"])),
                        format_func=lambda x: labels[x],
                        key=key_formatter(param),
                        index=defaults[param] if defaults is not None else 0,
                    )
    return template_config


def render_epw_upload() -> EPW:
    epw_file = st.file_uploader("Upload EPW file")
    if epw_file:
        epw, epw_df = load_epw_file(epw_file)
        render_epw_timeseries(epw_df)
        return epw
    else:
        return None


def render_epw_timeseries(epw: pd.DataFrame):
    timeseries_to_plot = st.selectbox("Select timeseries to plot", epw.columns, index=0)
    fig = px.line(epw, y=timeseries_to_plot)
    st.plotly_chart(fig, use_container_width=True)


@st.cache_resource
def load_epw_file(file) -> EPW:
    # # convert file to bytes
    import os

    os.makedirs("data/temp", exist_ok=True)
    with open("data/temp/uploaded_epw.epw", "wb") as f:
        f.write(file.getvalue())
    # file = file.bytes()
    epw = EPW("data/temp/uploaded_epw.epw")
    df = pd.DataFrame(
        {
            "Dry Bulb Temperature": epw.dry_bulb_temperature.values,
            "Dew Point Temperature": epw.dew_point_temperature.values,
            "Relative Humidity": epw.relative_humidity.values,
            "Wind Speed": epw.wind_speed.values,
            "Wind Direction": epw.wind_direction.values,
            "Global Horizontal Radiation": epw.global_horizontal_radiation.values,
            "Direct Normal Radiation": epw.direct_normal_radiation.values,
            "Diffuse Horizontal Radiation": epw.diffuse_horizontal_radiation.values,
        }
    )
    return epw, df


@st.cache_resource
def load_template_defaults():
    with open("app/template-defaults.json", "r") as f:
        template_defaults = json.load(f)
    return template_defaults
