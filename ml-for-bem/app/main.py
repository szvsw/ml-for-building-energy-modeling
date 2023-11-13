import os
import numpy as np
from typing import Tuple
from uuid import uuid4

import geopandas as gpd
import pandas as pd
import plotly.express as px
import pyproj
import requests
import streamlit as st
from archetypal import UmiTemplateLibrary
from dotenv import load_dotenv
from ladybug.epw import EPW

from app.app_utils import (
    filter_templates,
    load_lib,
    template_categories,
    template_climate_zones,
    load_space,
)

st.set_page_config(
    page_title="UBEM",
    layout="wide",
    initial_sidebar_state="expanded",
)
load_dotenv()

if "job" not in st.session_state:
    st.session_state.job = {}

BACKEND_URL = os.getenv("BACKEND_URL")


@st.cache_resource
def load_gis_file(file) -> Tuple[gpd.GeoDataFrame, list[str]]:
    gdf: gpd.GeoDataFrame = gpd.read_file(file)
    # gdf.to_crs(pyproj.CRS.from_epsg(4326), inplace=True)
    columns = gdf.columns.tolist()
    return gdf, columns


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


def render_gis_upload():
    gis_file = st.file_uploader("Upload GIS file")
    if gis_file is not None:
        gdf, columns = load_gis_file(gis_file)
        l, r = st.columns(2, gap="medium")
        with l:
            # find the first column with "ID" in it, otherwise, return 0
            id_col_ix = columns.index(
                next((col for col in columns if "ID" in col), columns[0])
            )
            template_col_ix = columns.index(
                next((col for col in columns if "TEMPLATE" in col.upper()), columns[0])
            )
            height_col_ix = columns.index(
                next((col for col in columns if "HEIGHT" in col.upper()), columns[0])
            )
            wwr_col_ix = columns.index(
                next(
                    (
                        col
                        for col in columns
                        if "WWR" in col.upper() or "WINDOW" in col.upper()
                    ),
                    columns[0],
                )
            )
            id_col = st.selectbox(
                "Select ID column (unique)",
                columns,
                index=id_col_ix,
            )
            template_name_col = st.selectbox(
                "Select template name column",
                columns,
                index=template_col_ix,
            )
            height_col = st.selectbox(
                "Select height column (m)",
                columns,
                index=height_col_ix,
            )
            wwr_col = st.selectbox(
                "Select wwr column (0-1)",
                columns,
                index=wwr_col_ix,
            )
            col_names = {
                "id_col": id_col,
                "template_name_col": template_name_col,
                "height_col": height_col,
                "wwr_col": wwr_col,
            }
        with r:
            color_by_column = st.selectbox("Select column to color by", columns)
            render_map(gdf, color=color_by_column)
        return gdf, col_names
    else:
        return None, None


def render_map(gdf: gpd.GeoDataFrame, color: str):
    gdf = gdf.to_crs(pyproj.CRS.from_epsg(4326))
    fig = px.choropleth(
        gdf,
        geojson=gdf.geometry,
        locations=gdf.index,
        color=color,
        fitbounds="locations",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        geo_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(title_text="GIS Map")
    # place legend below graph
    fig.update_layout(legend=dict(yanchor="top", y=-0.05, xanchor="left", x=0.01))
    st.plotly_chart(fig, use_container_width=True)


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


def render_hdf_templates():
    templates, schedules = load_lib()
    l, r = st.columns(2, gap="medium")
    with l:
        czs = st.multiselect("Climate Zone", template_climate_zones(templates))
    with r:
        cats = st.multiselect("Category", template_categories(templates))
    templates = filter_templates(templates, czs, cats)
    st.dataframe(templates)


@st.cache_resource
def load_utl(file) -> UmiTemplateLibrary:
    os.makedirs("data/temp", exist_ok=True)
    fname = f"data/temp/{file.name}.json"
    with open(fname, "wb") as f:
        f.write(file.read())
    utl = UmiTemplateLibrary.open(fname)
    return utl


def render_template_upload(template_names: list[str] = None):
    template_type = st.radio(
        "Select template source", ["Manual", "UMI Template Library"]
    )
    if template_type == "Manual":
        space_config = load_space()
        col_count = 3
        columns = [[] for _ in range(col_count)]
        counter = 0
        for param, param_def in space_config.items():
            if (
                param
                in [
                    "width",
                    "height",
                    "perim_depth",
                    "core_depth",
                    "roof_2_footprint",
                    "ground_2_footprint",
                    "orientation",
                    "wwr",
                ]
                or "shading" in param
            ):
                continue
            columns[counter].append((param, param_def))

            counter = (counter + 1) % len(columns)
        # st.write(space_config)
        if template_names is None:
            st.info("You must upload a GIS file first in Manual Mode.")
        else:
            template_defs = []
            errors = False
            for template_name in template_names:
                template_config = {}
                with st.expander(template_name):
                    cols = st.columns(len(columns))
                    for i, col_group in enumerate(columns):
                        with cols[i]:
                            for param, param_def in col_group:
                                if param_def["mode"] == "Continuous":
                                    template_config[param] = st.number_input(
                                        param,
                                        float(param_def["min"]),
                                        float(param_def["max"]),
                                        key=f"template_{template_name}_param_{param}",
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
                                        raise ValueError(
                                            f"Could not find labels for param: {param}"
                                        )
                                    assert (
                                        len(labels) == param_def["option_count"]
                                    ), f"Labels and option count do not match for param: {param}"
                                    template_config[param] = st.selectbox(
                                        param,
                                        list(range(param_def["option_count"])),
                                        format_func=lambda x: labels[x],
                                        key=f"template_{template_name}_param_{param}",
                                    )
                if (
                    template_config["HeatingSetpoint"]
                    > template_config["CoolingSetpoint"]
                ):
                    st.error(
                        f"Template {template_name} has heating setpoint > cooling setpoint"
                    )
                    errors = True
                template_defs.append(template_config)
            if errors:
                st.error("At least one template has an error!")
                return None
            template_features_df = pd.DataFrame(template_defs, index=template_names)
            template_features_df = template_features_df[
                [
                    col
                    for col in space_config.keys()
                    if col in template_features_df.columns
                ]
            ]
            st.dataframe(template_features_df)
        return None
    else:
        template_file = st.file_uploader("Upload UBEM Template file")
        if template_file:
            utl = load_utl(template_file)
            return utl
        else:
            return None


def render_submission(gdf, epw, utl, col_names):
    resources = [gdf, epw, utl]
    if all([resource is not None for resource in resources]):
        should_submit = st.button(
            "Submit UBEM", type="primary", use_container_width=True
        )
        if should_submit:
            uuid = uuid4()
            st.session_state.job = {
                "uuid": str(uuid),
            }
            tmp = f"data/temp/frontend/{uuid}"
            os.makedirs(tmp, exist_ok=True)
            gdf.to_file(f"{tmp}/gis.geojson", driver="GeoJSON")
            epw.save(f"{tmp}/epw.epw")
            utl.save(f"{tmp}/utl.json")
            # send the files to the backend
            files = {
                "gis_file": open(f"{tmp}/gis.geojson", "rb"),
                "epw_file": open(f"{tmp}/epw.epw", "rb"),
                "utl_file": open(f"{tmp}/utl.json", "rb"),
            }
            query_params = col_names.copy()
            query_params["uuid"] = uuid
            response = requests.post(
                f"{BACKEND_URL}/ubem", files=files, params=query_params
            )
            if response.status_code != 200:
                st.error(f"Error {response.status_code}")
            else:
                data = response.json()
                job_id = data["id"]
                st.session_state.job["runpod_id"] = job_id
                st.toast(f"UBEM job submitted!")
    else:
        st.info("You must upload all resources (GIS, EPW, Templates) first.")


def main():
    st.title("UBEM")

    st.divider()
    st.header("GIS")
    gdf, col_names = render_gis_upload()

    st.divider()
    st.header("EPW")
    epw = render_epw_upload()

    st.divider()
    st.header("Templates")
    utl = render_template_upload(
        template_names=(
            None if gdf is None else gdf[col_names["template_name_col"]].unique()
        )
    )

    st.divider()
    st.header("Submit UBEM")
    render_submission(gdf, epw, utl, col_names)
    st.divider()
    st.header("Results")

    if "runpod_id" in st.session_state.job and "annual" not in st.session_state.job:
        should_check_job_status = st.button(
            "Check job status", type="primary", use_container_width=True
        )
        if should_check_job_status:
            job_id = st.session_state.job["runpod_id"]
            response = requests.get(f"{BACKEND_URL}/ubem/status/{job_id}")
            if response.status_code != 200:
                st.error(f"Error {response.status_code}")
            else:
                data = response.json()
                if data["status"] == "COMPLETED":
                    for key, value in data["output"].items():
                        df = pd.DataFrame.from_dict(value, orient="tight")
                        st.session_state.job[key] = df
                else:
                    st.toast(f"Job status: {data['status']}")
    if "annual" in st.session_state.job:
        annual = st.session_state.job["annual"]
        gdf_with_results = gdf.merge(annual, left_index=True, right_on="building_id")
        # TODO: bad floor count config
        gdf_with_results["AREA"] = gdf_with_results.geometry.area
        gdf_with_results["FLOORS"] = np.ceil(gdf_with_results["HEIGHT"].values / 4)
        gdf_with_results["GFA"] = gdf_with_results["AREA"] * gdf_with_results["FLOORS"]
        end_uses = ["Heating", "Cooling"]
        colors = ["#ff6961", "#779ecb"]
        for end_use in end_uses:
            gdf_with_results[f"{end_use} Energy"] = (
                gdf_with_results[end_use] * gdf_with_results["GFA"]
            )

        df_energy_and_gfa = gdf_with_results[
            [
                col_names["template_name_col"],
                *[f"{end_use} Energy" for end_use in end_uses],
                "GFA",
            ]
        ]
        df_by_template = df_energy_and_gfa.groupby(col_names["template_name_col"]).sum()
        for end_use in end_uses:
            df_by_template[f"{end_use} Normalized"] = (
                df_by_template[f"{end_use} Energy"] / df_by_template["GFA"]
            )
        melted_normalized = df_by_template.reset_index(
            col_names["template_name_col"]
        ).melt(
            id_vars=[col_names["template_name_col"]],
            value_vars=[f"{end_use} Normalized" for end_use in end_uses],
            var_name="End Use",
            value_name="Energy (kWh/m2)",
        )
        melted_unnormalized = df_by_template.reset_index(
            col_names["template_name_col"]
        ).melt(
            id_vars=[col_names["template_name_col"]],
            value_vars=[f"{end_use} Energy" for end_use in end_uses],
            var_name="End Use",
            value_name="Energy (kWh)",
        )

        by_templates_tab, by_buildings_tab = st.tabs(["Templates", "Buildings"])
        with by_templates_tab:
            normalize_plot = st.toggle("Normalize plot")
            fig = px.bar(
                melted_normalized if normalize_plot else melted_unnormalized,
                x=col_names["template_name_col"],
                y="Energy (kWh/m2)" if normalize_plot else "Energy (kWh)",
                color="End Use",
                labels={col_names["template_name_col"]: "Template"},
                color_discrete_map={
                    f"{end_use} {'Normalized' if normalize_plot else 'Energy'}": color
                    for end_use, color in zip(end_uses, colors)
                },
            )
            st.plotly_chart(fig, use_container_width=True)
        with by_buildings_tab:
            color_by_column = st.selectbox(
                "Select column to color by", ["Heating", "Cooling"]
            )

            render_map(gdf_with_results, color=color_by_column)


main()
