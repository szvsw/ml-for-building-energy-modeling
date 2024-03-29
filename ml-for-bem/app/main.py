import json
import os
import numpy as np
from typing import Tuple, Literal, Union
from uuid import uuid4

import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
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
    load_schedule,
    make_controls_columns,
    render_controls,
    render_epw_upload,
    load_template_defaults,
)
from utils.constants import SCHEDULE_PATHS

st.set_page_config(
    page_title="UBEM",
    layout="wide",
    initial_sidebar_state="expanded",
)
load_dotenv()

if "job" not in st.session_state:
    st.session_state.job = {}
if "results" not in st.session_state:
    st.session_state.results = {}

# TODO: make pydantic settings
BACKEND_URL = os.getenv("BACKEND_URL")


@st.cache_resource
def load_gis_file(file) -> Tuple[gpd.GeoDataFrame, list[str]]:
    gdf: gpd.GeoDataFrame = gpd.read_file(file)
    # gdf.to_crs(pyproj.CRS.from_epsg(4326), inplace=True)
    columns = gdf.columns.tolist()
    return gdf, columns


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
        columns = make_controls_columns(space_config=space_config)
        # st.write(space_config)
        if template_names is None:
            st.warning("You must upload a GIS file first in Manual Mode.")
            return None, None
        else:
            template_defs = []
            template_schedules = []
            template_defaults = load_template_defaults()
            errors = False
            for template_name in template_names:
                template_config = {}
                with st.expander(template_name):
                    st.markdown("#### Building Parameters")

                    temp_key_formatter = lambda x: f"template_{template_name}_param_{x}"
                    template_config = render_controls(
                        columns,
                        template_config,
                        key_formatter=temp_key_formatter,
                        defaults=template_defaults,
                    )
                    st.divider()
                    st.markdown("#### Schedules")
                    ix = st.selectbox(
                        "Select schedule index",
                        list(range(16)),
                        key=f"template_{template_name}_schedule_ix",
                    )
                    scheds = load_schedule(ix)
                    template_schedules.append(scheds)
                    fig = go.Figure()
                    # add a trace for each schedule, which are along the first axis
                    for i in range(scheds.shape[0]):
                        plot_range = 24 * 14
                        fig.add_trace(
                            go.Scatter(
                                x=np.arange(plot_range),
                                y=scheds[i, :plot_range],
                                name=SCHEDULE_PATHS[i][-1].split("A")[0].split("S")[0],
                            )
                        )
                    fig.update_layout(
                        title="Schedules",
                        xaxis_title="Hour of Year",
                        yaxis_title="Schedule Value [0-1]",
                    )
                    st.plotly_chart(fig, use_container_width=True)
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
            template_schedules = np.array(template_schedules, dtype=np.float32)
            st.dataframe(template_features_df)
            return "ml", (template_features_df, template_schedules)
    else:
        template_file = st.file_uploader("Upload UBEM Template file")
        if template_file:
            utl = load_utl(template_file)
            return "utl", utl
        else:
            return None, None


def render_submission(
    gdf: gpd.GeoDataFrame,
    epw: EPW,
    lib: Union[UmiTemplateLibrary, Tuple[pd.DataFrame, np.ndarray]],
    lib_mode: Literal["utl", "ml"],
    col_names: dict[str, str],
):
    resources = [gdf, epw, lib]
    if all([resource is not None for resource in resources]):
        run_name = st.text_input("Run Name", value="Baseline")
        should_submit = st.button(
            "Submit UBEM", type="primary", use_container_width=True
        )
        if should_submit:
            url = f"{BACKEND_URL}/ubem"
            uuid = uuid4()
            st.session_state.job = {
                "uuid": str(uuid),
            }
            tmp = f"data/temp/frontend/{uuid}"
            os.makedirs(tmp, exist_ok=True)
            gdf.to_file(f"{tmp}/gis.geojson", driver="GeoJSON")
            epw.save(f"{tmp}/epw.epw")
            # send the files to the backend
            if lib_mode == "utl":
                lib.save(f"{tmp}/utl.json")
            elif lib_mode == "ml":
                utl = {
                    "templates": lib[0].to_dict(orient="tight"),
                    "schedules": lib[1].tolist(),
                }
                with open(f"{tmp}/utl.json", "w") as f:
                    json.dump(utl, f)
            else:
                raise ValueError(f"Unknown lib mode: {lib_mode}")
            files = {
                "gis_file": open(f"{tmp}/gis.geojson", "rb"),
                "epw_file": open(f"{tmp}/epw.epw", "rb"),
                "utl_file": open(f"{tmp}/utl.json", "rb"),
            }
            query_params = col_names.copy()
            query_params["uuid"] = uuid
            query_params["lib_mode"] = lib_mode

            response = requests.post(
                url,
                files=files,
                params=query_params,
            )
            if response.status_code != 200:
                st.error(f"Error {response.status_code}")
            else:
                data = response.json()
                job_id = data["id"]
                st.session_state.job["runpod_id"] = job_id
                st.session_state.job["run_name"] = run_name
                st.toast(f"UBEM job submitted!")
    else:
        st.warning("You must upload all resources (GIS, EPW, Templates) first.")


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
    lib_mode, lib = render_template_upload(
        template_names=(
            None if gdf is None else gdf[col_names["template_name_col"]].unique()
        )
    )

    st.divider()
    st.header("Submit UBEM")
    render_submission(gdf=gdf, epw=epw, lib=lib, lib_mode=lib_mode, col_names=col_names)
    st.divider()
    st.header("Results")

    if "runpod_id" in st.session_state.job:
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
                    run_name = st.session_state.job["run_name"]
                    st.session_state.job = {}
                    st.session_state.results[run_name] = {}
                    for key, value in data["output"].items():
                        df = pd.DataFrame.from_dict(value, orient="tight")
                        df["run_name"] = run_name
                        st.session_state.results[run_name][key] = df
                else:
                    st.toast(f"Job status: {data['status']}")
    if len(st.session_state.results) > 0:
        end_uses = ["Heating", "Cooling"]
        colors = ["#ff6961", "#779ecb"]
        area = gdf.geometry.area
        # TODO: bad floor count config
        floors = np.ceil(gdf[col_names["height_col"]].values / 4)
        gfa = area * floors
        # TODO: make sure index alignment is correct
        for i, result in enumerate(st.session_state.results.values()):
            result["annual"]["gfa"] = gfa
            result["annual"]["template"] = gdf[col_names["template_name_col"]]
            result["annual"]["sort_ix"] = i
        annual = pd.concat(
            [results["annual"] for results in st.session_state.results.values()], axis=0
        )
        annual = annual.reset_index("building_id")
        annual_melted = annual.melt(
            id_vars=[col for col in annual if col not in end_uses],
            var_name="End Use",
            value_name="Energy Density (kWh/m2)",
        )
        annual_melted = annual_melted.sort_values("sort_ix")

        annual_melted["Energy (kWh)"] = (
            annual_melted["Energy Density (kWh/m2)"] * annual_melted["gfa"]
        )
        results_by_template = annual_melted.groupby(
            ["sort_ix", "template", "run_name", "End Use"]
        ).sum()
        results_by_template["Energy Density (kWh/m2)"] = (
            results_by_template["Energy (kWh)"] / results_by_template["gfa"]
        )
        results_by_template = results_by_template.reset_index()
        results_by_template = results_by_template.sort_values("sort_ix")
        scenarios_tab, templates_tab = st.tabs(["Scenarios", "Templates"])
        with scenarios_tab:
            fig = px.bar(
                results_by_template,
                x="run_name",
                y="Energy (kWh)",
                color="End Use",
                color_discrete_map={
                    end_use: color for end_use, color in zip(end_uses, colors)
                },
                hover_name="template",
            )
            st.plotly_chart(fig, use_container_width=True)
        with templates_tab:
            l, r = st.columns(2)
            with l:
                normalize_plot = st.toggle("Normalize plot", value=True)
            with r:
                facet_by_template = st.toggle("Facet by template", value=True)
            y_axis = "Energy Density (kWh/m2)" if normalize_plot else "Energy (kWh)"
            fig = px.bar(
                results_by_template,
                x="run_name" if facet_by_template else "template",
                y=y_axis,
                color="End Use",
                facet_col="template" if facet_by_template else "run_name",
                facet_col_wrap=2,
                labels={"template": "Template", "run_name": "Scenario"},
                color_discrete_map={
                    end_use: color for end_use, color in zip(end_uses, colors)
                },
                height=800,
            )
            st.plotly_chart(fig, use_container_width=True)
        return

        gdf_with_results = gdf.merge(annual, left_index=True, right_on="building_id")
        # TODO: bad floor count config
        gdf_with_results["AREA"] = gdf_with_results.geometry.area
        gdf_with_results["FLOORS"] = np.ceil(gdf_with_results["HEIGHT"].values / 4)
        gdf_with_results["GFA"] = gdf_with_results["AREA"] * gdf_with_results["FLOORS"]
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
