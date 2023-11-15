import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import streamlit as st
import requests
from uuid import uuid4
from utils.constants import SCHEDULE_PATHS

from app.app_utils import (
    load_lib,
    load_schedule,
    render_epw_upload,
    load_space,
    make_controls_columns,
    render_controls,
    load_template_defaults,
)

BACKEND_URL = os.getenv("BACKEND_URL")

st.set_page_config(
    page_title="Shoebox",
    initial_sidebar_state="expanded",
)
if "job" not in st.session_state:
    st.session_state["job"] = {}
if "building_results" not in st.session_state:
    st.session_state["building_results"] = {}


def main():
    st.title("Single Building Design Analyzer")
    epw_tab, control_tab, results_tab = st.tabs(
        ["EPW", "Building Definition", "Results"]
    )
    with epw_tab:
        st.markdown("## EPW")
        epw = render_epw_upload()

    with control_tab:
        space_config = load_space()
        template_defaults = load_template_defaults()
        columns = make_controls_columns(
            space_config=space_config,
            skip_params=[
                "width",
                "orientation",
                "height",
                "perim_depth",
                "core_depth",
                "roof_2_footprint",
                "ground_2_footprint",
                "wwr",
            ],
        )
        template_config = {}
        with st.expander("Building Geometry"):
            l, r = st.columns(2)
            with l:
                template_config["n_floors"] = st.number_input(
                    "Number of Floors",
                    min_value=1,
                    max_value=20,
                    value=template_defaults["n_floors"],
                    step=1,
                )
            with r:
                template_config["height"] = st.number_input(
                    "Floor-to-Floor Height [m]",
                    min_value=3.0,
                    max_value=5.0,
                    value=template_defaults["height"],
                    step=0.1,
                )
            with l:
                template_config["building_width"] = st.number_input(
                    "Building Width [m]",
                    min_value=5,
                    max_value=100,
                    value=template_defaults["building_width"],
                    step=1,
                )
            with r:
                template_config["building_length"] = st.number_input(
                    "Building Length [m]",
                    min_value=5,
                    max_value=100,
                    value=template_defaults["building_length"],
                    step=1,
                )
            for orient, col in zip(["S", "E", "N", "W"], [l, r, l, r]):
                with col:
                    template_config[f"wwr_{orient}"] = st.number_input(
                        f"Window-to-Wall Ratio ({orient})",
                        min_value=0.10,
                        max_value=0.90,
                        value=template_defaults[f"wwr_{orient}"],
                        step=0.01,
                    )
        with st.expander("Non-Geometric Parameters"):
            temp_key_formatter = lambda x: f"shoebox_param_{x}"
            template_config = render_controls(
                columns,
                template_config,
                key_formatter=temp_key_formatter,
                defaults=template_defaults,
            )

            st.divider()
            st.markdown("## Schedules")
            ix = st.selectbox(
                "Select schedule index",
                list(range(16)),
                key=f"shoebox_schedule_ix",
            )
            scheds = load_schedule(ix)
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

        st.divider()
        st.markdown("## Job Submission")
        if epw is not None:
            run_name = st.text_input(
                "Run Name", key="shoebox_run_name", value="Baseline"
            )
            should_submit = st.button(
                "Submit", type="primary", use_container_width=True
            )
            if should_submit:
                url = f"{BACKEND_URL}/building"
                uuid = uuid4()
                st.session_state.job = {
                    "uuid": str(uuid),
                }
                tmp = f"data/temp/frontend/{uuid}"
                os.makedirs(tmp, exist_ok=True)
                epw.save(f"{tmp}/epw.epw")
                template_data = {
                    "features": template_config,
                    "schedules": scheds.tolist(),
                }
                with open(f"{tmp}/template.json", "w") as f:
                    json.dump(template_data, f)
                # TODO: template file should be in body or query params
                files = {
                    "epw_file": open(f"{tmp}/epw.epw", "rb"),
                    "template_file": open(f"{tmp}/template.json", "rb"),
                }
                query_params = {}
                query_params["uuid"] = uuid

                response = requests.post(
                    url,
                    files=files,
                    params=query_params,
                )
                if response.status_code == 200:
                    st.session_state.job = {}
                    run_results = {}
                    run_data = response.json()
                    annual = pd.DataFrame.from_dict(run_data["annual"], orient="tight")
                    monthly = pd.DataFrame.from_dict(
                        run_data["monthly"], orient="tight"
                    )
                    shoeboxes = pd.DataFrame.from_dict(
                        run_data["shoeboxes"], orient="tight"
                    )
                    run_results["annual"] = annual
                    run_results["monthly"] = monthly
                    run_results["shoeboxes"] = shoeboxes
                    st.session_state.building_results[run_name] = run_results
                    st.success("Head over to results to see the data!")
                else:
                    st.session_state.job = {}
                    # TODO: better error
                    st.error(f"Error submitting job. Please try again.")
        else:
            st.warning("Please upload an EPW file before submitting job.")

    with results_tab:
        if len(st.session_state.building_results) > 0:
            st.markdown("## Results")
            monthly_results = []
            annual_results = []
            for run_name, results in st.session_state.building_results.items():
                data = results["monthly"]
                data.index = [run_name]
                monthly_results.append(data)
                data = results["annual"]
                data.index = [run_name]
                annual_results.append(data)
            monthly_results = pd.concat(monthly_results, axis=0)
            annual_results = (
                pd.concat(annual_results, axis=0)
                .reset_index(names="Version")
                .melt(
                    value_name="Energy Intensity (kWh/m2)",
                    var_name="End Use",
                    id_vars=["Version"],
                )
            )
            monthly_results = (
                monthly_results.stack()
                .reset_index(names=["Version", "Month"])
                .melt(
                    id_vars=["Version", "Month"],
                    var_name="End Use",
                    value_name="Energy Intensity (kWh/m2)",
                )
            ).sort_values(["Version", "End Use", "Month"])

            plot_resolution = st.selectbox(
                "Select plot resolution", ["Annual", "Monthly"], index=0
            )
            if plot_resolution == "Monthly":
                fig = px.line(
                    monthly_results,
                    x="Month",
                    y="Energy Intensity (kWh/m2)",
                    color="End Use",
                    line_dash="Version",
                    color_discrete_map={
                        "Heating": "#EF553B",
                        "Cooling": "#636EFA",
                    },
                )
                fig.update_layout(
                    legend=dict(
                        yanchor="top", y=-0.2, xanchor="left", x=0.01, orientation="h"
                    ),
                )
                st.plotly_chart(fig, use_container_width=True)
            elif plot_resolution == "Annual":
                fig = px.bar(
                    annual_results,
                    x="Version",
                    y="Energy Intensity (kWh/m2)",
                    color="End Use",
                    color_discrete_map={
                        "Heating": "#EF553B",
                        "Cooling": "#636EFA",
                    },
                )
                fig.update_layout(legend=dict(orientation="h"))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please submit a job to see results.")
    with open("template-defaults.json", "w") as f:
        json.dump(template_config, f, indent=4)


main()
