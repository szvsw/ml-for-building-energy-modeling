import streamlit as st
import os
import pandas as pd
import geopandas as gpd
from typing import Tuple
import plotly.express as px
import pyproj
from ladybug.epw import EPW
from archetypal import UmiTemplateLibrary

from umi.ubem import UBEM

from app.app_utils import (
    load_lib,
    template_climate_zones,
    template_categories,
    filter_templates,
)

st.set_page_config(
    page_title="UBEM",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_gis_file(file) -> Tuple[gpd.GeoDataFrame, list[str]]:
    gdf: gpd.GeoDataFrame = gpd.read_file(file)
    gdf.to_crs(pyproj.CRS.from_epsg(4326), inplace=True)
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
            id_col = st.selectbox("Select ID column (unique)", columns)
            template_name_col = st.selectbox("Select template name column", columns)
            height_col = st.selectbox("Select height column (m)", columns)
            wwr_col = st.selectbox("Select wwr column (0-1)", columns)
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


def render_epw_upload():
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


def render_template_upload():
    template_file = st.file_uploader("Upload UBEM Template file")
    if template_file:
        utl = load_utl(template_file)
        return utl
    else:
        return None


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
    utl = render_template_upload()
    resources = [gdf, epw, utl]
    # if all([resource is not None for resource in resources]):
    #     ubem = UBEM(
    #         gdf=gdf,
    #         **col_names,
    #         epw=epw,
    #         template_lib=utl,
    #         sensor_spacing=3,
    #         shoebox_width=3,
    #         floor_to_floor_height=4,
    #         perim_offset=4,
    #         shoebox_gen_type="edge_unshaded",
    #     )
    #     st.write(ubem.gis_features_df)
    # bytes_data = uploaded_file.getvalue()

    #     # Sending the file to the FastAPI server
    #     response = requests.post(
    #         "http://localhost:8000/uploadfile/",
    #         files={"file": (uploaded_file.name, bytes_data)}
    #     )

    #     if response.status_code == 200:
    #         st.success("File uploaded successfully.")
    #         st.json(response.json())
    #     else:
    #         st.error("Failed to upload file.")


main()
