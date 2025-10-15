"""
Streamlit implementation of the interactive well and acreage explorer.

This module ports the original R/Shiny implementation to Python while
replacing data wrangling with Polars and spatial work with GeoPandas.
Polars executes most transformations using a vectorised Rust backend,
which proved to be more resilient than the former R/data.table stack.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import functools
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import geopandas as gpd
import pandas as pd
import polars as pl
import pydeck as pdk
import streamlit as st
from shapely.geometry import Point

try:  # Optional dependency – only required when a live DB is available
    import oracledb  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    oracledb = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_PATH = Path(os.getenv("APP_BASE_PATH", Path.cwd()))
CACHE_PATH = BASE_PATH / "cache"
CACHE_PATH.mkdir(parents=True, exist_ok=True)

PROCESSED_PARQUET = Path(
    os.getenv(
        "APP_PROCESSED_PARQUET",
        BASE_PATH / "processed_app_data_vMERGED_FINAL_v24_DB_sticks_latlen.parquet",
    )
)
WOODMACK_COVERAGE_EXCEL = Path(
    os.getenv("APP_WOODMACK_COVERAGE_XLSX", BASE_PATH / "Woodmack.Coverage.2024.xlsx")
)
PLAY_SUBPLAY_PATH = Path(
    os.getenv("APP_SUBPLAY_SHAPEFILES", BASE_PATH / "SubplayShapefile")
)
COMPANY_SHAPEFILES_PATH = Path(
    os.getenv("APP_COMPANY_SHAPEFILES", BASE_PATH / "Shapefile")
)

DEFAULT_CONNECTION = {
    "user": os.getenv("ORACLE_USER", "WOODMAC"),
    "password": os.getenv("ORACLE_PASSWORD", "c0pp3r"),
    "dsn": os.getenv("ORACLE_DSN", "GDC_LINK.geologic.com"),
    "mode": oracledb.AUTH_MODE_DEFAULT if oracledb else None,
}

# Conversion factors
E3M3_TO_MCF = 35.3147
M3_TO_BBL = 6.28981
AVG_DAYS_PER_MONTH = 30.4375
MCF_PER_BOE = 6


# ---------------------------------------------------------------------------
# Dataclasses and helpers
# ---------------------------------------------------------------------------
@dataclass
class SpatialLayer:
    """Container describing a vector layer loaded from disk."""

    name: str
    data: gpd.GeoDataFrame


def standardize_uwi(values: Iterable[str]) -> pl.Series:
    """Normalise UWI identifiers to uppercase alphanumerics."""

    return (
        pl.Series(values)
        .cast(pl.Utf8, strict=False)
        .str.replace_all(r"[^A-Za-z0-9]", "")
        .str.to_uppercase()
    )


def clean_column_names(frame: pl.DataFrame) -> pl.DataFrame:
    """Replicate the behaviour of `make.names` from the R code."""

    cleaned = []
    for name in frame.columns:
        safe = name.replace(" ", "_")
        safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in safe)
        safe = "_".join(filter(None, safe.split("_")))
        cleaned.append(safe.upper())
    return frame.rename(dict(zip(frame.columns, cleaned)))


def _read_excel(path: Path, sheet: str) -> pl.DataFrame:
    if not path.exists():
        st.warning(f"Excel file not found: {path}")
        return pl.DataFrame()

    df = pd.read_excel(path, sheet_name=sheet)
    return pl.from_pandas(df)


def prepare_filter_choices(values: pl.Series) -> list[str]:
    if values.is_null().all() or values.len() == 0:
        return []
    unique = (
        values.cast(pl.Utf8, strict=False)
        .drop_nulls()
        .filter(pl.col(values.name) != "")
        .unique()
        .sort()
    )
    return unique.to_list()


@functools.cache
def load_wells_from_parquet() -> pl.DataFrame:
    if not PROCESSED_PARQUET.exists():
        return pl.DataFrame()
    return pl.read_parquet(PROCESSED_PARQUET)


def connect_to_db() -> Optional["oracledb.Connection"]:
    if oracledb is None:
        return None

    try:
        return oracledb.connect(
            user=DEFAULT_CONNECTION["user"],
            password=DEFAULT_CONNECTION["password"],
            dsn=DEFAULT_CONNECTION["dsn"],
            mode=DEFAULT_CONNECTION["mode"],
        )
    except Exception as exc:  # pragma: no cover - network / credentials
        st.error(f"Unable to connect to Oracle: {exc}")
        return None


@st.cache_data(show_spinner="Loading well master data...")
def load_wells_from_db() -> pl.DataFrame:
    connection = connect_to_db()
    if connection is None:
        return pl.DataFrame()

    query = """
        SELECT W.UWI,
               W.GSL_UWI,
               W.SURFACE_LATITUDE,
               W.SURFACE_LONGITUDE,
               W.BOTTOM_HOLE_LATITUDE,
               W.BOTTOM_HOLE_LONGITUDE,
               W.GSL_FULL_LATERAL_LENGTH,
               W.ABANDONMENT_DATE,
               W.WELL_NAME,
               W.CURRENT_STATUS,
               W.OPERATOR AS OPERATOR_CODE,
               W.CONFIDENTIAL_TYPE,
               P.STRAT_UNIT_ID,
               W.SPUD_DATE,
               PFS.FIRST_PROD_DATE,
               W.FINAL_TD,
               W.PROVINCE_STATE,
               W.COUNTRY,
               FL.FIELD_NAME
        FROM WELL W
        LEFT JOIN PDEN P ON W.GSL_UWI = P.GSL_UWI
        LEFT JOIN FIELD FL ON W.ASSIGNED_FIELD = FL.FIELD_ID
        LEFT JOIN PDEN_FIRST_SUM PFS ON W.GSL_UWI = PFS.GSL_UWI
        WHERE W.SURFACE_LATITUDE IS NOT NULL
          AND W.SURFACE_LONGITUDE IS NOT NULL
          AND (W.ABANDONMENT_DATE IS NULL OR W.ABANDONMENT_DATE > SYSDATE - (365*20))
    """

    with contextlib.closing(connection) as conn:
        frame = pd.read_sql(query, conn)

    wells = pl.from_pandas(frame)
    if "UWI" in wells.columns:
        wells = wells.with_columns(standardize_uwi(wells["UWI"]).alias("UWI_STD"))
    if "GSL_UWI" in wells.columns:
        wells = wells.with_columns(standardize_uwi(wells["GSL_UWI"]).alias("GSL_UWI_STD"))
    return wells


def load_operator_lookup() -> pl.DataFrame:
    df = _read_excel(WOODMACK_COVERAGE_EXCEL, "Operator")
    if df.is_empty():
        return df
    df = clean_column_names(df)
    required = {"OPERATOR", "GSL_PARENT_BA_NAME"}
    if not required.issubset(df.columns):
        st.warning("Operator coverage sheet is missing expected columns.")
        return pl.DataFrame()
    return (
        df.select(
            pl.col("OPERATOR").cast(pl.Utf8).alias("WoodmackJoinOperatorCode"),
            pl.col("GSL_PARENT_BA_NAME").cast(pl.Utf8).alias("OperatorName"),
        )
        .drop_nulls()
        .unique(subset=["WoodmackJoinOperatorCode"])
    )


def merge_well_metadata(wells: pl.DataFrame) -> pl.DataFrame:
    if wells.is_empty():
        return wells

    operators = load_operator_lookup()
    if not operators.is_empty():
        wells = wells.join(
            operators,
            left_on="OPERATOR_CODE",
            right_on="WoodmackJoinOperatorCode",
            how="left",
        )
    rename_map = {
        "SURFACE_LATITUDE": "SurfaceLatitude",
        "SURFACE_LONGITUDE": "SurfaceLongitude",
        "BOTTOM_HOLE_LATITUDE": "BH_Latitude",
        "BOTTOM_HOLE_LONGITUDE": "BH_Longitude",
        "GSL_FULL_LATERAL_LENGTH": "LateralLength",
        "ABANDONMENT_DATE": "AbandonmentDate",
        "WELL_NAME": "WellName",
        "CURRENT_STATUS": "CurrentStatus",
        "OPERATOR_CODE": "OperatorCode",
        "STRAT_UNIT_ID": "StratUnitID",
        "SPUD_DATE": "SpudDate",
        "FIRST_PROD_DATE": "FirstProdDate",
        "FINAL_TD": "FinalTD",
        "PROVINCE_STATE": "ProvinceState",
        "COUNTRY": "Country",
        "FIELD_NAME": "FieldName",
        "CONFIDENTIAL_TYPE": "ConfidentialType",
    }
    for old, new in rename_map.items():
        if old in wells.columns:
            wells = wells.rename({old: new})
    return wells


def load_spatial_layers(folder: Path) -> list[SpatialLayer]:
    if not folder.exists():
        return []

    layers: list[SpatialLayer] = []
    for shp in folder.glob("*.shp"):
        try:
            data = gpd.read_file(shp)
            data = data.to_crs(4326)
        except Exception as exc:  # pragma: no cover - shapefile issues
            st.warning(f"Unable to read {shp.name}: {exc}")
            continue
        layer_name = shp.stem.replace("_", " ").title()
        layers.append(SpatialLayer(layer_name, data))
    return layers


def wells_to_geodataframe(wells: pl.DataFrame) -> gpd.GeoDataFrame:
    if wells.is_empty() or "SurfaceLatitude" not in wells.columns:
        return gpd.GeoDataFrame(columns=wells.columns)

    df = wells.to_pandas()
    geometry = [Point(xy) for xy in zip(df["SurfaceLongitude"], df["SurfaceLatitude"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    return gdf


def build_map_layer(wells: gpd.GeoDataFrame, color_field: str = "OperatorName") -> pdk.Layer:
    tooltip_fields = {
        "Well": "WellName",
        "Operator": "OperatorName",
        "Formation": "Formation",
        "First Production": "FirstProdDate",
    }
    data = wells[["SurfaceLongitude", "SurfaceLatitude", *tooltip_fields.values()]].copy()
    return pdk.Layer(
        "ScatterplotLayer",
        data=data,
        get_position="[SurfaceLongitude, SurfaceLatitude]",
        get_radius=350,
        get_fill_color="[200, 30, 0, 160]",
        pickable=True,
        tooltip={"text": "\n".join(f"{key}: {{{value}}}" for key, value in tooltip_fields.items())},
    )


def render_map(wells: gpd.GeoDataFrame, selected_layers: list[SpatialLayer]) -> None:
    if wells.empty:
        st.info("No wells match the current filter criteria.")
        return

    map_layers = [build_map_layer(wells)]
    for layer in selected_layers:
        map_layers.append(
            pdk.Layer(
                "GeoJsonLayer",
                data=json.loads(layer.data.to_json()),
                stroked=True,
                filled=False,
                get_line_color="[0, 121, 191]",
                get_line_width=2,
            )
        )

    bounds = wells.total_bounds
    view_state = pdk.ViewState(
        longitude=float((bounds[0] + bounds[2]) / 2),
        latitude=float((bounds[1] + bounds[3]) / 2),
        zoom=5,
    )
    st.pydeck_chart(pdk.Deck(layers=map_layers, initial_view_state=view_state, tooltip={"html": "{Well}"}))


def filter_wells(wells: pl.DataFrame) -> pl.DataFrame:
    operators = st.multiselect(
        "Operator",
        options=prepare_filter_choices(wells.get_column("OperatorName")) if "OperatorName" in wells.columns else [],
    )
    formations = st.multiselect(
        "Formation",
        options=prepare_filter_choices(wells.get_column("Formation")) if "Formation" in wells.columns else [],
    )
    fields = st.multiselect(
        "Field",
        options=prepare_filter_choices(wells.get_column("FieldName")) if "FieldName" in wells.columns else [],
    )
    provinces = st.multiselect(
        "Province/State",
        options=prepare_filter_choices(wells.get_column("ProvinceState")) if "ProvinceState" in wells.columns else [],
    )
    date_col = wells.get_column("FirstProdDate") if "FirstProdDate" in wells.columns else None
    start, end = st.date_input(
        "First Production Date",
        value=(dt.date.today() - dt.timedelta(days=3650), dt.date.today()),
    )

    mask = pl.lit(True)
    if operators:
        mask &= pl.col("OperatorName").is_in(operators)
    if formations:
        mask &= pl.col("Formation").is_in(formations)
    if fields:
        mask &= pl.col("FieldName").is_in(fields)
    if provinces:
        mask &= pl.col("ProvinceState").is_in(provinces)
    if date_col is not None:
        mask &= pl.col("FirstProdDate").is_between(start, end)

    filtered = wells.filter(mask)
    st.caption(f"{filtered.height:,} wells match the current filters")
    return filtered


def load_base_well_data() -> pl.DataFrame:
    cached = load_wells_from_parquet()
    if not cached.is_empty():
        return cached
    db_data = load_wells_from_db()
    if db_data.is_empty():
        st.warning("No well data available from cache or database.")
    return db_data


def main() -> None:
    st.set_page_config(page_title="Well & Acreage Explorer", layout="wide")
    st.title("Interactive Well and Acreage Map Application")
    st.write(
        """
        This Streamlit application is a Python reimplementation of the
        legacy Shiny dashboard. Heavy data manipulation is handled by
        Polars, which executes on a highly optimised Rust engine to
        prevent the crashes we used to experience with single-threaded
        R pipelines.
        """
    )

    wells = merge_well_metadata(load_base_well_data())
    if wells.is_empty():
        st.stop()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("Map Filters")
        filtered = filter_wells(wells)
    with col2:
        st.header("Spatial Layers")
        available_layers = load_spatial_layers(PLAY_SUBPLAY_PATH) + load_spatial_layers(
            COMPANY_SHAPEFILES_PATH
        )
        selected = st.multiselect(
            "Overlay layers",
            options=[layer.name for layer in available_layers],
        )
        selected_layers = [layer for layer in available_layers if layer.name in selected]

    render_map(wells_to_geodataframe(filtered), selected_layers)

    st.header("Data Preview")
    st.dataframe(filtered.head(200).to_pandas())

    st.download_button(
        "Download filtered wells as CSV",
        data=filtered.write_csv().encode(),
        file_name="filtered_wells.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
