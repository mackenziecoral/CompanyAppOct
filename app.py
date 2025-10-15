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
from typing import Iterable, Optional, Sequence

import geopandas as gpd
import pandas as pd
import polars as pl
import pydeck as pdk
import streamlit as st
from shapely.geometry import Point

import plotly.graph_objects as go

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

MONTH_ABBREVIATIONS = [
    "JAN",
    "FEB",
    "MAR",
    "APR",
    "MAY",
    "JUN",
    "JUL",
    "AUG",
    "SEP",
    "OCT",
    "NOV",
    "DEC",
]
MONTH_TO_NUMBER = {abbr: idx for idx, abbr in enumerate(MONTH_ABBREVIATIONS, start=1)}
PRODUCT_TYPE_CHOICES = ["OIL", "CND", "GAS", "BOE"]


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


def ensure_standard_identifiers(frame: pl.DataFrame) -> pl.DataFrame:
    """Guarantee that standardised UWI columns exist for downstream joins."""

    updates = []
    if "UWI" in frame.columns and "UWI_STD" not in frame.columns:
        updates.append(standardize_uwi(frame["UWI"]).alias("UWI_STD"))
    if "GSL_UWI" in frame.columns and "GSL_UWI_STD" not in frame.columns:
        updates.append(standardize_uwi(frame["GSL_UWI"]).alias("GSL_UWI_STD"))
    if updates:
        frame = frame.with_columns(updates)
    return frame


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


def chunked(sequence: Sequence[str], size: int) -> Iterable[list[str]]:
    """Yield successive chunks from *sequence* of length *size*."""

    for idx in range(0, len(sequence), size):
        yield list(sequence[idx : idx + size])


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
        return ensure_standard_identifiers(cached)
    db_data = load_wells_from_db()
    if db_data.is_empty():
        st.warning("No well data available from cache or database.")
    return ensure_standard_identifiers(db_data)


@st.cache_data(show_spinner="Fetching monthly production...")
def load_production_for_uwis(uwis: tuple[str, ...]) -> pl.DataFrame:
    """Retrieve monthly production for the supplied wells."""

    uwis = tuple(sorted({u for u in uwis if u}))
    if not uwis:
        return pl.DataFrame()

    cache_key = f"production_{hash(uwis)}.parquet"
    cache_file = CACHE_PATH / cache_key
    if cache_file.exists():
        try:
            return pl.read_parquet(cache_file)
        except Exception:  # pragma: no cover - cache corruption
            cache_file.unlink(missing_ok=True)

    connection = connect_to_db()
    if connection is None:
        return pl.DataFrame()

    month_columns = ", ".join(f"{abbr}_VOLUME" for abbr in MONTH_ABBREVIATIONS)
    query_template = f"""
        SELECT GSL_UWI,
               YEAR,
               PRODUCT_TYPE,
               ACTIVITY_TYPE,
               {month_columns}
        FROM PDEN_VOL_BY_MONTH
        WHERE ACTIVITY_TYPE = 'PRODUCTION'
          AND PRODUCT_TYPE IN ('OIL', 'CND', 'GAS')
          AND GSL_UWI IN ({{placeholders}})
    """

    frames: list[pl.DataFrame] = []
    try:
        with contextlib.closing(connection) as conn:
            with contextlib.closing(conn.cursor()) as cursor:
                for batch in chunked(uwis, 250):
                    placeholders = ", ".join(f":{idx + 1}" for idx in range(len(batch)))
                    query = query_template.format(placeholders=placeholders)
                    cursor.execute(query, tuple(batch))
                    rows = cursor.fetchall()
                    if not rows:
                        continue
                    columns = [desc[0] for desc in cursor.description]
                    frames.append(pl.DataFrame(rows, schema=columns))
    except Exception as exc:  # pragma: no cover - database errors
        st.error(f"Unable to fetch production data: {exc}")
        return pl.DataFrame()

    if not frames:
        return pl.DataFrame()

    data = pl.concat(frames, how="vertical_relaxed")
    try:
        data.write_parquet(cache_file)
    except Exception:  # pragma: no cover - filesystem issues
        pass
    return data


def reshape_production(raw: pl.DataFrame) -> pl.DataFrame:
    """Convert monthly columns to a tidy long table with conversions applied."""

    if raw.is_empty():
        return pl.DataFrame()

    frame = clean_column_names(raw)
    frame = ensure_standard_identifiers(frame)

    required = {"GSL_UWI_STD", "YEAR", "PRODUCT_TYPE"}
    if not required.issubset(frame.columns):
        return pl.DataFrame()

    month_cols = [f"{abbr}_VOLUME" for abbr in MONTH_ABBREVIATIONS if f"{abbr}_VOLUME" in frame.columns]
    if len(month_cols) != len(MONTH_ABBREVIATIONS):
        return pl.DataFrame()

    melted = frame.unpivot(
        month_cols,
        index=["GSL_UWI_STD", "YEAR", "PRODUCT_TYPE"],
        variable_name="MONTH_LABEL",
        value_name="RAW_VOLUME",
    )

    melted = melted.with_columns(
        [
            pl.col("YEAR").cast(pl.Int32, strict=False),
            pl.col("RAW_VOLUME").cast(pl.Float64, strict=False).fill_null(0.0),
            pl.col("PRODUCT_TYPE").cast(pl.Utf8, strict=False).str.to_uppercase(),
            pl.col("MONTH_LABEL").cast(pl.Utf8, strict=False).str.slice(0, 3).alias("MONTH_ABBR"),
        ]
    )
    melted = melted.with_columns(
        pl.col("MONTH_ABBR").map_elements(lambda abbr: MONTH_TO_NUMBER.get(abbr), return_dtype=pl.Int16).alias("MONTH")
    )
    melted = melted.drop_nulls(subset=["YEAR", "MONTH"])
    melted = melted.with_columns(
        pl.datetime(pl.col("YEAR"), pl.col("MONTH"), 1, 0, 0, 0).cast(pl.Date).alias("PROD_DATE")
    )

    melted = melted.with_columns(
        pl.when(pl.col("PRODUCT_TYPE") == "GAS")
        .then(pl.col("RAW_VOLUME") * E3M3_TO_MCF)
        .otherwise(pl.col("RAW_VOLUME") * M3_TO_BBL)
        .alias("VOLUME_CONVERTED")
    )

    melted = melted.filter(pl.col("VOLUME_CONVERTED") > 0)
    return melted.select(["GSL_UWI_STD", "PROD_DATE", "PRODUCT_TYPE", "VOLUME_CONVERTED"])


def filter_products(production: pl.DataFrame, products: Sequence[str]) -> pl.DataFrame:
    if production.is_empty():
        return production
    if "BOE" in products:
        return production
    selected = {product for product in products if product in {"OIL", "CND", "GAS"}}
    if not selected:
        selected = {"OIL", "CND", "GAS"}
    return production.filter(pl.col("PRODUCT_TYPE").is_in(sorted(selected)))


def aggregate_monthly_production(
    production: pl.DataFrame,
    metadata: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Pivot the per-product values into consolidated monthly metrics."""

    if production.is_empty():
        return production

    base = production
    if metadata is not None and not metadata.is_empty() and "GSL_UWI_STD" in metadata.columns:
        cols = [col for col in ["GSL_UWI_STD", "OperatorName", "WellName", "Formation", "FieldName"] if col in metadata.columns]
        if cols:
            base = base.join(metadata.select(cols).unique(subset=["GSL_UWI_STD"]), on="GSL_UWI_STD", how="left")

    aggregations = [
        pl.sum(pl.when(pl.col("PRODUCT_TYPE") == "OIL").then(pl.col("VOLUME_CONVERTED")).otherwise(0.0)).alias("Monthly_Oil_BBL"),
        pl.sum(pl.when(pl.col("PRODUCT_TYPE") == "CND").then(pl.col("VOLUME_CONVERTED")).otherwise(0.0)).alias(
            "Monthly_Condensate_BBL"
        ),
        pl.sum(pl.when(pl.col("PRODUCT_TYPE") == "GAS").then(pl.col("VOLUME_CONVERTED")).otherwise(0.0)).alias(
            "Monthly_Gas_MCF"
        ),
    ]

    group_cols = [col for col in ["GSL_UWI_STD", "OperatorName", "WellName", "Formation", "FieldName", "PROD_DATE"] if col in base.columns]
    monthly = base.group_by(group_cols).agg(aggregations)

    monthly = monthly.with_columns(
        [
            (pl.col("Monthly_Oil_BBL") / AVG_DAYS_PER_MONTH).alias("Daily_Oil_Rate_BBLD"),
            (pl.col("Monthly_Condensate_BBL") / AVG_DAYS_PER_MONTH).alias("Daily_Condensate_Rate_BBLD"),
            (pl.col("Monthly_Gas_MCF") / AVG_DAYS_PER_MONTH).alias("Daily_Gas_Rate_MCFD"),
        ]
    )

    monthly = monthly.with_columns(
        [
            (
                pl.col("Monthly_Oil_BBL")
                + pl.col("Monthly_Condensate_BBL")
                + (pl.col("Monthly_Gas_MCF") / MCF_PER_BOE)
            ).alias("Monthly_BOE"),
            (
                (pl.col("Monthly_Oil_BBL") + pl.col("Monthly_Condensate_BBL") + (pl.col("Monthly_Gas_MCF") / MCF_PER_BOE))
                / AVG_DAYS_PER_MONTH
            ).alias("Daily_BOE_Rate_BBLD"),
        ]
    )
    return monthly.sort(group_cols)


def cumulative_metrics(frame: pl.DataFrame, partition_cols: list[str] | None = None) -> pl.DataFrame:
    """Append cumulative sums to monthly data."""

    if frame.is_empty():
        return frame

    partition_cols = partition_cols or []
    sort_cols = partition_cols + ["PROD_DATE"] if partition_cols else ["PROD_DATE"]
    sorted_frame = frame.sort(sort_cols)

    if partition_cols:
        return sorted_frame.with_columns(
            [
                pl.col("Monthly_Oil_BBL").cumsum().over(partition_cols).alias("CumOilTrueBBL"),
                pl.col("Monthly_Condensate_BBL").cumsum().over(partition_cols).alias("CumCondensateBBL"),
                pl.col("Monthly_Gas_MCF").cumsum().over(partition_cols).alias("CumGasMCF"),
                pl.col("Monthly_BOE").cumsum().over(partition_cols).alias("CumBOE"),
            ]
        )
    return sorted_frame.with_columns(
        [
            pl.col("Monthly_Oil_BBL").cumsum().alias("CumOilTrueBBL"),
            pl.col("Monthly_Condensate_BBL").cumsum().alias("CumCondensateBBL"),
            pl.col("Monthly_Gas_MCF").cumsum().alias("CumGasMCF"),
            pl.col("Monthly_BOE").cumsum().alias("CumBOE"),
        ]
    )


def filter_by_date_range(frame: pl.DataFrame, start: dt.date, end: dt.date) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    return frame.filter(pl.col("PROD_DATE").is_between(start, end))


def build_rate_plot(df: pd.DataFrame, products: Sequence[str], title: str) -> go.Figure:
    """Create a multi-series Plotly figure for daily rates."""

    colour_map = {
        "OIL": "#53143F",
        "CND": "#058F96",
        "GAS": "#F57D01",
        "BOE": "#5A63E3",
    }
    column_map = {
        "OIL": ("Daily_Oil_Rate_BBLD", "Oil"),
        "CND": ("Daily_Condensate_Rate_BBLD", "Condensate"),
        "GAS": ("Daily_Gas_Rate_MCFD", "Gas"),
        "BOE": ("Daily_BOE_Rate_BBLD", "BOE"),
    }

    fig = go.Figure()
    for product in products:
        column, label = column_map.get(product, (None, None))
        if column is None or column not in df.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=df["PROD_DATE"],
                y=df[column],
                mode="lines+markers",
                name=label,
                line=dict(color=colour_map.get(product, "#666666")),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Production Month",
        yaxis_title="Daily rate",
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


def build_type_curve_plot(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    series = [
        ("Average_Daily_BOE", "Average", "#5A63E3"),
        ("P90_BOE", "P90", "#058F96"),
        ("P50_BOE", "P50", "#53143F"),
        ("P10_BOE", "P10", "#F57D01"),
    ]
    for column, label, colour in series:
        if column not in df.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=df["MonthsOn"],
                y=df[column],
                mode="lines+markers",
                name=label,
                line=dict(color=colour),
            )
        )

    fig.update_layout(
        title="Type Curve Daily BOE",
        xaxis_title="Months Since First Production",
        yaxis_title="Average Daily BOE", 
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


def prepare_single_well_series(
    gsl_uwi_std: str,
    metadata: pl.DataFrame,
    products: Sequence[str],
    date_range: tuple[dt.date, dt.date],
) -> pl.DataFrame:
    raw = load_production_for_uwis((gsl_uwi_std,))
    tidy = reshape_production(raw)
    if tidy.is_empty():
        return tidy

    tidy = filter_products(tidy, products)
    monthly = aggregate_monthly_production(tidy, metadata)
    monthly = monthly.filter(pl.col("GSL_UWI_STD") == gsl_uwi_std)
    monthly = filter_by_date_range(monthly, *date_range)
    monthly = cumulative_metrics(monthly, partition_cols=["GSL_UWI_STD"])

    keep_cols = {"PROD_DATE", "Monthly_BOE", "Daily_BOE_Rate_BBLD", "CumBOE"}
    product_columns = {
        "OIL": {"Monthly_Oil_BBL", "Daily_Oil_Rate_BBLD", "CumOilTrueBBL"},
        "CND": {"Monthly_Condensate_BBL", "Daily_Condensate_Rate_BBLD", "CumCondensateBBL"},
        "GAS": {"Monthly_Gas_MCF", "Daily_Gas_Rate_MCFD", "CumGasMCF"},
        "BOE": {"Monthly_BOE", "Daily_BOE_Rate_BBLD", "CumBOE"},
    }
    for product in products:
        keep_cols.update(product_columns.get(product, set()))
    keep_cols.update({"GSL_UWI_STD", "WellName", "OperatorName", "Formation", "FieldName"} & set(monthly.columns))
    present = [col for col in keep_cols if col in monthly.columns]
    return monthly.select(present)


def prepare_group_series(
    gsl_uwis: Sequence[str],
    metadata: pl.DataFrame,
    products: Sequence[str],
    date_range: tuple[dt.date, dt.date],
) -> pl.DataFrame:
    if not gsl_uwis:
        return pl.DataFrame()
    raw = load_production_for_uwis(tuple(gsl_uwis))
    tidy = reshape_production(raw)
    if tidy.is_empty():
        return tidy

    tidy = filter_products(tidy, products)
    monthly = aggregate_monthly_production(tidy, metadata)
    monthly = monthly.filter(pl.col("GSL_UWI_STD").is_in(gsl_uwis))
    monthly = filter_by_date_range(monthly, *date_range)
    monthly = monthly.group_by("PROD_DATE").agg(
        [
            pl.sum("Monthly_Oil_BBL").alias("Monthly_Oil_BBL"),
            pl.sum("Monthly_Condensate_BBL").alias("Monthly_Condensate_BBL"),
            pl.sum("Monthly_Gas_MCF").alias("Monthly_Gas_MCF"),
            pl.sum("Monthly_BOE").alias("Monthly_BOE"),
        ]
    )
    monthly = monthly.with_columns(
        [
            (pl.col("Monthly_Oil_BBL") / AVG_DAYS_PER_MONTH).alias("Daily_Oil_Rate_BBLD"),
            (pl.col("Monthly_Condensate_BBL") / AVG_DAYS_PER_MONTH).alias("Daily_Condensate_Rate_BBLD"),
            (pl.col("Monthly_Gas_MCF") / AVG_DAYS_PER_MONTH).alias("Daily_Gas_Rate_MCFD"),
            (pl.col("Monthly_BOE") / AVG_DAYS_PER_MONTH).alias("Daily_BOE_Rate_BBLD"),
        ]
    )
    monthly = cumulative_metrics(monthly)

    keep_cols = {"PROD_DATE", "Monthly_BOE", "Daily_BOE_Rate_BBLD", "CumBOE"}
    product_columns = {
        "OIL": {"Monthly_Oil_BBL", "Daily_Oil_Rate_BBLD", "CumOilTrueBBL"},
        "CND": {"Monthly_Condensate_BBL", "Daily_Condensate_Rate_BBLD", "CumCondensateBBL"},
        "GAS": {"Monthly_Gas_MCF", "Daily_Gas_Rate_MCFD", "CumGasMCF"},
        "BOE": {"Monthly_BOE", "Daily_BOE_Rate_BBLD", "CumBOE"},
    }
    for product in products:
        keep_cols.update(product_columns.get(product, set()))
    present = [col for col in keep_cols if col in monthly.columns]
    return monthly.select(present)


def prepare_operator_group_series(
    operator_map: pl.DataFrame,
    metadata: pl.DataFrame,
    products: Sequence[str],
    date_range: tuple[dt.date, dt.date],
) -> pl.DataFrame:
    if operator_map.is_empty():
        return operator_map
    gsl_uwis = operator_map["GSL_UWI_STD"].drop_nulls().unique().to_list()
    raw = load_production_for_uwis(tuple(gsl_uwis))
    tidy = reshape_production(raw)
    if tidy.is_empty():
        return tidy

    tidy = filter_products(tidy, products)
    monthly = aggregate_monthly_production(tidy, metadata)
    monthly = monthly.join(operator_map.select(["GSL_UWI_STD", "OperatorName"]).unique(), on="GSL_UWI_STD", how="inner")
    monthly = filter_by_date_range(monthly, *date_range)
    monthly = monthly.group_by(["OperatorName", "PROD_DATE"]).agg(
        [
            pl.sum("Monthly_Oil_BBL").alias("Monthly_Oil_BBL"),
            pl.sum("Monthly_Condensate_BBL").alias("Monthly_Condensate_BBL"),
            pl.sum("Monthly_Gas_MCF").alias("Monthly_Gas_MCF"),
            pl.sum("Monthly_BOE").alias("Monthly_BOE"),
        ]
    )
    monthly = monthly.with_columns(
        [
            (pl.col("Monthly_Oil_BBL") / AVG_DAYS_PER_MONTH).alias("Daily_Oil_Rate_BBLD"),
            (pl.col("Monthly_Condensate_BBL") / AVG_DAYS_PER_MONTH).alias("Daily_Condensate_Rate_BBLD"),
            (pl.col("Monthly_Gas_MCF") / AVG_DAYS_PER_MONTH).alias("Daily_Gas_Rate_MCFD"),
            (pl.col("Monthly_BOE") / AVG_DAYS_PER_MONTH).alias("Daily_BOE_Rate_BBLD"),
        ]
    )
    monthly = cumulative_metrics(monthly, partition_cols=["OperatorName"])
    keep_cols = ["OperatorName", "PROD_DATE", "Monthly_BOE", "Daily_BOE_Rate_BBLD", "CumBOE"]
    product_columns = {
        "OIL": ["Monthly_Oil_BBL", "Daily_Oil_Rate_BBLD", "CumOilTrueBBL"],
        "CND": ["Monthly_Condensate_BBL", "Daily_Condensate_Rate_BBLD", "CumCondensateBBL"],
        "GAS": ["Monthly_Gas_MCF", "Daily_Gas_Rate_MCFD", "CumGasMCF"],
        "BOE": ["Monthly_BOE", "Daily_BOE_Rate_BBLD", "CumBOE"],
    }
    for product in products:
        keep_cols.extend([col for col in product_columns.get(product, []) if col not in keep_cols])
    keep_cols = [col for col in keep_cols if col in monthly.columns]
    return monthly.select(keep_cols)


def prepare_type_curve(
    gsl_uwis: Sequence[str],
    metadata: pl.DataFrame,
    products: Sequence[str],
) -> pl.DataFrame:
    if not gsl_uwis:
        return pl.DataFrame()
    raw = load_production_for_uwis(tuple(gsl_uwis))
    tidy = reshape_production(raw)
    if tidy.is_empty():
        return tidy

    tidy = filter_products(tidy, products)
    monthly = aggregate_monthly_production(tidy, metadata)
    monthly = monthly.filter(pl.col("GSL_UWI_STD").is_in(gsl_uwis))
    monthly = cumulative_metrics(monthly, partition_cols=["GSL_UWI_STD"])
    monthly = monthly.sort(["GSL_UWI_STD", "PROD_DATE"]).with_columns(
        pl.cumcount().over("GSL_UWI_STD").alias("MonthsOn")
    )

    summary = monthly.group_by("MonthsOn").agg(
        [
            pl.mean("Daily_BOE_Rate_BBLD").alias("Average_Daily_BOE"),
            pl.quantile("Daily_BOE_Rate_BBLD", 0.1).alias("P10_BOE"),
            pl.quantile("Daily_BOE_Rate_BBLD", 0.5).alias("P50_BOE"),
            pl.quantile("Daily_BOE_Rate_BBLD", 0.9).alias("P90_BOE"),
            pl.len().alias("SampleSize"),
        ]
    )
    summary = summary.sort("MonthsOn")
    if "BOE" not in products:
        # Still return the table but inform user upstream that BOE drives curve
        pass
    return summary


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

    metadata_cols = [col for col in ["GSL_UWI_STD", "OperatorName", "WellName", "Formation", "FieldName"] if col in wells.columns]
    wells_metadata = wells.select(metadata_cols).unique() if metadata_cols else pl.DataFrame()

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

    st.header("Production Analysis")
    analysis_products = st.multiselect(
        "Filter analyses by product type",
        options=PRODUCT_TYPE_CHOICES,
        default=PRODUCT_TYPE_CHOICES,
    )
    if not analysis_products:
        st.warning("Select at least one product to enable production analyses.")
        return

    tabs = st.tabs(
        [
            "Single Well Analysis",
            "Filtered Group Cumulative",
            "Type Curve Analysis",
            "Operator Group Cumulative",
        ]
    )

    with tabs[0]:
        st.subheader("Single Well Analysis")
        if metadata_cols and "GSL_UWI_STD" in wells.columns:
            lookup_df = wells.select(
                [
                    pl.col("GSL_UWI_STD"),
                    *(pl.col(col) for col in ["WellName", "OperatorName", "Formation"] if col in wells.columns),
                ]
            ).unique(subset=["GSL_UWI_STD"])
            if lookup_df.is_empty():
                st.info("No wells available for analysis.")
            else:
                lookup_pd = lookup_df.to_pandas()
                lookup_pd["label"] = lookup_pd.apply(
                    lambda row: " - ".join(
                        [
                            value
                            for value in [
                                row.get("OperatorName", ""),
                                row.get("WellName", ""),
                                row.get("Formation", ""),
                            ]
                            if value
                        ]
                    )
                    + f" ({row['GSL_UWI_STD']})",
                    axis=1,
                )
                selected_label = st.selectbox(
                    "Select well",
                    options=lookup_pd["label"],
                    help="Choose a well by operator/well name to inspect production.",
                )
                selected_row = lookup_pd.loc[lookup_pd["label"] == selected_label].iloc[0]
                selected_uwi = selected_row["GSL_UWI_STD"]
                default_start = dt.date.today() - dt.timedelta(days=3650)
                default_end = dt.date.today()
                start_date, end_date = st.date_input(
                    "Production date range",
                    value=(default_start, default_end),
                    key="single_well_date_range",
                )
                series = prepare_single_well_series(selected_uwi, wells_metadata, analysis_products, (start_date, end_date))
                if series.is_empty():
                    st.info("No production data available for the selected well.")
                else:
                    series_pd = series.to_pandas()
                    st.plotly_chart(
                        build_rate_plot(series_pd, analysis_products, f"Daily rates for {selected_label}"),
                        use_container_width=True,
                    )
                    st.dataframe(series_pd)
                    st.download_button(
                        "Download single well production",
                        data=series.write_csv().encode(),
                        file_name=f"production_{selected_uwi}.csv",
                        mime="text/csv",
                    )
        else:
            st.info("Well metadata is insufficient to run single well analysis.")

    with tabs[1]:
        st.subheader("Filtered Group Cumulative")
        gsl_uwis = filtered.get_column("GSL_UWI_STD") if "GSL_UWI_STD" in filtered.columns else None
        if gsl_uwis is None or gsl_uwis.is_null().all():
            st.info("Map filters currently return no wells with valid GSL_UWI identifiers.")
        else:
            default_start = dt.date.today() - dt.timedelta(days=3650)
            default_end = dt.date.today()
            group_start, group_end = st.date_input(
                "Group production date range",
                value=(default_start, default_end),
                key="filtered_group_date_range",
            )
            group_series = prepare_group_series(gsl_uwis.drop_nulls().unique().to_list(), wells_metadata, analysis_products, (group_start, group_end))
            if group_series.is_empty():
                st.info("No production data available for the filtered group.")
            else:
                group_pd = group_series.to_pandas()
                st.plotly_chart(
                    build_rate_plot(group_pd, analysis_products, "Filtered group daily rates"),
                    use_container_width=True,
                )
                st.dataframe(group_pd)
                st.download_button(
                    "Download filtered group production",
                    data=group_series.write_csv().encode(),
                    file_name="filtered_group_production.csv",
                    mime="text/csv",
                )

    with tabs[2]:
        st.subheader("Type Curve Analysis (Arps)")
        gsl_uwis = filtered.get_column("GSL_UWI_STD") if "GSL_UWI_STD" in filtered.columns else None
        if gsl_uwis is None or gsl_uwis.is_null().all():
            st.info("Select wells on the map to generate a type curve.")
        else:
            type_curve = prepare_type_curve(gsl_uwis.drop_nulls().unique().to_list(), wells_metadata, analysis_products)
            if type_curve.is_empty():
                st.info("No production data available to build a type curve for the current selection.")
            else:
                if "BOE" not in analysis_products:
                    st.warning(
                        "Type curve statistics are derived from BOE rates. Include BOE in the product filter for comparable curves.",
                        icon="⚠️",
                    )
                curve_pd = type_curve.to_pandas()
                st.plotly_chart(build_type_curve_plot(curve_pd), use_container_width=True)
                st.dataframe(curve_pd)
                st.download_button(
                    "Download type curve statistics",
                    data=type_curve.write_csv().encode(),
                    file_name="type_curve.csv",
                    mime="text/csv",
                )

    with tabs[3]:
        st.subheader("Operator Group Cumulative")
        if wells_metadata.is_empty() or "OperatorName" not in wells_metadata.columns:
            st.info("Operator names are unavailable; cannot build operator groups.")
        else:
            operators = (
                wells_metadata.select(["OperatorName", "GSL_UWI_STD"])
                .drop_nulls()
                .group_by("OperatorName")
                .count()
                .sort("OperatorName")
            )
            operator_options = operators["OperatorName"].to_list()
            selected_operators = st.multiselect(
                "Select operators",
                options=operator_options,
                default=operator_options[: min(len(operator_options), 3)],
            )
            if not selected_operators:
                st.info("Select at least one operator to build the cumulative view.")
            else:
                op_start, op_end = st.date_input(
                    "Operator production date range",
                    value=(dt.date.today() - dt.timedelta(days=3650), dt.date.today()),
                    key="operator_group_date_range",
                )
                operator_map = wells_metadata.filter(pl.col("OperatorName").is_in(selected_operators))
                operator_series = prepare_operator_group_series(operator_map, wells_metadata, analysis_products, (op_start, op_end))
                if operator_series.is_empty():
                    st.info("No production data found for the selected operators in the chosen range.")
                else:
                    op_pd = operator_series.to_pandas()
                    fig = go.Figure()
                    for operator in selected_operators:
                        subset = op_pd[op_pd["OperatorName"] == operator]
                        if subset.empty:
                            continue
                        fig.add_trace(
                            go.Scatter(
                                x=subset["PROD_DATE"],
                                y=subset["Daily_BOE_Rate_BBLD"],
                                mode="lines+markers",
                                name=f"{operator} BOE",
                            )
                        )
                    fig.update_layout(
                        title="Operator group daily BOE",
                        xaxis_title="Production Month",
                        yaxis_title="Daily BOE",
                        hovermode="x unified",
                        template="plotly_white",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(op_pd)
                    st.download_button(
                        "Download operator group production",
                        data=operator_series.write_csv().encode(),
                        file_name="operator_group_production.csv",
                        mime="text/csv",
                    )


if __name__ == "__main__":
    main()
