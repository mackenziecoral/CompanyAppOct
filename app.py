# --- 0. Load Necessary Packages ---
import os
import re
import math
import warnings
import html
from datetime import datetime, date, timedelta
import pickle
from itertools import cycle
from functools import lru_cache
import textwrap
import sys
import urllib.parse
from contextlib import contextmanager

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sqlalchemy import create_engine, event, text, bindparam
from sqlalchemy.exc import SQLAlchemyError
import oracledb
from scipy.optimize import curve_fit
import streamlit as st
import pydeck as pdk
import plotly.graph_objects as go
import plotnine as p9
from operator_coverage_data import OPERATOR_COVERAGE

# --- Database Connection Details ---
# (REPLACEMENT BLOCK: use python-oracledb in Thick mode with your Instant Client)

def _log(msg: str):
    print(msg, file=sys.stderr)

# ---- HARD SETTINGS (no placeholders)
DB_USER     = "WOODMAC"
DB_PASSWORD = "c0pp3r"
TNS_ALIAS   = "GDC_LINK.geologic.com"  # must match an alias in tnsnames.ora

# Your exact Instant Client location (as provided)
INSTANT_CLIENT_DIR = r"C:/Users/I37643/OneDrive - Wood Mackenzie Limited/Documents/InstantClient_64bit/instantclient_23_7"

# Prefer <instantclient>/network/admin if present; else fall back to instantclient dir
_possible_tns_admin = os.path.join(os.path.dirname(INSTANT_CLIENT_DIR), "network", "admin")
if os.path.isdir(_possible_tns_admin):
    TNS_ADMIN_DIR = _possible_tns_admin
else:
    TNS_ADMIN_DIR = INSTANT_CLIENT_DIR

# Ensure DLLs and Oracle Net config are discoverable
os.environ["PATH"] = INSTANT_CLIENT_DIR + os.pathsep + os.environ.get("PATH", "")
os.environ["TNS_ADMIN"] = TNS_ADMIN_DIR

# Initialize Thick mode explicitly (loads Instant Client + uses TNS_ADMIN)
try:
    oracledb.init_oracle_client(lib_dir=INSTANT_CLIENT_DIR, config_dir=TNS_ADMIN_DIR)
    _log(f"Oracle client initialized. TNS_ADMIN={TNS_ADMIN_DIR}")
except Exception as e:
    _log(f"⚠️ init_oracle_client warning: {e} — proceeding; may already be initialized.")

# Build the SQLAlchemy connection string for python-oracledb
CONNECTION_STRING = f"oracle+oracledb://{DB_USER}:{DB_PASSWORD}@{TNS_ALIAS}"

# Global engine object to hold the database connection pool.
engine = None

def connect_to_db():
    """
    Establishes and returns a SQLAlchemy engine object (oracle+oracledb, Thick mode).
    """
    global engine

    # If engine exists, test it and return if healthy
    if engine is not None:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1 FROM DUAL"))
            _log("Database connection is still valid.")
            return engine
        except Exception as e:
            _log(f"Connection lost, attempting to reconnect. Error: {e}")
            engine = None

    try:
        _log(f"Attempting to connect to Oracle: {TNS_ALIAS} as user: {DB_USER}")

        _engine = create_engine(
            CONNECTION_STRING,
            echo=False,          # Set True to debug SQL
            pool_pre_ping=True,  # drop stale connections automatically
            pool_recycle=3600,   # recycle hourly
            future=True,
        )

        # Set a per-statement call timeout (milliseconds) when supported
        @event.listens_for(_engine, "connect")
        def _set_call_timeout(dbapi_conn, conn_record):
            try:
                dbapi_conn.call_timeout = 30000  # 30s
            except Exception:
                pass  # ignore on older client versions

        # Smoke test
        with _engine.connect() as conn:
            conn.execute(text("SELECT 1 FROM DUAL"))

        _log("✅ SUCCESS: Database connection established.")
        engine = _engine
        return engine

    except SQLAlchemyError as e:
        _log("❌ ERROR during create_engine or connect call: Failed to connect to Oracle.")
        _log(f"Detailed Python error: {e}")
        engine = None
        return None

# Initial connection attempt when the app starts.
engine = connect_to_db()

# --- Resilient SQL read helper (handles ORA-08103 by retrying once) ---
from sqlalchemy.exc import DatabaseError

def read_sql_resilient(sql_text, params=None, max_retries=1):
    """
    Execute a SELECT and return a DataFrame.
    Retries once on ORA-08103 by disposing the pool and reconnecting.
    """
    attempt = 0
    while True:
        try:
            with engine.connect() as conn:
                # Accept either plain SQL string or sqlalchemy.text(...)
                return pd.read_sql(sql_text, conn, params=params)
        except DatabaseError as e:
            msg = str(e).upper()
            if attempt < max_retries and ("ORA-08103" in msg):
                attempt += 1
                try:
                    engine.dispose()
                except Exception:
                    pass
                continue
            raise


# Note on app shutdown: SQLAlchemy's connection pooling is designed to manage
# connections automatically. Unlike the R Shiny 'onStop', it's not typical
# in Python web apps to manually close the engine on shutdown. The pool handles
# stale connections and recycling.

# --- 1. Define File Paths and Constants ---
# UPDATE THIS PATH TO YOUR LOCAL DIRECTORY
BASE_PATH = "C:/Users/I37643/OneDrive - Wood Mackenzie Limited/Documents/WoodMac/APP" 
# Using .pkl (pickle) is a common Python format for saving data structures like DataFrames
PROCESSED_DATA_FILE = os.path.join(BASE_PATH, "processed_app_data.pkl")

PLAY_SUBPLAY_SHAPEFILE_DIR = os.path.join(BASE_PATH, "SubplayShapefile")
COMPANY_SHAPEFILES_DIR = os.path.join(BASE_PATH, "Shapefile")

# Conversion factors
E3M3_TO_MCF = 35.3147
M3_TO_BBL = 6.28981
AVG_DAYS_PER_MONTH = 30.4375
MCF_PER_BOE = 6  # Standard conversion for BOE calculations

MAX_WELL_RESULTS = 20000

FINAL_WELL_COLUMNS = [
    "UWI", "GSL_UWI", "SurfaceLatitude", "SurfaceLongitude",
    "BH_Latitude", "BH_Longitude", "LateralLength",
    "AbandonmentDate", "WellName", "CurrentStatus", "OperatorCode", "StratUnitID",
    "SpudDate", "FirstProdDate", "FinalTD", "ProvinceState", "Country",
    "UWI_Std", "GSL_UWI_Std", "OperatorName", "Formation", "FieldName",
    "ConfidentialType"
]

# Custom Color Palette
CUSTOM_PALETTE = [
  "#53143F", "#F94355", "#058F96", "#F57D01", "#205B2E", "#A8011E",
  "#5A63E3", "#FFD31A", "#4E207F", "#CC9900", "#A6A6A6", "#9ACEE2",
  "#A9899E", "#FFA3AA", "#92C5C9", "#F9BE96", "#92AC96", "#D6890B",
  "#ABAFF0", "#FFE89C", "#A68EBF", "#D7C68E", "#D1D1D1", "#CBE5EF"
]

# --- 2. Helper Functions ---

def _render_plotly_chart(fig):
    """Render a Plotly figure using the widest supported Streamlit API."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*use_container_width.*",
            category=FutureWarning,
        )
        try:
            st.plotly_chart(fig, width="stretch")
            return
        except TypeError:
            # Older Streamlit versions may not support the ``width`` keyword.
            st.plotly_chart(fig)


def _render_dataframe(df: pd.DataFrame):
    """Render a dataframe using the widest supported Streamlit API."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*use_container_width.*",
            category=FutureWarning,
        )
        try:
            st.dataframe(df, width="stretch")
            return
        except TypeError:
            st.dataframe(df)

def standardize_uwi(uwi_series: pd.Series) -> pd.Series:
    """ Replaces R 'standardize_uwi' function. """
    if not isinstance(uwi_series, pd.Series):
        uwi_series = pd.Series(uwi_series)
    return uwi_series.astype(str).str.replace(r'[^A-Za-z0-9]', '', regex=True).str.upper()

def clean_df_colnames(df_input: pd.DataFrame, df_name_for_message: str = "a dataframe") -> pd.DataFrame:
    """ Replaces R 'clean_df_colnames' function, making column names database/Python friendly. """
    if df_input is None or not isinstance(df_input, pd.DataFrame) or df_input.empty:
        return df_input
    
    df = df_input.copy()
    
    def clean_name(name):
        new_name = re.sub(r'\.+', '_', str(name))
        new_name = re.sub(r'\s+', '_', new_name)
        new_name = re.sub(r'[^a-zA-Z0-9_]', '', new_name)
        new_name = re.sub(r'_+', '_', new_name)
        return new_name.upper()

    original_names = df.columns
    new_names = [clean_name(name) for name in original_names]
    
    # Handle potential duplicates created by cleaning
    final_names = []
    counts = {}
    for name in new_names:
        if name in counts:
            counts[name] += 1
            final_names.append(f"{name}_{counts[name]}")
        else:
            counts[name] = 0
            final_names.append(name)

    df.columns = final_names
    return df


def normalize_operator_code_series(series: pd.Series | None) -> pd.Series:
    """Return operator codes in a comparable alphanumeric format."""
    if series is None or not isinstance(series, pd.Series):
        return pd.Series(dtype=object)

    normalized = series.astype(object)
    normalized = normalized.where(~normalized.isna())
    normalized = normalized.astype(str).str.strip().str.upper()
    normalized = normalized.str.replace(r"\.0$", "", regex=True)
    normalized = normalized.str.replace(r"[^0-9A-Z]", "", regex=True)
    normalized = normalized.replace({"": np.nan, "NAN": np.nan, "NONE": np.nan})
    return normalized


def ensure_operator_and_formation_columns(df: pd.DataFrame | gpd.GeoDataFrame) -> pd.DataFrame | gpd.GeoDataFrame:
    """Guarantee OperatorName and Formation columns have usable values."""

    if df is None or len(df) == 0:
        return df

    working = df.copy()

    # --- Operator cleanup ---
    if 'OperatorName' not in working.columns:
        working['OperatorName'] = np.nan

    operator_clean = working['OperatorName'].astype(object)
    operator_clean = operator_clean.where(operator_clean.notna())
    operator_clean = operator_clean.astype(str).str.strip()
    operator_clean = operator_clean.replace({'': np.nan, 'nan': np.nan, 'None': np.nan})

    fallback_operator_columns = [
        'OperatorNameDisplay', 'OPERATORNAME_DISPLAY',
        'Operator', 'OPERATOR',
        'OperatorCode', 'OPERATOR_CODE'
    ]

    for col in fallback_operator_columns:
        if col not in working.columns:
            continue
        series = working[col]
        if col in {'OperatorCode', 'OPERATOR_CODE'}:
            series = normalize_operator_code_series(series)
        else:
            series = series.astype(object)
            series = series.where(series.notna())
            series = series.astype(str).str.strip()
            series = series.replace({'': np.nan, 'nan': np.nan, 'None': np.nan})
        operator_clean = operator_clean.fillna(series)

    operator_clean = operator_clean.fillna('Unknown')
    working['OperatorName'] = operator_clean

    # --- Formation cleanup ---
    if 'Formation' not in working.columns:
        working['Formation'] = np.nan

    formation_clean = working['Formation'].astype(object)
    formation_clean = formation_clean.where(formation_clean.notna())
    formation_clean = formation_clean.astype(str).str.strip()
    formation_clean = formation_clean.replace({'': np.nan, 'nan': np.nan, 'None': np.nan})

    def _normalize_strat_value(val):
        if pd.isna(val):
            return np.nan
        try:
            return str(int(float(val)))
        except (TypeError, ValueError):
            cleaned = re.sub(r"[^0-9A-Za-z]", "", str(val)).strip()
            return cleaned or np.nan

    strat_fallback_columns = ['Formation', 'STRAT_UNIT_NAME', 'StratUnitName', 'StratUnitID', 'STRAT_UNIT_ID']
    for col in strat_fallback_columns:
        if col not in working.columns:
            continue
        series = working[col]
        if col != 'Formation':
            series = series.apply(_normalize_strat_value)
        else:
            series = series.astype(object)
            series = series.where(series.notna())
            series = series.astype(str).str.strip()
            series = series.replace({'': np.nan, 'nan': np.nan, 'None': np.nan})
        formation_clean = formation_clean.fillna(series)

    formation_clean = formation_clean.fillna('Unknown')
    working['Formation'] = formation_clean

    return working


@lru_cache(maxsize=1)
def load_operator_coverage_df() -> pd.DataFrame:
    """Return Wood Mackenzie operator coverage from the embedded mapping."""

    if not OPERATOR_COVERAGE:
        return pd.DataFrame(columns=["OPERATOR_CODE", "OPERATORNAME_DISPLAY"])

    codes = pd.Series(list(OPERATOR_COVERAGE.keys()), dtype=object)
    names = pd.Series(list(OPERATOR_COVERAGE.values()), dtype=object)

    normalized_codes = normalize_operator_code_series(codes)
    display_names = names.where(names.notna(), other="")
    display_names = display_names.astype(str).str.strip()
    display_names = display_names.replace({
        "": np.nan,
        "nan": np.nan,
        "NAN": np.nan,
        "None": np.nan,
        "NONE": np.nan,
    })

    coverage_df = pd.DataFrame({
        "OPERATOR_CODE": normalized_codes,
        "OPERATORNAME_DISPLAY": display_names,
    })

    coverage_df = coverage_df.replace({
        "": np.nan,
        "nan": np.nan,
        "NAN": np.nan,
        "None": np.nan,
        "NONE": np.nan,
    })
    coverage_df = coverage_df.dropna(subset=["OPERATOR_CODE"])
    coverage_df["OPERATORNAME_DISPLAY"] = coverage_df["OPERATORNAME_DISPLAY"].fillna(coverage_df["OPERATOR_CODE"])
    coverage_df["OPERATORNAME_DISPLAY"] = coverage_df["OPERATORNAME_DISPLAY"].astype(str).str.strip()
    coverage_df = coverage_df.drop_duplicates(subset=["OPERATOR_CODE"])

    return coverage_df[["OPERATOR_CODE", "OPERATORNAME_DISPLAY"]]


def _base_well_filters() -> str:
    return (
        "W.SURFACE_LATITUDE IS NOT NULL AND W.SURFACE_LONGITUDE IS NOT NULL\n"
        "  AND (W.ABANDONMENT_DATE IS NULL OR W.ABANDONMENT_DATE > SYSDATE - (365*20))"
    )


@lru_cache(maxsize=1)
def get_operator_display_lookup() -> dict[str, str]:
    global engine
    if engine is None:
        engine = connect_to_db()
    if engine is None:
        return {}

    sql = text(
        """
        SELECT DISTINCT W.OPERATOR AS OPERATOR_CODE
        FROM WELL W
        WHERE """ + _base_well_filters() + " AND W.OPERATOR IS NOT NULL"
    )

    try:
        operators_df = read_sql_resilient(sql)
    except Exception:
        operators_df = pd.DataFrame()

    coverage_df = load_operator_coverage_df()
    coverage_map = (
        dict(zip(coverage_df['OPERATOR_CODE'], coverage_df['OPERATORNAME_DISPLAY']))
        if not coverage_df.empty
        else {}
    )

    if operators_df.empty:
        return {
            str(code): str(name) if isinstance(name, str) and name.strip() else str(code)
            for code, name in coverage_map.items()
        }

    operators_df = clean_df_colnames(operators_df, "Distinct Operator Codes")
    column = 'OPERATOR_CODE' if 'OPERATOR_CODE' in operators_df.columns else operators_df.columns[0]
    codes = normalize_operator_code_series(operators_df[column])

    lookup: dict[str, str] = {}
    for code in codes.dropna().unique():
        display = coverage_map.get(code)
        lookup[str(code)] = display.strip() if isinstance(display, str) and display.strip() else str(code)

    return lookup


@lru_cache(maxsize=1)
def get_formation_lookup() -> dict[str, str]:
    global engine
    if engine is None:
        engine = connect_to_db()
    if engine is None:
        return {}

    base_sql = text(
        """
        SELECT DISTINCT P.STRAT_UNIT_ID, SU.SHORT_NAME
        FROM WELL W
        LEFT JOIN PDEN P ON W.GSL_UWI = P.GSL_UWI
        LEFT JOIN STRAT_UNIT SU ON P.STRAT_UNIT_ID = SU.STRAT_UNIT_ID
        WHERE """ + _base_well_filters() + " AND P.STRAT_UNIT_ID IS NOT NULL"
    )

    try:
        strat_df = read_sql_resilient(base_sql)
    except Exception:
        strat_df = pd.DataFrame()

    if strat_df.empty:
        try:
            strat_df = read_sql_resilient(
                text(
                    """
                    SELECT STRAT_UNIT_ID, SHORT_NAME
                    FROM STRAT_UNIT
                    WHERE STRAT_UNIT_ID IS NOT NULL
                    """
                )
            )
        except Exception:
            strat_df = pd.DataFrame()

    if strat_df.empty:
        return {}

    strat_df = clean_df_colnames(strat_df, "Distinct Strat Units")
    lookup: dict[str, str] = {}
    for _, row in strat_df.iterrows():
        raw_id = row.get('STRAT_UNIT_ID')
        if pd.isna(raw_id):
            continue
        str_id = str(raw_id).strip()
        if not str_id:
            continue
        short_name = row.get('SHORT_NAME')
        if isinstance(short_name, str):
            display = short_name.strip() or str_id
        else:
            display = str_id
        lookup[str_id] = display
    return lookup


@lru_cache(maxsize=1)
def get_field_options() -> list[str]:
    global engine
    if engine is None:
        engine = connect_to_db()
    if engine is None:
        return []

    sql = text(
        """
        SELECT DISTINCT FL.FIELD_NAME
        FROM WELL W
        LEFT JOIN FIELD FL ON W.ASSIGNED_FIELD = FL.FIELD_ID
        WHERE """ + _base_well_filters() + " AND FL.FIELD_NAME IS NOT NULL"
    )

    try:
        fields_df = read_sql_resilient(sql)
    except Exception:
        return []

    if fields_df.empty:
        return []

    fields_df = clean_df_colnames(fields_df, "Distinct Field Names")
    column = 'FIELD_NAME' if 'FIELD_NAME' in fields_df.columns else fields_df.columns[0]
    field_series = fields_df[column].astype(str).str.strip()
    field_series = field_series[field_series != '']
    return sorted(field_series.unique().tolist())


@lru_cache(maxsize=1)
def get_province_options() -> list[str]:
    global engine
    if engine is None:
        engine = connect_to_db()
    if engine is None:
        return []

    sql = text(
        """
        SELECT DISTINCT W.PROVINCE_STATE
        FROM WELL W
        WHERE """ + _base_well_filters() + " AND W.PROVINCE_STATE IS NOT NULL"
    )

    try:
        province_df = read_sql_resilient(sql)
    except Exception:
        return []

    if province_df.empty:
        return []

    province_df = clean_df_colnames(province_df, "Distinct Provinces")
    column = 'PROVINCE_STATE' if 'PROVINCE_STATE' in province_df.columns else province_df.columns[0]
    series = province_df[column].astype(str).str.strip()
    series = series[series != '']
    return sorted(series.unique().tolist())


@lru_cache(maxsize=1)
def get_first_prod_bounds() -> tuple[date, date] | None:
    global engine
    if engine is None:
        engine = connect_to_db()
    if engine is None:
        return None

    sql = text(
        """
        SELECT MIN(PFS.FIRST_PROD_DATE) AS MIN_DATE, MAX(PFS.FIRST_PROD_DATE) AS MAX_DATE
        FROM WELL W
        LEFT JOIN PDEN_FIRST_SUM PFS ON W.GSL_UWI = PFS.GSL_UWI
        WHERE """ + _base_well_filters()
    )

    try:
        bounds_df = read_sql_resilient(sql)
    except Exception:
        return None

    if bounds_df.empty:
        return None

    bounds_df = clean_df_colnames(bounds_df, "First Prod Bounds")
    min_col = 'MIN_DATE' if 'MIN_DATE' in bounds_df.columns else bounds_df.columns[0]
    max_col = 'MAX_DATE' if 'MAX_DATE' in bounds_df.columns else bounds_df.columns[-1]
    min_date = pd.to_datetime(bounds_df[min_col].iloc[0], errors='coerce')
    max_date = pd.to_datetime(bounds_df[max_col].iloc[0], errors='coerce')
    if pd.isna(min_date) or pd.isna(max_date):
        return None
    return min_date.date(), max_date.date()


def _process_well_records(df: pd.DataFrame, truncated: bool) -> gpd.GeoDataFrame:
    empty_gdf = gpd.GeoDataFrame(columns=FINAL_WELL_COLUMNS, geometry=[], crs="EPSG:4326")
    empty_gdf.attrs['truncated'] = False

    if df is None or df.empty:
        return empty_gdf

    df = clean_df_colnames(df, "Filtered Wells from DB")

    if 'UWI' in df.columns:
        df['UWI_STD'] = standardize_uwi(df['UWI'])
    else:
        df['UWI_STD'] = np.nan

    if 'GSL_UWI' in df.columns:
        df['GSL_UWI_STD'] = standardize_uwi(df['GSL_UWI'])
    else:
        df['GSL_UWI_STD'] = np.nan

    if 'STRAT_UNIT_ID' in df.columns:
        df['STRAT_UNIT_ID'] = df['STRAT_UNIT_ID'].astype(object)

    if 'OPERATOR_CODE' in df.columns:
        df['OPERATOR_CODE'] = normalize_operator_code_series(df['OPERATOR_CODE'])

    coverage_df = load_operator_coverage_df()
    if not coverage_df.empty and 'OPERATOR_CODE' in df.columns:
        coverage_map = dict(zip(coverage_df['OPERATOR_CODE'], coverage_df['OPERATORNAME_DISPLAY']))
        df['OperatorNameDisplay'] = df['OPERATOR_CODE'].map(coverage_map)

    rename_map = {
        'UWI': 'UWI',
        'GSL_UWI': 'GSL_UWI',
        'SURFACE_LATITUDE': 'SurfaceLatitude',
        'SURFACE_LONGITUDE': 'SurfaceLongitude',
        'BOTTOM_HOLE_LATITUDE': 'BH_Latitude',
        'BOTTOM_HOLE_LONGITUDE': 'BH_Longitude',
        'GSL_FULL_LATERAL_LENGTH': 'LateralLength',
        'ABANDONMENT_DATE': 'AbandonmentDate',
        'WELL_NAME': 'WellName',
        'CURRENT_STATUS': 'CurrentStatus',
        'OPERATOR_CODE': 'OperatorCode',
        'STRAT_UNIT_ID': 'StratUnitID',
        'STRAT_UNIT_NAME': 'Formation',
        'SPUD_DATE': 'SpudDate',
        'FIRST_PROD_DATE': 'FirstProdDate',
        'FINAL_TD': 'FinalTD',
        'PROVINCE_STATE': 'ProvinceState',
        'COUNTRY': 'Country',
        'FIELD_NAME': 'FieldName',
        'CONFIDENTIAL_TYPE': 'ConfidentialType',
        'UWI_STD': 'UWI_Std',
        'GSL_UWI_STD': 'GSL_UWI_Std',
        'OperatorNameDisplay': 'OperatorName'
    }
    df = df.rename(columns=rename_map)

    formation_lookup = get_formation_lookup()
    if formation_lookup and 'StratUnitID' in df.columns:
        df['Formation'] = df.get('Formation').fillna(df['StratUnitID'].map(formation_lookup))

    operator_lookup = get_operator_display_lookup()
    if operator_lookup and 'OperatorCode' in df.columns:
        df['OperatorName'] = df.get('OperatorName').fillna(df['OperatorCode'].map(operator_lookup))

    df = ensure_operator_and_formation_columns(df)

    if 'GSL_UWI_Std' in df.columns:
        df = df.drop_duplicates(subset=['GSL_UWI_Std'])
    elif 'GSL_UWI' in df.columns:
        df = df.drop_duplicates(subset=['GSL_UWI'])

    for col in FINAL_WELL_COLUMNS:
        if col not in df.columns:
            df[col] = None

    date_cols = ['AbandonmentDate', 'SpudDate', 'FirstProdDate']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    numeric_cols = ['SurfaceLatitude', 'SurfaceLongitude', 'BH_Latitude', 'BH_Longitude', 'LateralLength', 'FinalTD']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    valid_df = df.dropna(subset=['SurfaceLongitude', 'SurfaceLatitude'])
    if valid_df.empty:
        result = gpd.GeoDataFrame(
            df[FINAL_WELL_COLUMNS].copy(),
            geometry=gpd.GeoSeries([None] * len(df), crs="EPSG:4326"),
            crs="EPSG:4326"
        )
        result.attrs['truncated'] = truncated
        return result

    gdf = gpd.GeoDataFrame(
        valid_df[FINAL_WELL_COLUMNS],
        geometry=gpd.points_from_xy(valid_df.SurfaceLongitude, valid_df.SurfaceLatitude),
        crs="EPSG:4269"
    ).to_crs("EPSG:4326")
    gdf.attrs['truncated'] = truncated
    return gdf


def fetch_wells_from_db(
    operator_codes: list[str] | None,
    formation_ids: list[str] | None,
    fields: list[str] | None,
    provinces: list[str] | None,
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None,
) -> gpd.GeoDataFrame:
    global engine
    if engine is None:
        engine = connect_to_db()
    if engine is None:
        raise RuntimeError("Database connection not available.")

    params: dict[str, object] = {}
    optional_clauses: list[str] = []

    if operator_codes:
        codes_series = normalize_operator_code_series(pd.Series(operator_codes))
        normalized_codes = [code for code in codes_series.dropna().unique().tolist() if str(code).strip()]
        if normalized_codes:
            optional_clauses.append("W.OPERATOR IN :operator_codes")
            params['operator_codes'] = [str(code) for code in normalized_codes]

    if formation_ids:
        normalized_formations = [str(fid).strip() for fid in formation_ids if str(fid).strip()]
        if normalized_formations:
            optional_clauses.append("P.STRAT_UNIT_ID IN :formation_ids")
            params['formation_ids'] = normalized_formations

    if fields:
        normalized_fields = [str(f).strip() for f in fields if str(f).strip()]
        if normalized_fields:
            optional_clauses.append("FL.FIELD_NAME IN :fields")
            params['fields'] = normalized_fields

    if provinces:
        normalized_provinces = [str(p).strip() for p in provinces if str(p).strip()]
        if normalized_provinces:
            optional_clauses.append("W.PROVINCE_STATE IN :provinces")
            params['provinces'] = normalized_provinces

    if date_range and all(date_range):
        start_date, end_date = date_range
        start_ts = pd.to_datetime(start_date, errors='coerce')
        end_ts = pd.to_datetime(end_date, errors='coerce')
        if pd.notna(start_ts) and pd.notna(end_ts):
            optional_clauses.append("PFS.FIRST_PROD_DATE BETWEEN :first_prod_start AND :first_prod_end")
            params['first_prod_start'] = start_ts
            params['first_prod_end'] = end_ts

    base_filters = _base_well_filters()
    if optional_clauses:
        base_filters = base_filters + "\n  AND " + "\n  AND ".join(optional_clauses)

    inner_sql = f"""
        SELECT DISTINCT
            W.UWI, W.GSL_UWI, W.SURFACE_LATITUDE, W.SURFACE_LONGITUDE,
            W.BOTTOM_HOLE_LATITUDE, W.BOTTOM_HOLE_LONGITUDE, W.GSL_FULL_LATERAL_LENGTH,
            W.ABANDONMENT_DATE, W.WELL_NAME, W.CURRENT_STATUS, W.OPERATOR AS OPERATOR_CODE, W.CONFIDENTIAL_TYPE,
            P.STRAT_UNIT_ID, SU.SHORT_NAME AS STRAT_UNIT_NAME, W.SPUD_DATE, PFS.FIRST_PROD_DATE,
            W.FINAL_TD, W.PROVINCE_STATE, W.COUNTRY, FL.FIELD_NAME
        FROM WELL W
        LEFT JOIN PDEN P ON W.GSL_UWI = P.GSL_UWI
        LEFT JOIN STRAT_UNIT SU ON P.STRAT_UNIT_ID = SU.STRAT_UNIT_ID
        LEFT JOIN FIELD FL ON W.ASSIGNED_FIELD = FL.FIELD_ID
        LEFT JOIN PDEN_FIRST_SUM PFS ON W.GSL_UWI = PFS.GSL_UWI
        WHERE {base_filters}
    """

    outer_sql = f"SELECT * FROM ({inner_sql}) WHERE ROWNUM <= :max_rows"
    params['max_rows'] = MAX_WELL_RESULTS

    text_obj = text(outer_sql)
    if 'operator_codes' in params:
        text_obj = text_obj.bindparams(bindparam('operator_codes', expanding=True))
    if 'formation_ids' in params:
        text_obj = text_obj.bindparams(bindparam('formation_ids', expanding=True))
    if 'fields' in params:
        text_obj = text_obj.bindparams(bindparam('fields', expanding=True))
    if 'provinces' in params:
        text_obj = text_obj.bindparams(bindparam('provinces', expanding=True))

    try:
        wells_df = read_sql_resilient(text_obj, params=params)
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch wells: {exc}")

    truncated = len(wells_df) >= MAX_WELL_RESULTS if wells_df is not None else False
    return _process_well_records(wells_df, truncated)

def prepare_filter_choices(column_series: pd.Series, col_name_for_msg: str = "column") -> list:
    """ Replaces R 'prepare_filter_choices' function for populating dropdowns. """
    if column_series is None or column_series.empty:
        print(f"DEBUG (prepare_filter_choices): Input series for '{col_name_for_msg}' is None or empty.")
        return []
    
    num_na_initial = column_series.isna().sum()
    num_empty_initial = (column_series.astype(str) == '').sum()
    
    choices = column_series.dropna()
    choices = choices[choices.astype(str).str.strip() != '']
    choices = choices[choices.astype(str).str.upper() != 'NA']
    
    unique_choices = sorted(choices.unique().tolist())
    
    print(f"DEBUG (prepare_filter_choices for '{col_name_for_msg}'): "
          f"Original NAs: {num_na_initial}, Original empty strings: {num_empty_initial}. "
          f"Final unique choices: {len(unique_choices)}")
    
    return unique_choices

def load_process_spatial_layer(shp_path: str, layer_name: str, target_crs: int = 4326, 
                               simplify: bool = False, tolerance: float = None, 
                               make_valid_geom: bool = False) -> gpd.GeoDataFrame | None:
    """ Replaces R 'load_process_spatial_layer' using GeoPandas. """
    print(f"--- Start Processing Layer: {layer_name} ---")
    print(f"Shapefile path: {shp_path}")
    try:
        if not os.path.exists(shp_path):
            warnings.warn(f"Shapefile does not exist at path: {shp_path} for layer {layer_name}")
            return None
        
        data_gdf = gpd.read_file(shp_path)
        if data_gdf.empty:
            warnings.warn(f"Shapefile for layer {layer_name} is empty.")
            return None
        
        print(f"Layer {layer_name} read successfully, rows: {len(data_gdf)}")
        
        if data_gdf.crs is None or data_gdf.crs.to_epsg() != target_crs:
            print(f"Transforming layer {layer_name} to CRS {target_crs}")
            data_gdf = data_gdf.to_crs(epsg=target_crs)
        
        if make_valid_geom:
            print(f"Making geometries valid for layer {layer_name}")
            data_gdf['geometry'] = data_gdf.geometry.make_valid()
        
        if simplify and tolerance is not None and tolerance > 0:
            print(f"Simplifying layer {layer_name} with tolerance {tolerance}")
            data_gdf['geometry'] = data_gdf.geometry.simplify(tolerance=tolerance, preserve_topology=True)
        
        data_gdf = data_gdf[~data_gdf.geometry.is_empty]
        if data_gdf.empty:
            warnings.warn(f"All geometries became empty for layer {layer_name} after processing.")
            return None
        
        attrs_df = data_gdf.drop(columns='geometry')
        cleaned_attrs_df = clean_df_colnames(attrs_df, f"attributes of {layer_name}")
        
        data_gdf_final = gpd.GeoDataFrame(cleaned_attrs_df, geometry=data_gdf.geometry, crs=data_gdf.crs)
        
        if "SHP_LAYER_NAME" not in data_gdf_final.columns:
            data_gdf_final["SHP_LAYER_NAME"] = layer_name
            
        print(f"--- Successfully processed layer: {layer_name} - Final Features: {len(data_gdf_final)}")
        return data_gdf_final
        
    except Exception as e:
        print(f"!!!!!!!! ERROR processing shapefile {shp_path} for layer {layer_name}: {e} !!!!!!!!")
        return None

def geometric_mean(x: pd.Series, na_rm: bool = True) -> float:
    """ Replaces R 'geometric_mean' function using NumPy. """
    if na_rm:
        x = x.dropna()
    
    if (x < 0).any() or x.empty:
        return np.nan
    
    if (x == 0).any():
        return 0.0
        
    return np.exp(np.mean(np.log(x.loc[x > 0])))

# --- 3. Load and Pre-process Data ---
# This function encapsulates the entire data loading and preprocessing logic from the R script.
def load_and_process_all_data():

    empty_gdf = gpd.GeoDataFrame(columns=FINAL_WELL_COLUMNS, geometry=[], crs="EPSG:4326")

    app_data = {
        'wells_gdf': empty_gdf,
        'play_subplay_layers_list': [],
        'company_layers_list': []
    }

    if os.path.exists(PROCESSED_DATA_FILE):
        print(f"Attempting to load cached layers from: {PROCESSED_DATA_FILE}")
        try:
            with open(PROCESSED_DATA_FILE, 'rb') as f:
                loaded_data = pickle.load(f)
            if isinstance(loaded_data, dict):
                play_layers = loaded_data.get('play_subplay_layers_list', [])
                company_layers = loaded_data.get('company_layers_list', [])
                if isinstance(play_layers, list) and play_layers:
                    app_data['play_subplay_layers_list'] = play_layers
                if isinstance(company_layers, list) and company_layers:
                    app_data['company_layers_list'] = company_layers
                print("SUCCESS: Loaded shapefile layers from cache.")
        except Exception as e:
            print(f"WARNING: Unable to load cached data ({e}). Will rebuild from source files.")

    if not app_data['play_subplay_layers_list']:
        print("--- Loading Play/Subplay Acreage ---")
        play_layers: list[dict] = []
        if os.path.isdir(PLAY_SUBPLAY_SHAPEFILE_DIR):
            for root_dir, _, files in os.walk(PLAY_SUBPLAY_SHAPEFILE_DIR):
                for filename in files:
                    if filename.lower().endswith('.shp'):
                        shp_path = os.path.join(root_dir, filename)
                        layer_name = os.path.splitext(filename)[0]
                        gdf = load_process_spatial_layer(
                            shp_path,
                            layer_name,
                            simplify=True,
                            tolerance=100,
                            make_valid_geom=True,
                        )
                        if gdf is not None:
                            play_layers.append({'name': layer_name, 'data': gdf})
        app_data['play_subplay_layers_list'] = play_layers

    if not app_data['company_layers_list']:
        print("--- Loading DISSOLVED Company Acreage ---")
        company_layers: list[dict] = []
        if os.path.isdir(COMPANY_SHAPEFILES_DIR):
            for filename in os.listdir(COMPANY_SHAPEFILES_DIR):
                if filename.lower().endswith('.shp'):
                    shp_path = os.path.join(COMPANY_SHAPEFILES_DIR, filename)
                    layer_name = os.path.splitext(filename)[0]
                    gdf = load_process_spatial_layer(
                        shp_path,
                        layer_name,
                        simplify=False,
                        make_valid_geom=False,
                    )
                    if gdf is not None:
                        company_layers.append({'name': layer_name, 'data': gdf})
        app_data['company_layers_list'] = company_layers

    try:
        cache_payload = {
            'wells_gdf': empty_gdf,
            'play_subplay_layers_list': app_data['play_subplay_layers_list'],
            'company_layers_list': app_data['company_layers_list'],
        }
        with open(PROCESSED_DATA_FILE, 'wb') as f:
            pickle.dump(cache_payload, f)
        print(f"Cached shapefile layers to: {PROCESSED_DATA_FILE}")
    except Exception as e:
        print(f"Warning: Unable to cache shapefile layers ({e})")

    return app_data

# --- Load data into global variables for the app ---
APP_DATA = load_and_process_all_data()
wells_gdf_global = ensure_operator_and_formation_columns(APP_DATA['wells_gdf'])
play_subplay_layers_list_global = APP_DATA['play_subplay_layers_list']
company_layers_list_global = APP_DATA['company_layers_list']

# Prepare initial choices for UI dropdowns
initial_play_subplay_layer_names = sorted([layer['name'] for layer in play_subplay_layers_list_global])
initial_company_layer_names = sorted([layer['name'] for layer in company_layers_list_global])


# Final checks before starting server
print("--- FINAL CHECK OF wells_gdf_global BEFORE STREAMLIT ---")
if wells_gdf_global is not None and not wells_gdf_global.empty:
    print(f"  nrow(wells_gdf_global): {len(wells_gdf_global)}")
    for col in ['ConfidentialType', 'BH_Latitude', 'BH_Longitude', 'LateralLength', 'FirstProdDate', 'OperatorName', 'Formation']:
        if col in wells_gdf_global.columns:
            print(f"  '{col}' column IS present. Sample: {wells_gdf_global[col].dropna().head(3).to_list()}")
        else:
            print(f"  '{col}' column IS NOT present.")
else:
    print("  wells_gdf_global is empty before server starts (wells load on demand).")



# --- 4. Streamlit Interface ---
STREAMLIT_PAGE_TITLE = "Interactive Well and Acreage Map Application (Streamlit)"


def _hex_to_rgba(hex_color: str, alpha: int = 160) -> list[int]:
    hex_color = (hex_color or "").lstrip('#')
    if len(hex_color) != 6:
        return [0, 0, 0, alpha]
    return [int(hex_color[i:i+2], 16) for i in (0, 2, 4)] + [alpha]


def _geodf_to_polygon_records(gdf: gpd.GeoDataFrame) -> list[dict]:
    if gdf is None or gdf.empty:
        return []

    records: list[dict] = []

    def _polygon_coords(poly):
        coords = list(poly.exterior.coords)
        return [[float(x), float(y)] for x, y in coords]

    for _, row in gdf.iterrows():
        geom = row.get('geometry')
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == 'Polygon':
            records.append({'coordinates': _polygon_coords(geom)})
        elif geom.geom_type == 'MultiPolygon':
            for sub_poly in geom.geoms:
                records.append({'coordinates': _polygon_coords(sub_poly)})
    return records


def _assign_operator_colors(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if df is None or df.empty:
        return df, {}

    color_cycle = cycle(CUSTOM_PALETTE)
    color_lookup: dict[str, str] = {}
    rgba_values: list[list[int]] = []

    for operator in df.get('OperatorName', pd.Series(dtype=object)):
        key = str(operator) if pd.notna(operator) and str(operator).strip() else 'Unknown'
        if key not in color_lookup:
            color_lookup[key] = next(color_cycle)
        rgba_values.append(_hex_to_rgba(color_lookup[key], alpha=180))

    df = df.copy()
    if rgba_values:
        df['color_r'] = [val[0] for val in rgba_values]
        df['color_g'] = [val[1] for val in rgba_values]
        df['color_b'] = [val[2] for val in rgba_values]
        df['color_a'] = [val[3] for val in rgba_values]
    else:
        df['color_r'] = df['color_g'] = df['color_b'] = df['color_a'] = 180

    return df, color_lookup


def build_pydeck_map(
    filtered_df: gpd.GeoDataFrame,
    selected_play_layers: list[str],
    selected_company_layers: list[str],
) -> tuple[pdk.Deck | None, list[tuple[str, str]]]:
    layers: list[pdk.Layer] = []
    legend_entries: list[tuple[str, str]] = []

    view_state = pdk.ViewState(latitude=55.0, longitude=-106.0, zoom=3.5, pitch=0)

    if filtered_df is not None and not filtered_df.empty:
        map_df = filtered_df.dropna(subset=['SurfaceLatitude', 'SurfaceLongitude']).copy()
        if not map_df.empty:
            map_df, color_lookup = _assign_operator_colors(map_df)
            if 'FirstProdDate' in map_df.columns:
                map_df['FirstProdDate'] = map_df['FirstProdDate'].dt.strftime('%Y-%m-%d')
            tooltip_html = (
                "<b>Well:</b> {WellName}<br/>"
                "<b>Operator:</b> {OperatorName}<br/>"
                "<b>Field:</b> {FieldName}<br/>"
                "<b>Status:</b> {CurrentStatus}<br/>"
                "<b>First Prod:</b> {FirstProdDate}"
            )
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=map_df,
                    get_position="[SurfaceLongitude, SurfaceLatitude]",
                    get_radius=350,
                    get_fill_color="[color_r, color_g, color_b, color_a]",
                    pickable=True,
                    auto_highlight=True,
                )
            )
            if not map_df[['SurfaceLatitude', 'SurfaceLongitude']].isna().all().all():
                view_state = pdk.ViewState(
                    latitude=float(map_df['SurfaceLatitude'].mean()),
                    longitude=float(map_df['SurfaceLongitude'].mean()),
                    zoom=4,
                    pitch=0,
                )
        else:
            tooltip_html = None
    else:
        tooltip_html = None

    poly_color_cycle = cycle(CUSTOM_PALETTE)
    for layer_info in play_subplay_layers_list_global:
        if layer_info['name'] in selected_play_layers:
            records = _geodf_to_polygon_records(layer_info['data'])
            if records:
                hex_color = next(poly_color_cycle)
                rgba = _hex_to_rgba(hex_color, alpha=70)
                poly_df = pd.DataFrame(records)
                poly_df['fill_r'], poly_df['fill_g'], poly_df['fill_b'], poly_df['fill_a'] = rgba
                poly_df['line_r'], poly_df['line_g'], poly_df['line_b'] = rgba[:3]
                layers.append(
                    pdk.Layer(
                        "PolygonLayer",
                        data=poly_df,
                        get_polygon="coordinates",
                        get_fill_color="[fill_r, fill_g, fill_b, fill_a]",
                        get_line_color="[line_r, line_g, line_b, 200]",
                        line_width_min_pixels=1,
                        pickable=False,
                    )
                )
                legend_entries.append((f"Play/Subplay: {layer_info['name']}", hex_color))

    company_color_cycle = cycle(reversed(CUSTOM_PALETTE))
    for layer_info in company_layers_list_global:
        if layer_info['name'] in selected_company_layers:
            records = _geodf_to_polygon_records(layer_info['data'])
            if records:
                hex_color = next(company_color_cycle)
                rgba = _hex_to_rgba(hex_color, alpha=90)
                poly_df = pd.DataFrame(records)
                poly_df['fill_r'], poly_df['fill_g'], poly_df['fill_b'], poly_df['fill_a'] = rgba
                poly_df['line_r'], poly_df['line_g'], poly_df['line_b'] = rgba[:3]
                layers.append(
                    pdk.Layer(
                        "PolygonLayer",
                        data=poly_df,
                        get_polygon="coordinates",
                        get_fill_color="[fill_r, fill_g, fill_b, fill_a]",
                        get_line_color="[line_r, line_g, line_b, 220]",
                        line_width_min_pixels=1,
                        pickable=False,
                    )
                )
                legend_entries.append((f"Company Acreage: {layer_info['name']}", hex_color))

    if not layers:
        return None, legend_entries

    deck = pdk.Deck(
        map_provider="carto",
        map_style="light",
        initial_view_state=view_state,
        layers=layers,
        tooltip={"html": tooltip_html} if tooltip_html else None,
    )
    return deck, legend_entries


def fetch_single_well_production(gsl_uwi_std: str) -> pd.DataFrame:
    if not gsl_uwi_std or gsl_uwi_std.strip() == "":
        return pd.DataFrame()

    if engine is None:
        connect_to_db()
    if engine is None:
        raise RuntimeError("Database connection not available.")

    sql_query = text(
        """
        SELECT GSL_UWI, YEAR, PRODUCT_TYPE, ACTIVITY_TYPE,
               JAN_VOLUME, FEB_VOLUME, MAR_VOLUME, APR_VOLUME,
               MAY_VOLUME, JUN_VOLUME, JUL_VOLUME, AUG_VOLUME,
               SEP_VOLUME, OCT_VOLUME, NOV_VOLUME, DEC_VOLUME
        FROM PDEN_VOL_BY_MONTH
        WHERE GSL_UWI = :uwi
          AND ACTIVITY_TYPE = 'PRODUCTION'
          AND PRODUCT_TYPE IN ('OIL', 'CND', 'GAS')
        """
    )

    well_prod_raw = read_sql_resilient(sql_query, params={"uwi": gsl_uwi_std})
    if well_prod_raw.empty:
        return pd.DataFrame()

    well_prod_raw.columns = [col.lower() for col in well_prod_raw.columns]
    month_cols = {f"{m}_volume": i + 1 for i, m in enumerate(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])}
    prod_long = pd.melt(
        well_prod_raw,
        id_vars=['gsl_uwi', 'year', 'product_type'],
        value_vars=list(month_cols.keys()),
        var_name='month_col',
        value_name='volume',
    )
    prod_long['month'] = prod_long['month_col'].map(month_cols)
    prod_long['prod_date'] = pd.to_datetime(prod_long['year'].astype(str) + '-' + prod_long['month'].astype(str) + '-01')

    is_liquid = prod_long['product_type'].isin(['OIL', 'CND'])
    is_gas = prod_long['product_type'] == 'GAS'
    prod_long.loc[is_liquid, 'volume_converted'] = prod_long['volume'] * M3_TO_BBL
    prod_long.loc[is_gas, 'volume_converted'] = prod_long['volume'] * E3M3_TO_MCF
    prod_long = prod_long.dropna(subset=['volume_converted'])

    prod_pivot = prod_long.pivot_table(
        index=['gsl_uwi', 'prod_date'],
        columns='product_type',
        values='volume_converted',
        aggfunc='sum',
    ).reset_index()

    prod_pivot = prod_pivot.rename(
        columns={
            'OIL': 'MonthlyOilTrueBBL',
            'CND': 'MonthlyCondensateBBL',
            'GAS': 'MonthlyGasMCF',
        }
    ).fillna(0)

    return prod_pivot


def process_single_well_production(
    prod_df: pd.DataFrame,
    product_types: list[str],
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None,
) -> pd.DataFrame:
    if prod_df is None or prod_df.empty:
        return pd.DataFrame()

    df = prod_df.copy()
    if date_range:
        start_date, end_date = date_range
        df = df[df['prod_date'].between(start_date, end_date, inclusive='both')]
    df = df.sort_values('prod_date')
    if df.empty:
        return df

    df['DaysInMonth'] = df['prod_date'].dt.days_in_month
    df['MonthlyOilTrueBBL'] = df.get('MonthlyOilTrueBBL', 0)
    df['MonthlyCondensateBBL'] = df.get('MonthlyCondensateBBL', 0)
    df['MonthlyGasMCF'] = df.get('MonthlyGasMCF', 0)

    df['OilTrueRateBBLD'] = df['MonthlyOilTrueBBL'] / df['DaysInMonth']
    df['CondensateRateBBLD'] = df['MonthlyCondensateBBL'] / df['DaysInMonth']
    df['GasRateMCFD'] = df['MonthlyGasMCF'] / df['DaysInMonth']

    include_oil = 'OIL' in product_types or 'BOE' in product_types
    include_cnd = 'CND' in product_types or 'BOE' in product_types
    include_gas = 'GAS' in product_types or 'BOE' in product_types

    oil_vol = df['MonthlyOilTrueBBL'] if include_oil else 0
    cnd_vol = df['MonthlyCondensateBBL'] if include_cnd else 0
    gas_vol = df['MonthlyGasMCF'] if include_gas else 0

    df['BOERateBBLD'] = (oil_vol + cnd_vol + (gas_vol / MCF_PER_BOE)) / df['DaysInMonth']
    df['CumOilTrueBBL'] = df['MonthlyOilTrueBBL'].cumsum()
    df['CumCondensateBBL'] = df['MonthlyCondensateBBL'].cumsum()
    df['CumGasMCF'] = df['MonthlyGasMCF'].cumsum()

    return df


def make_single_well_rate_chart(df: pd.DataFrame, product_types: list[str], well_label: str) -> go.Figure:
    fig = go.Figure()
    if df is None or df.empty:
        fig.update_layout(title="No production data available for selected well.")
        return fig

    mapping = [
        ('OIL', 'OilTrueRateBBLD', 'Oil Rate (BBL/d)'),
        ('CND', 'CondensateRateBBLD', 'Condensate Rate (BBL/d)'),
        ('GAS', 'GasRateMCFD', 'Gas Rate (MCF/d)'),
        ('BOE', 'BOERateBBLD', 'BOE Rate (BBL/d)')
    ]
    for product, column, label in mapping:
        if product in product_types and column in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['prod_date'],
                    y=df[column],
                    mode='lines+markers',
                    name=label,
                )
            )

    fig.update_layout(
        title=f"Daily Production Rate - {well_label}",
        xaxis_title="Production Month",
        yaxis_title="Rate",
        hovermode="x unified",
        legend_title="Product",
    )
    return fig


def make_single_well_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame({"Message": ["No production data to display."]})

    cols_to_show = {
        'prod_date': 'Prod. Month',
        'MonthlyOilTrueBBL': 'Oil (BBL/Month)',
        'MonthlyCondensateBBL': 'Cond. (BBL/Month)',
        'MonthlyGasMCF': 'Gas (MCF/Month)',
        'OilTrueRateBBLD': 'Oil Rate (BBL/d)',
        'CondensateRateBBLD': 'Cond. Rate (BBL/d)',
        'GasRateMCFD': 'Gas Rate (MCF/d)',
        'BOERateBBLD': 'BOE Rate (BBL/d)',
        'CumOilTrueBBL': 'Cum. Oil (BBL)',
        'CumCondensateBBL': 'Cum. Cond. (BBL)',
        'CumGasMCF': 'Cum. Gas (MCF)',
    }
    df_display = df[list(cols_to_show.keys())].copy()
    df_display.rename(columns=cols_to_show, inplace=True)
    df_display['Prod. Month'] = df_display['Prod. Month'].dt.strftime('%Y-%m')
    return df_display.round(2)


def single_well_download_csv(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "No data available"
    export_df = df.copy()
    export_df['prod_date'] = export_df['prod_date'].dt.strftime('%Y-%m')
    return export_df.to_csv(index=False)


def compute_filtered_group_metrics(
    filtered_wells: gpd.GeoDataFrame,
    selected_products: list[str],
    breakout_col: str,
) -> dict | None:
    if filtered_wells is None or filtered_wells.empty:
        raise ValueError("No wells selected by current filters.")
    if not selected_products:
        raise ValueError("Select at least one product type for analysis.")
    if 'LateralLength' not in filtered_wells.columns:
        raise ValueError("LateralLength column not available in well data.")

    target_columns = [
        'GSL_UWI_Std',
        'OperatorName',
        'LateralLength',
        'Formation',
        'FieldName',
        'ProvinceState',
        'FirstProdDate',
    ]
    missing_cols = [col for col in target_columns if col not in filtered_wells.columns]
    if missing_cols:
        raise RuntimeError(
            "Filtered wells dataset is missing required columns: " + ", ".join(missing_cols)
        )

    target_uwis_df = filtered_wells[target_columns].copy()
    target_uwis_df['FirstProdDate'] = pd.to_datetime(target_uwis_df['FirstProdDate'], errors='coerce')

    first_prod_year = target_uwis_df['FirstProdDate'].dt.year.astype('Int64')
    target_uwis_df['FirstProdYear'] = first_prod_year.astype(str).replace({'<NA>': 'Unknown'})
    replacement_map = {'nan': np.nan, 'None': np.nan, '': np.nan, 'NaT': np.nan}
    target_uwis_df['OperatorName'] = (
        target_uwis_df['OperatorName'].astype(str).replace(replacement_map).fillna('Unknown')
    )
    target_uwis_df['Formation'] = (
        target_uwis_df['Formation'].astype(str).replace(replacement_map).fillna('Unknown')
    )
    target_uwis_df['FieldName'] = (
        target_uwis_df['FieldName'].astype(str).replace(replacement_map).fillna('Unknown')
    )
    target_uwis_df['ProvinceState'] = (
        target_uwis_df['ProvinceState'].astype(str).replace(replacement_map).fillna('Unknown')
    )
    target_uwis = target_uwis_df['GSL_UWI_Std'].dropna().astype(str).unique().tolist()
    if not target_uwis:
        raise ValueError("No valid wells to fetch production for.")

    if engine is None:
        connect_to_db()
    if engine is None:
        raise RuntimeError("Database connection not available.")

    all_prod_data: list[pd.DataFrame] = []
    for i in range(0, len(target_uwis), 300):
        batch_uwis = target_uwis[i:i + 300]
        sql_prod = text(
            """
            SELECT GSL_UWI, YEAR, PRODUCT_TYPE, JAN_VOLUME, FEB_VOLUME, MAR_VOLUME, APR_VOLUME,
                   MAY_VOLUME, JUN_VOLUME, JUL_VOLUME, AUG_VOLUME, SEP_VOLUME, OCT_VOLUME,
                   NOV_VOLUME, DEC_VOLUME
            FROM PDEN_VOL_BY_MONTH
            WHERE GSL_UWI IN :uwis AND ACTIVITY_TYPE = 'PRODUCTION'
            """
        ).bindparams(bindparam("uwis", expanding=True))
        batch_df = read_sql_resilient(sql_prod, params={"uwis": list(batch_uwis)})
        if not batch_df.empty:
            all_prod_data.append(batch_df)

    if not all_prod_data:
        raise ValueError("No production data found for filtered wells.")

    full_prod_df = pd.concat(all_prod_data, ignore_index=True)
    full_prod_df = clean_df_colnames(full_prod_df, "Filtered Group Production")
    if 'GSL_UWI' in full_prod_df.columns:
        full_prod_df['GSL_UWI_Std'] = standardize_uwi(full_prod_df['GSL_UWI'])
    elif 'GSL_UWI_STD' in full_prod_df.columns:
        full_prod_df['GSL_UWI_Std'] = full_prod_df['GSL_UWI_STD']
    else:
        raise RuntimeError("Unable to identify GSL_UWI_Std in production data.")

    effective_products = set(selected_products)
    if 'BOE' in effective_products:
        effective_products.update(['OIL', 'CND', 'GAS'])
    full_prod_df['PRODUCT_TYPE'] = full_prod_df['PRODUCT_TYPE'].astype(str).str.upper()
    full_prod_df = full_prod_df[full_prod_df['PRODUCT_TYPE'].isin(effective_products)]
    if full_prod_df.empty:
        raise ValueError("No production volumes available for selected product filters.")

    prod_merged = pd.merge(
        full_prod_df,
        target_uwis_df,
        on='GSL_UWI_Std',
        how='left',
    )
    if breakout_col not in prod_merged.columns:
        raise RuntimeError(f"Breakout column {breakout_col} not found after merge.")

    if breakout_col in {'OperatorName', 'Formation', 'FieldName', 'ProvinceState', 'FirstProdYear'}:
        prod_merged[breakout_col] = (
            prod_merged[breakout_col]
            .astype(str)
            .replace({'nan': np.nan, 'None': np.nan, '<NA>': np.nan, '': np.nan, 'NaT': np.nan})
            .fillna('Unknown')
        )

    month_cols = [f"{m.upper()}_VOLUME" for m in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']]
    base_id_vars = [
        'GSL_UWI_Std',
        'YEAR',
        'PRODUCT_TYPE',
        'LateralLength',
        'OperatorName',
        'Formation',
        'FieldName',
        'ProvinceState',
    ]
    id_vars = list(dict.fromkeys(base_id_vars + [breakout_col]))
    missing_id_vars = [col for col in id_vars if col not in prod_merged.columns]
    if missing_id_vars:
        raise RuntimeError(
            "Production dataset is missing required columns for melting: " + ", ".join(missing_id_vars)
        )

    prod_long = pd.melt(
        prod_merged,
        id_vars=id_vars,
        value_vars=month_cols,
        var_name='MONTH_COL',
        value_name='VOLUME'
    )

    if breakout_col in prod_long.columns:
        prod_long[breakout_col] = (
            prod_long[breakout_col]
            .astype(str)
            .replace({'nan': np.nan, 'None': np.nan, '<NA>': np.nan, '': np.nan, 'NaT': np.nan})
            .fillna('Unknown')
        )
    prod_long = prod_long.dropna(subset=['VOLUME'])
    prod_long['VOLUME'] = pd.to_numeric(prod_long['VOLUME'], errors='coerce').fillna(0)
    prod_long['MONTH_NUM'] = prod_long['MONTH_COL'].str.extract(r"(\w+)_").iloc[:, 0].map({m.upper(): i for i, m in enumerate(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], start=1)})
    prod_long['PROD_DATE'] = pd.to_datetime(prod_long['YEAR'].astype(int).astype(str) + '-' + prod_long['MONTH_NUM'].astype(int).astype(str) + '-01')

    prod_long['MonthlyOilTrueBBL_raw'] = np.where(prod_long['PRODUCT_TYPE'] == 'OIL', prod_long['VOLUME'] * M3_TO_BBL, 0)
    prod_long['MonthlyCondensateBBL_raw'] = np.where(prod_long['PRODUCT_TYPE'] == 'CND', prod_long['VOLUME'] * M3_TO_BBL, 0)
    prod_long['MonthlyGasMCF_raw'] = np.where(prod_long['PRODUCT_TYPE'] == 'GAS', prod_long['VOLUME'] * E3M3_TO_MCF, 0)

    prod_long['DaysInMonth'] = prod_long['PROD_DATE'].dt.days_in_month
    prod_long['DailyBOE_well'] = (prod_long['MonthlyOilTrueBBL_raw'] + prod_long['MonthlyCondensateBBL_raw'] + (prod_long['MonthlyGasMCF_raw'] / MCF_PER_BOE)) / prod_long['DaysInMonth']
    prod_long['DailyBOE_per_1000ft_well'] = np.where(
        prod_long['LateralLength'] > 0,
        prod_long['DailyBOE_well'] / (prod_long['LateralLength'] / 1000),
        np.nan,
    )

    peak_info = prod_long.groupby('GSL_UWI_Std')['DailyBOE_well'].idxmax()
    peak_dates = prod_long.loc[peak_info, ['GSL_UWI_Std', 'PROD_DATE']].rename(columns={'PROD_DATE': 'PeakProdDate'})
    prod_from_peak = pd.merge(prod_long, peak_dates, on='GSL_UWI_Std', how='left')
    prod_from_peak = prod_from_peak[prod_from_peak['PROD_DATE'] >= prod_from_peak['PeakProdDate']]
    prod_from_peak['MonthOnProd'] = ((prod_from_peak['PROD_DATE'] - prod_from_peak['PeakProdDate']).dt.days / AVG_DAYS_PER_MONTH).round().astype(int) + 1

    normalized_data = prod_from_peak.groupby([breakout_col, 'MonthOnProd']).agg(
        AvgNormBOERate_per_1000ft=('DailyBOE_per_1000ft_well', 'mean'),
        WellCount=('GSL_UWI_Std', 'nunique')
    ).reset_index()
    normalized_data['CumNormBOE_per_1000ft'] = normalized_data.groupby(breakout_col)['AvgNormBOERate_per_1000ft'].transform(lambda x: (x.fillna(0) * AVG_DAYS_PER_MONTH).cumsum())

    calendar_data = prod_long.groupby([breakout_col, 'PROD_DATE']).agg(
        TotalMonthlyOilTrueBBL_sum=('MonthlyOilTrueBBL_raw', 'sum'),
        TotalMonthlyCondensateBBL_sum=('MonthlyCondensateBBL_raw', 'sum'),
        TotalMonthlyGasMCF_sum=('MonthlyGasMCF_raw', 'sum'),
        WellCount=('GSL_UWI_Std', 'nunique')
    ).reset_index()
    calendar_data['AvgDailyBOE_perGroup'] = (
        (calendar_data['TotalMonthlyOilTrueBBL_sum'] + calendar_data['TotalMonthlyCondensateBBL_sum'] + (calendar_data['TotalMonthlyGasMCF_sum'] / MCF_PER_BOE))
        / calendar_data['PROD_DATE'].dt.days_in_month
    )
    calendar_data['AvgDailyBOE_perGroup_per1000ft'] = np.nan

    cumulative_boe = calendar_data.copy()
    cumulative_boe['MonthlyBOE'] = (
        calendar_data['TotalMonthlyOilTrueBBL_sum'] +
        calendar_data['TotalMonthlyCondensateBBL_sum'] +
        (calendar_data['TotalMonthlyGasMCF_sum'] / MCF_PER_BOE)
    )
    cumulative_boe['CumBOE'] = cumulative_boe.groupby(breakout_col)['MonthlyBOE'].cumsum()

    summary_table = prod_long.groupby([breakout_col, 'PROD_DATE']).agg(
        TotalMonthlyOilTrueBBL=('MonthlyOilTrueBBL_raw', 'sum'),
        TotalMonthlyCondensateBBL=('MonthlyCondensateBBL_raw', 'sum'),
        TotalMonthlyGasMCF=('MonthlyGasMCF_raw', 'sum'),
        WellCount=('GSL_UWI_Std', 'nunique')
    ).reset_index()
    summary_table['AvgDailyBOE'] = (
        (summary_table['TotalMonthlyOilTrueBBL'] + summary_table['TotalMonthlyCondensateBBL'] + (summary_table['TotalMonthlyGasMCF'] / MCF_PER_BOE))
        / summary_table['PROD_DATE'].dt.days_in_month
    )

    return {
        'normalized': normalized_data,
        'calendar': calendar_data,
        'cumulative_boe': cumulative_boe,
        'summary_table': summary_table,
    }


def make_filtered_group_plots(metrics: dict, breakout_label: str) -> tuple[go.Figure, go.Figure, go.Figure]:
    empty_fig = go.Figure().update_layout(title="No data available.")
    if not metrics:
        return empty_fig, empty_fig, empty_fig

    normalized = metrics.get('normalized')
    calendar = metrics.get('calendar')
    cumulative = metrics.get('cumulative_boe')

    norm_fig = go.Figure()
    if normalized is not None and not normalized.empty:
        for key, group_df in normalized.groupby(breakout_label):
            norm_fig.add_trace(
                go.Scatter(
                    x=group_df['MonthOnProd'],
                    y=group_df['AvgNormBOERate_per_1000ft'],
                    mode='lines+markers',
                    name=str(key),
                )
            )
        norm_fig.update_layout(
            title=f"Average Daily BOE per 1000ft by {breakout_label} (Normalized)",
            xaxis_title="Month On Production",
            yaxis_title="Avg Daily BOE / 1000ft",
            hovermode="x unified",
        )
    else:
        norm_fig = empty_fig

    cum_fig = go.Figure()
    if cumulative is not None and not cumulative.empty:
        for key, group_df in cumulative.groupby(breakout_label):
            cum_fig.add_trace(
                go.Scatter(
                    x=group_df['PROD_DATE'],
                    y=group_df['CumBOE'],
                    mode='lines+markers',
                    name=str(key),
                )
            )
        cum_fig.update_layout(
            title=f"Cumulative BOE by {breakout_label}",
            xaxis_title="Production Month",
            yaxis_title="Cumulative BOE",
            hovermode="x unified",
        )
    else:
        cum_fig = empty_fig

    cal_fig = go.Figure()
    if calendar is not None and not calendar.empty:
        for key, group_df in calendar.groupby(breakout_label):
            cal_fig.add_trace(
                go.Scatter(
                    x=group_df['PROD_DATE'],
                    y=group_df['AvgDailyBOE_perGroup'],
                    mode='lines+markers',
                    name=str(key),
                )
            )
        cal_fig.update_layout(
            title=f"Average Daily BOE by {breakout_label} (Calendar Time)",
            xaxis_title="Production Month",
            yaxis_title="Avg Daily BOE",
            hovermode="x unified",
        )
    else:
        cal_fig = empty_fig

    return norm_fig, cum_fig, cal_fig


def prepare_filtered_group_table(metrics: dict, breakout_label: str) -> pd.DataFrame:
    summary_table = metrics.get('summary_table') if metrics else None
    if summary_table is None or summary_table.empty:
        return pd.DataFrame({"Message": ["No production data to summarize."]})

    table_df = summary_table.copy()
    table_df = table_df.rename(columns={
        breakout_label: breakout_label,
        'PROD_DATE': 'Prod_Month',
        'TotalMonthlyOilTrueBBL': 'Total_Oil_BBL',
        'TotalMonthlyCondensateBBL': 'Total_Condensate_BBL',
        'TotalMonthlyGasMCF': 'Total_Gas_MCF',
        'WellCount': 'Well_Count',
        'AvgDailyBOE': 'Avg_Daily_BOE',
    })
    table_df['Prod_Month'] = table_df['Prod_Month'].dt.strftime('%Y-%m')
    return table_df


def arps_hyperbolic(t, qi, di, b):
    return qi / (1 + b * di * t)**(1 / b)


def arps_exponential(t, qi, di):
    return qi * np.exp(-di * t)


def arps_harmonic(t, qi, di):
    return qi / (1 + di * t)


def compute_type_curve_analysis(
    filtered_wells: gpd.GeoDataFrame,
    selected_products: list[str],
    model_choice: str,
    arps_product_choice: str,
) -> dict:
    if filtered_wells is None or filtered_wells.empty:
        raise ValueError("No wells selected by current filters for type curve analysis.")
    if not selected_products:
        raise ValueError("Select at least one product type for analysis.")

    target_uwis = filtered_wells['GSL_UWI_Std'].dropna().astype(str).unique().tolist()
    if not target_uwis:
        raise ValueError("No valid well identifiers available for type curve analysis.")
    if len(target_uwis) > 1000:
        raise ValueError(f"Too many wells selected ({len(target_uwis)}). Please filter to fewer than 1000 wells.")

    effective_products = set(selected_products)
    if 'BOE' in effective_products:
        effective_products.update({'OIL', 'CND', 'GAS'})
        effective_products.discard('BOE')

    if arps_product_choice == 'Oil':
        db_products = [p for p in ('OIL', 'CND') if p in effective_products]
    else:
        db_products = ['GAS'] if 'GAS' in effective_products else []
    if not db_products:
        raise ValueError("No production data available for the selected Arps product with current filters.")

    if engine is None:
        connect_to_db()
    if engine is None:
        raise RuntimeError("Database connection not available for type curve analysis.")

    all_prod_data: list[pd.DataFrame] = []
    for i in range(0, len(target_uwis), 300):
        batch_uwis = target_uwis[i:i + 300]
        sql_prod = text(
            """
            SELECT GSL_UWI, YEAR, PRODUCT_TYPE, JAN_VOLUME, FEB_VOLUME, MAR_VOLUME, APR_VOLUME,
                   MAY_VOLUME, JUN_VOLUME, JUL_VOLUME, AUG_VOLUME, SEP_VOLUME, OCT_VOLUME,
                   NOV_VOLUME, DEC_VOLUME
            FROM PDEN_VOL_BY_MONTH
            WHERE GSL_UWI IN :uwis AND ACTIVITY_TYPE = 'PRODUCTION'
            """
        ).bindparams(bindparam("uwis", expanding=True))
        batch_df = read_sql_resilient(sql_prod, params={"uwis": list(batch_uwis)})
        if not batch_df.empty:
            all_prod_data.append(batch_df)

    if not all_prod_data:
        raise ValueError("No production data found for the selected wells.")

    full_prod_df = pd.concat(all_prod_data, ignore_index=True)
    full_prod_df = clean_df_colnames(full_prod_df, "Arps Type Curve")
    if 'GSL_UWI' in full_prod_df.columns:
        full_prod_df['GSL_UWI_STD'] = standardize_uwi(full_prod_df['GSL_UWI'])
    elif 'GSL_UWI_STD' not in full_prod_df.columns:
        raise RuntimeError("Unable to determine GSL_UWI_Std for type curve data.")

    full_prod_df['PRODUCT_TYPE'] = full_prod_df['PRODUCT_TYPE'].astype(str).str.upper()
    full_prod_df = full_prod_df[full_prod_df['PRODUCT_TYPE'].isin(db_products)]
    if full_prod_df.empty:
        raise ValueError("No production data available for selected product types after filtering.")

    month_cols = [f"{m.upper()}_VOLUME" for m in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']]
    required_cols = {'GSL_UWI_STD', 'YEAR', 'PRODUCT_TYPE'} | set(month_cols)
    if not required_cols.issubset(full_prod_df.columns):
        missing = required_cols.difference(full_prod_df.columns)
        raise RuntimeError(f"Missing required production columns for type curve fitting: {sorted(missing)}")

    prod_long = pd.melt(
        full_prod_df,
        id_vars=['GSL_UWI_STD', 'YEAR', 'PRODUCT_TYPE'],
        value_vars=month_cols,
        var_name='MONTH_COL',
        value_name='VOLUME',
    )
    prod_long['MONTH_NUM'] = prod_long['MONTH_COL'].str.extract(r"(\w+)_").iloc[:, 0].map({m.upper(): i for i, m in enumerate(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], start=1)})
    prod_long['PROD_DATE'] = pd.to_datetime(prod_long['YEAR'].astype(int).astype(str) + '-' + prod_long['MONTH_NUM'].astype(int).astype(str) + '-01')
    prod_long['VOLUME'] = pd.to_numeric(prod_long['VOLUME'], errors='coerce').fillna(0)

    if arps_product_choice == 'Oil':
        prod_long['ConvertedVolume'] = prod_long['VOLUME'] * M3_TO_BBL
    else:
        prod_long['ConvertedVolume'] = prod_long['VOLUME'] * E3M3_TO_MCF

    prod_long['DaysInMonth'] = prod_long['PROD_DATE'].dt.days_in_month
    prod_long['DailyRate'] = prod_long['ConvertedVolume'] / prod_long['DaysInMonth']

    peak_info = prod_long.groupby('GSL_UWI_STD')['DailyRate'].idxmax()
    peak_dates = prod_long.loc[peak_info, ['GSL_UWI_STD', 'PROD_DATE']].rename(columns={'PROD_DATE': 'PeakProdDate'})
    prod_from_peak = pd.merge(prod_long, peak_dates, on='GSL_UWI_STD', how='left')
    prod_from_peak = prod_from_peak[prod_from_peak['PROD_DATE'] >= prod_from_peak['PeakProdDate']]
    prod_from_peak['MonthOnProd'] = ((prod_from_peak['PROD_DATE'] - prod_from_peak['PeakProdDate']).dt.days / AVG_DAYS_PER_MONTH).round().astype(int) + 1

    decline_df = prod_from_peak.groupby('MonthOnProd').agg(
        AvgDailyRate=('DailyRate', 'mean'),
        WellCount=('GSL_UWI_STD', 'nunique')
    ).reset_index()
    decline_df = decline_df.sort_values('MonthOnProd')
    if decline_df.empty:
        raise ValueError("Insufficient data after normalization for type curve fitting.")

    months = decline_df['MonthOnProd'].to_numpy(dtype=float)
    rates = decline_df['AvgDailyRate'].to_numpy(dtype=float)
    positive_mask = rates > 0
    months = months[positive_mask]
    rates = rates[positive_mask]
    if len(months) < 4:
        raise ValueError("Not enough positive rate points to fit the Arps decline curve.")

    if model_choice == 'hyperbolic':
        fit_func = arps_hyperbolic
        p0 = (rates.max(), 0.8 / AVG_DAYS_PER_MONTH, 1.0)
        bounds = ((1e-6, 1e-6, 0.01), (rates.max() * 2, 1.0, 3.0))
    elif model_choice == 'harmonic':
        fit_func = arps_harmonic
        p0 = (rates.max(), 0.8 / AVG_DAYS_PER_MONTH)
        bounds = ((1e-6, 1e-6), (rates.max() * 2, 1.0))
    else:
        fit_func = arps_exponential
        p0 = (rates.max(), 0.8 / AVG_DAYS_PER_MONTH)
        bounds = ((1e-6, 1e-6), (rates.max() * 2, 1.0))

    try:
        fit_params, _ = curve_fit(fit_func, months, rates, p0=p0, bounds=bounds, maxfev=10000)
    except Exception as exc:
        raise RuntimeError(f"Failed to fit {model_choice} Arps curve: {exc}")

    prediction_months = np.arange(1, 361)
    if model_choice == 'hyperbolic':
        predicted_rates = arps_hyperbolic(prediction_months, *fit_params)
    elif model_choice == 'harmonic':
        predicted_rates = arps_harmonic(prediction_months, *fit_params)
    else:
        predicted_rates = arps_exponential(prediction_months, *fit_params)

    decline_df = decline_df[decline_df['MonthOnProd'].isin(months)]
    decline_df = decline_df.assign(AvgDailyRate=rates)

    eur_25yr = np.sum(predicted_rates[: int((25 * 12))]) * AVG_DAYS_PER_MONTH

    def compute_decline_lines(params: np.ndarray) -> list[str]:
        lines: list[str] = []
        if model_choice == 'hyperbolic':
            qi, di, b = params
            lines.append(f"qi = {qi:,.0f} bbl/d, di = {di * 365:,.2f} 1/yr, b = {b:,.2f}")
        else:
            qi, di = params
            lines.append(f"qi = {qi:,.0f} rate units, di = {di * 365:,.2f} 1/yr")
        return lines

    return {
        'decline_df': decline_df,
        'prediction_months': prediction_months,
        'predicted_rates': predicted_rates,
        'fit_params': fit_params,
        'model_choice': model_choice,
        'arps_product_choice': arps_product_choice,
        'eur_25yr': eur_25yr,
        'decline_lines': compute_decline_lines(fit_params),
    }


def make_type_curve_plot(result: dict) -> go.Figure:
    fig = go.Figure()
    decline_df = result.get('decline_df')
    if decline_df is not None and not decline_df.empty:
        fig.add_trace(
            go.Scatter(
                x=decline_df['MonthOnProd'],
                y=decline_df['AvgDailyRate'],
                mode='markers',
                name='Observed Avg Rate',
            )
        )
    prediction_months = result.get('prediction_months')
    predicted_rates = result.get('predicted_rates')
    if prediction_months is not None and predicted_rates is not None:
        fig.add_trace(
            go.Scatter(
                x=prediction_months,
                y=predicted_rates,
                mode='lines',
                name=f"Fitted {result.get('model_choice', '').title()} Curve",
            )
        )
    fig.update_layout(
        title="Arps Decline Curve (Peak Normalized)",
        xaxis_title="Months Since Peak",
        yaxis_title="Average Daily Rate",
        hovermode="x unified",
    )
    return fig


def format_type_curve_summary(result: dict) -> str:
    lines = ["Fitted Decline Parameters:"]
    lines.extend(result.get('decline_lines', []))
    eur = result.get('eur_25yr')
    if eur is not None:
        unit = 'BBL' if result.get('arps_product_choice') == 'Oil' else 'MCF'
        lines.append(f"Estimated 25-year EUR: {eur:,.0f} {unit}")
    return "\n".join(lines)


def prepare_type_curve_table(result: dict) -> pd.DataFrame:
    decline_df = result.get('decline_df')
    if decline_df is None or decline_df.empty:
        return pd.DataFrame({"Message": ["No data available for type curve."]})
    table_df = decline_df.copy()
    table_df.rename(columns={'MonthOnProd': 'Month_On_Production', 'AvgDailyRate': 'Average_Daily_Rate', 'WellCount': 'Well_Count'}, inplace=True)
    return table_df


def compute_operator_group_data(
    selected_operator_codes: list[str],
    selected_products: list[str],
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None,
) -> pd.DataFrame:
    if not selected_operator_codes:
        raise ValueError("Please select at least one operator.")
    if not selected_products:
        raise ValueError("Select at least one product type for analysis.")

    codes_series = normalize_operator_code_series(pd.Series(selected_operator_codes))
    operator_codes = [code for code in codes_series.dropna().unique().tolist() if str(code).strip()]
    if not operator_codes:
        raise ValueError("Unable to normalize selected operator codes.")

    if engine is None:
        connect_to_db()
    if engine is None:
        raise RuntimeError("Database connection not available.")

    operator_sql = text(
        """
        SELECT DISTINCT W.GSL_UWI, W.OPERATOR AS OPERATOR_CODE
        FROM WELL W
        WHERE """ + _base_well_filters() + " AND W.OPERATOR IN :operator_codes AND W.GSL_UWI IS NOT NULL"
    ).bindparams(bindparam("operator_codes", expanding=True))

    wells_df = read_sql_resilient(operator_sql, params={"operator_codes": operator_codes})
    wells_df = clean_df_colnames(wells_df, "Operator Well Mapping")
    if wells_df.empty:
        raise ValueError("No wells found for selected operators.")

    if 'GSL_UWI' in wells_df.columns:
        wells_df['GSL_UWI_STD'] = standardize_uwi(wells_df['GSL_UWI'])
    if 'OPERATOR_CODE' in wells_df.columns:
        wells_df['OPERATOR_CODE'] = normalize_operator_code_series(wells_df['OPERATOR_CODE'])

    operator_lookup = get_operator_display_lookup()
    wells_df['OperatorName'] = wells_df.get('OPERATOR_CODE').map(operator_lookup).fillna(wells_df.get('OPERATOR_CODE'))
    wells_df['OperatorName'] = wells_df['OperatorName'].fillna('Unknown')

    operator_wells = wells_df[['GSL_UWI_STD', 'OperatorName']].copy()
    operator_wells = operator_wells[operator_wells['GSL_UWI_STD'].notna()]
    operator_wells['OperatorName'] = operator_wells['OperatorName'].astype(str).str.strip()
    operator_wells.loc[operator_wells['OperatorName'] == '', 'OperatorName'] = 'Unknown'

    target_uwis = operator_wells['GSL_UWI_STD'].dropna().astype(str).unique().tolist()
    if not target_uwis:
        raise ValueError("No wells found for selected operators.")

    all_prod_data: list[pd.DataFrame] = []
    for i in range(0, len(target_uwis), 300):
        batch_uwis = target_uwis[i:i + 300]
        sql_prod = text(
            """
            SELECT GSL_UWI, YEAR, PRODUCT_TYPE, JAN_VOLUME, FEB_VOLUME, MAR_VOLUME, APR_VOLUME,
                   MAY_VOLUME, JUN_VOLUME, JUL_VOLUME, AUG_VOLUME, SEP_VOLUME, OCT_VOLUME,
                   NOV_VOLUME, DEC_VOLUME
            FROM PDEN_VOL_BY_MONTH
            WHERE GSL_UWI IN :uwis AND ACTIVITY_TYPE = 'PRODUCTION'
            """
        ).bindparams(bindparam("uwis", expanding=True))
        batch_df = read_sql_resilient(sql_prod, params={"uwis": list(batch_uwis)})
        if not batch_df.empty:
            all_prod_data.append(batch_df)

    if not all_prod_data:
        raise ValueError("No production data found for this operator group.")

    full_prod_df = pd.concat(all_prod_data, ignore_index=True)
    full_prod_df = clean_df_colnames(full_prod_df, "Operator Group Production")
    if 'GSL_UWI' in full_prod_df.columns:
        full_prod_df['GSL_UWI_STD'] = standardize_uwi(full_prod_df['GSL_UWI'])
    elif 'GSL_UWI_STD' not in full_prod_df.columns:
        raise RuntimeError("Unable to determine GSL_UWI_Std in production data.")

    month_cols = [f"{m.upper()}_VOLUME" for m in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']]
    prod_long = pd.melt(
        full_prod_df,
        id_vars=['GSL_UWI_STD', 'YEAR', 'PRODUCT_TYPE'],
        value_vars=month_cols,
        var_name='MONTH_COL',
        value_name='VOLUME',
    )
    prod_long['MONTH_NUM'] = prod_long['MONTH_COL'].str.extract(r"(\w+)_").iloc[:, 0].map({m.upper(): i for i, m in enumerate(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], start=1)})
    prod_long['PROD_DATE'] = pd.to_datetime(prod_long['YEAR'].astype(int).astype(str) + '-' + prod_long['MONTH_NUM'].astype(int).astype(str) + '-01')
    prod_long['VOLUME'] = pd.to_numeric(prod_long['VOLUME'], errors='coerce').fillna(0)

    prod_long = prod_long[prod_long['PRODUCT_TYPE'].isin([p for p in selected_products if p != 'BOE'])]
    prod_long = prod_long.merge(operator_wells, on='GSL_UWI_STD', how='left')

    if date_range:
        start_date, end_date = date_range
        prod_long = prod_long[prod_long['PROD_DATE'].between(start_date, end_date, inclusive='both')]
    if prod_long.empty:
        raise ValueError("No production data in selected date range.")

    prod_long['MonthlyOilTrueBBL'] = np.where(prod_long['PRODUCT_TYPE'].isin(['OIL', 'CND']), prod_long['VOLUME'] * M3_TO_BBL, 0)
    prod_long['MonthlyCondensateBBL'] = np.where(prod_long['PRODUCT_TYPE'] == 'CND', prod_long['VOLUME'] * M3_TO_BBL, 0)
    prod_long['MonthlyGasMCF'] = np.where(prod_long['PRODUCT_TYPE'] == 'GAS', prod_long['VOLUME'] * E3M3_TO_MCF, 0)

    prod_long['DaysInMonth'] = prod_long['PROD_DATE'].dt.days_in_month
    prod_long['OilRate'] = prod_long['MonthlyOilTrueBBL'] / prod_long['DaysInMonth']
    prod_long['CondRate'] = prod_long['MonthlyCondensateBBL'] / prod_long['DaysInMonth']
    prod_long['GasRate'] = prod_long['MonthlyGasMCF'] / prod_long['DaysInMonth']
    prod_long['BOERate'] = (prod_long['MonthlyOilTrueBBL'] + prod_long['MonthlyCondensateBBL'] + (prod_long['MonthlyGasMCF'] / MCF_PER_BOE)) / prod_long['DaysInMonth']

    grouped = prod_long.groupby(['OperatorName', 'PROD_DATE']).agg(
        MonthlyOilBBL=('MonthlyOilTrueBBL', 'sum'),
        MonthlyCondensateBBL=('MonthlyCondensateBBL', 'sum'),
        MonthlyGasMCF=('MonthlyGasMCF', 'sum'),
        AvgOilRateBBLD=('OilRate', 'mean'),
        AvgCndRateBBLD=('CondRate', 'mean'),
        AvgGasRateMCFD=('GasRate', 'mean'),
        AvgBOERateBBLD=('BOERate', 'mean'),
        WellCount=('GSL_UWI_STD', 'nunique'),
    ).reset_index()
    grouped['CumOilBBL'] = grouped.groupby('OperatorName')['MonthlyOilBBL'].cumsum()
    grouped['CumCondensateBBL'] = grouped.groupby('OperatorName')['MonthlyCondensateBBL'].cumsum()
    grouped['CumGasMCF'] = grouped.groupby('OperatorName')['MonthlyGasMCF'].cumsum()

    return grouped


def make_operator_group_plot(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df is None or df.empty:
        fig.update_layout(title="No operator production data available.")
        return fig
    for operator, group_df in df.groupby('OperatorName'):
        fig.add_trace(
            go.Scatter(
                x=group_df['PROD_DATE'],
                y=group_df['AvgBOERateBBLD'],
                mode='lines+markers',
                name=str(operator),
            )
        )
    fig.update_layout(
        title="Operator Group Average Daily BOE",
        xaxis_title="Production Month",
        yaxis_title="Avg Daily BOE",
        hovermode="x unified",
    )
    return fig


def prepare_operator_group_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame({"Message": ["No operator group data available."]})
    table_df = df.copy()
    table_df = table_df.rename(columns={
        'PROD_DATE': 'Prod_Month',
    })
    table_df['Prod_Month'] = table_df['Prod_Month'].dt.strftime('%Y-%m')
    return table_df


def _default_first_prod_range() -> tuple[date, date]:
    bounds = get_first_prod_bounds()
    if bounds:
        return bounds
    today = date.today()
    return today - timedelta(days=3650), today


DEFAULT_FIRST_PROD_RANGE = _default_first_prod_range()
DEFAULT_OPERATOR_RANGE = (date.today() - timedelta(days=365 * 5), date.today())


def _ensure_session_defaults():
    if 'filtered_wells' not in st.session_state:
        st.session_state.filtered_wells = gpd.GeoDataFrame(columns=FINAL_WELL_COLUMNS, geometry=[], crs="EPSG:4326")
    if 'play_layer_selection' not in st.session_state:
        st.session_state.play_layer_selection = []
    if 'company_layer_selection' not in st.session_state:
        st.session_state.company_layer_selection = []
    if 'filtered_group_metrics' not in st.session_state:
        st.session_state.filtered_group_metrics = None
    if 'filtered_group_breakout' not in st.session_state:
        st.session_state.filtered_group_breakout = 'OperatorName'
    if 'type_curve_result' not in st.session_state:
        st.session_state.type_curve_result = None
    if 'type_curve_error' not in st.session_state:
        st.session_state.type_curve_error = None
    if 'operator_group_df' not in st.session_state:
        st.session_state.operator_group_df = None


def _build_well_selector_options(df: pd.DataFrame) -> dict[str, str]:
    if df is None or df.empty:
        return {"No wells match filters": ""}
    options: dict[str, str] = {"Select a well from filtered list...": ""}
    for _, row in df.iterrows():
        well_name = row.get('WellName') or '[Missing]'
        uwi_value = row.get('UWI') or '[Missing]'
        gsl_value = row.get('GSL_UWI_Std')
        if not gsl_value:
            continue
        label = f"{textwrap.shorten(str(well_name), width=30, placeholder='...')} - UWI: {textwrap.shorten(str(uwi_value), width=15, placeholder='...')}"
        options[label] = gsl_value
    if len(options) == 1:
        return {"No wells with valid IDs in filter": ""}
    return options


def main() -> None:
    st.set_page_config(page_title=STREAMLIT_PAGE_TITLE, layout="wide")
    st.title("Interactive Well and Acreage Map Application")
    st.markdown("This Streamlit interface mirrors the legacy Shiny workflows using Python.")

    _ensure_session_defaults()

    sidebar = st.sidebar
    sidebar.header("Well Selection Criteria")

    operator_lookup = get_operator_display_lookup()
    operator_codes_sorted = sorted(
        operator_lookup.keys(),
        key=lambda code: operator_lookup[code].upper() if isinstance(operator_lookup[code], str) else str(operator_lookup[code]),
    )
    operators = sidebar.multiselect(
        "Operator",
        operator_codes_sorted,
        format_func=lambda code: operator_lookup.get(code, code)
    )

    formation_lookup = get_formation_lookup()
    formation_ids_sorted = sorted(
        formation_lookup.keys(),
        key=lambda fid: formation_lookup[fid].upper() if isinstance(formation_lookup[fid], str) else str(formation_lookup[fid]),
    )
    formations = sidebar.multiselect(
        "Formation",
        formation_ids_sorted,
        format_func=lambda fid: formation_lookup.get(fid, fid)
    )

    field_options = get_field_options()
    fields = sidebar.multiselect("Field", field_options)

    province_options = get_province_options()
    provinces = sidebar.multiselect("Province/State", province_options)
    date_start, date_end = sidebar.date_input(
        "Filter by First Production Date",
        value=DEFAULT_FIRST_PROD_RANGE,
    )
    play_layers = sidebar.multiselect(
        "Play/Subplay Boundaries",
        options=initial_play_subplay_layer_names,
        default=st.session_state.play_layer_selection,
    )
    company_layers = sidebar.multiselect(
        "Company Acreage",
        options=initial_company_layer_names,
        default=st.session_state.company_layer_selection,
    )

    col_apply, col_reset = sidebar.columns(2)
    with col_apply:
        if st.button("Apply Filters", key="apply_filters"):
            try:
                filtered = fetch_wells_from_db(
                    operators,
                    formations,
                    fields,
                    provinces,
                    (pd.to_datetime(date_start), pd.to_datetime(date_end)) if date_start and date_end else None,
                )
            except Exception as exc:
                st.error(str(exc))
            else:
                st.session_state.filtered_wells = filtered
                st.session_state.play_layer_selection = play_layers
                st.session_state.company_layer_selection = company_layers
                st.session_state.filtered_group_metrics = None
                st.session_state.filtered_group_breakout = 'OperatorName'
                st.success(f"Filters applied. {len(filtered):,} wells selected.")
                if getattr(filtered, 'attrs', {}).get('truncated'):
                    st.warning(
                        f"Showing up to {MAX_WELL_RESULTS:,} wells. Narrow filters to load a smaller result set.")
    with col_reset:
        if st.button("Reset Filters", key="reset_filters"):
            st.session_state.filtered_wells = gpd.GeoDataFrame(columns=FINAL_WELL_COLUMNS, geometry=[], crs="EPSG:4326")
            st.session_state.play_layer_selection = []
            st.session_state.company_layer_selection = []
            st.session_state.filtered_group_metrics = None
            st.session_state.type_curve_result = None
            st.session_state.operator_group_df = None
            st.info("Filters reset.")

    if sidebar.button("Reconnect to Database", key="reconnect_db"):
        new_engine = connect_to_db()
        if new_engine is not None:
            st.success("Successfully reconnected to Oracle.")
        else:
            st.error("Failed to reconnect to Oracle.")

    filtered_wells = st.session_state.filtered_wells
    confidential_count = 0
    if filtered_wells is not None and not filtered_wells.empty and 'ConfidentialType' in filtered_wells.columns:
        confidential_count = (filtered_wells['ConfidentialType'].astype(str).str.upper() == 'CONFIDENTIAL').sum()
    sidebar.metric("Wells Displayed", f"{len(filtered_wells):,}")
    sidebar.metric("Confidential Wells", f"{confidential_count:,}")

    deck, legend_entries = build_pydeck_map(
        filtered_wells,
        st.session_state.play_layer_selection,
        st.session_state.company_layer_selection,
    )

    tab_map, tab_prod, tab_filtered, tab_type_curve, tab_operator = st.tabs([
        "Well Map",
        "Production Analysis",
        "Filtered Group Cumulative",
        "Type Curve Analysis",
        "Operator Group",
    ])

    with tab_map:
        if deck is not None:
            st.pydeck_chart(deck)
        else:
            st.info("Apply filters to render the map.")
        if legend_entries:
            st.markdown("### Map Legend")
            legend_cols = st.columns(min(3, len(legend_entries)))
            for idx, (label, color) in enumerate(legend_entries):
                col = legend_cols[idx % len(legend_cols)]
                col.markdown(f"<div style='display:flex; align-items:center; gap:8px;'>"
                             f"<span style='display:inline-block; width:14px; height:14px; background:{color}; border:1px solid #333;'></span>"
                             f"{label}</div>", unsafe_allow_html=True)

    with tab_prod:
        st.subheader("Single Well Analysis")
        product_type_choices = {
            "Oil": "OIL",
            "Condensate": "CND",
            "Gas": "GAS",
            "BOE (Oil+Cnd+Gas)": "BOE",
        }
        selected_products_labels = st.multiselect(
            "Product Types",
            options=list(product_type_choices.keys()),
            default=list(product_type_choices.keys()),
        )
        selected_products = [product_type_choices[label] for label in selected_products_labels]

        well_options = _build_well_selector_options(filtered_wells)
        selected_well_label = st.selectbox("Select Well", list(well_options.keys()))
        selected_well = well_options.get(selected_well_label, "")

        prod_data = pd.DataFrame()
        processed_prod = pd.DataFrame()
        if selected_well:
            try:
                prod_data = fetch_single_well_production(selected_well)
                min_prod_date = prod_data['prod_date'].min() if not prod_data.empty else None
                max_prod_date = prod_data['prod_date'].max() if not prod_data.empty else None
                if min_prod_date is not None and max_prod_date is not None:
                    start_dt, end_dt = st.slider(
                        "Production Date Range",
                        min_value=min_prod_date.to_pydatetime(),
                        max_value=max_prod_date.to_pydatetime(),
                        value=(min_prod_date.to_pydatetime(), max_prod_date.to_pydatetime()),
                        format="YYYY-MM",
                    )
                else:
                    start_dt = end_dt = None
                processed_prod = process_single_well_production(
                    prod_data,
                    selected_products,
                    (pd.to_datetime(start_dt), pd.to_datetime(end_dt)) if start_dt and end_dt else None,
                )
            except Exception as exc:
                st.error(f"Failed to fetch production data: {exc}")

        well_label = selected_well_label if selected_well else "No well selected"
        _render_plotly_chart(
            make_single_well_rate_chart(processed_prod, selected_products, well_label)
        )
        st.subheader("Production Data Table")
        _render_dataframe(make_single_well_table(processed_prod))
        if not processed_prod.empty:
            st.download_button(
                "Download Production CSV",
                data=single_well_download_csv(processed_prod),
                file_name=f"production_{selected_well}.csv",
                mime="text/csv",
            )

    with tab_filtered:
        st.subheader("Filtered Group Cumulative")
        breakout_choices = {
            "Operator": "OperatorName",
            "Formation": "Formation",
            "Field": "FieldName",
            "Province/State": "ProvinceState",
            "First Prod Year": "FirstProdYear",
        }
        breakout_label = st.selectbox("Group By", list(breakout_choices.keys()), index=0)
        group_product_labels = st.multiselect(
            "Product Types",
            options=list(product_type_choices.keys()),
            default=list(product_type_choices.keys()),
            key="filtered_group_products",
        )
        group_products = [product_type_choices[label] for label in group_product_labels]
        if st.button("Calculate Rates", key="calculate_filtered_rates"):
            try:
                metrics = compute_filtered_group_metrics(
                    filtered_wells,
                    group_products,
                    breakout_choices[breakout_label],
                )
                st.session_state.filtered_group_metrics = metrics
                st.session_state.filtered_group_breakout = breakout_choices[breakout_label]
                st.success("Filtered group metrics calculated.")
            except Exception as exc:
                st.error(str(exc))
        metrics = st.session_state.filtered_group_metrics
        if metrics:
            norm_fig, cum_fig, cal_fig = make_filtered_group_plots(metrics, st.session_state.filtered_group_breakout)
            _render_plotly_chart(norm_fig)
            _render_plotly_chart(cum_fig)
            _render_plotly_chart(cal_fig)
            st.subheader("Filtered Group Summary")
            table_df = prepare_filtered_group_table(metrics, st.session_state.filtered_group_breakout)
            _render_dataframe(table_df)
            st.download_button(
                "Download Group Summary",
                data=table_df.to_csv(index=False),
                file_name="filtered_group_summary.csv",
                mime="text/csv",
            )
        else:
            st.info("Click 'Calculate Rates' to generate group metrics.")

    with tab_type_curve:
        st.subheader("Type Curve Analysis (Arps)")
        type_products_labels = st.multiselect(
            "Product Types",
            options=list(product_type_choices.keys()),
            default=list(product_type_choices.keys()),
            key="type_curve_products",
        )
        type_products = [product_type_choices[label] for label in type_products_labels]
        arps_product_choice = st.selectbox("Arps Product", ["Oil", "Gas"], index=0)
        model_choice = st.selectbox("Arps Model", ["hyperbolic", "exponential", "harmonic"], index=0)
        if st.button("Generate Type Curve", key="generate_type_curve"):
            try:
                result = compute_type_curve_analysis(
                    filtered_wells,
                    type_products,
                    model_choice,
                    arps_product_choice,
                )
                st.session_state.type_curve_result = result
                st.session_state.type_curve_error = None
                st.success("Type curve generated.")
            except Exception as exc:
                st.session_state.type_curve_result = None
                st.session_state.type_curve_error = str(exc)
                st.error(str(exc))
        result = st.session_state.type_curve_result
        if result:
            _render_plotly_chart(make_type_curve_plot(result))
            st.text(format_type_curve_summary(result))
            st.subheader("Aggregated Data Used for Type Curve")
            table_df = prepare_type_curve_table(result)
            _render_dataframe(table_df)
            st.download_button(
                "Download Type Curve Table",
                data=table_df.to_csv(index=False),
                file_name="type_curve_data.csv",
                mime="text/csv",
            )
        elif st.session_state.type_curve_error:
            st.info("Adjust filters and try again.")

    with tab_operator:
        st.subheader("Operator Group Cumulative")
        operator_lookup_tab = get_operator_display_lookup()
        operator_codes_tab = sorted(operator_lookup_tab.keys(), key=lambda code: operator_lookup_tab[code])
        selected_ops = st.multiselect(
            "Select Operators",
            operator_codes_tab,
            format_func=lambda code: operator_lookup_tab.get(code, code),
            key="operator_group_select"
        )
        operator_product_labels = st.multiselect(
            "Product Types",
            options=list(product_type_choices.keys()),
            default=list(product_type_choices.keys()),
            key="operator_group_products",
        )
        operator_products = [product_type_choices[label] for label in operator_product_labels]
        op_date_start, op_date_end = st.date_input(
            "Production Date Range",
            value=DEFAULT_OPERATOR_RANGE,
            key="operator_group_dates",
        )
        if st.button("Update Operator Plot", key="update_operator_plot"):
            try:
                operator_df = compute_operator_group_data(
                    selected_ops,
                    operator_products,
                    (pd.to_datetime(op_date_start), pd.to_datetime(op_date_end)) if op_date_start and op_date_end else None,
                )
                st.session_state.operator_group_df = operator_df
                st.success("Operator group data fetched.")
            except Exception as exc:
                st.session_state.operator_group_df = None
                st.error(str(exc))
        operator_df = st.session_state.operator_group_df
        _render_plotly_chart(make_operator_group_plot(operator_df))
        st.subheader("Operator Group Summary")
        table_df = prepare_operator_group_table(operator_df) if operator_df is not None else pd.DataFrame({"Message": ["No data."]})
        _render_dataframe(table_df)
        if operator_df is not None and not operator_df.empty:
            st.download_button(
                "Download Operator Summary",
                data=table_df.to_csv(index=False),
                file_name="operator_group_summary.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
