# --- 0. Load Necessary Packages ---
import os
import re
import warnings
from datetime import datetime, date
import pickle

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sqlalchemy import create_engine, text
import oracledb
from shiny.types import FileInfo

from shiny import App, ui, render, reactive, session
import htmltools

# Note: Plotting and mapping libraries are imported here but will be used in later sections.
import plotly.express as px
import plotly.graph_objects as go
from shinywidgets import output_widget, render_widget
from ipyleaflet import Map, Marker, GeoData, LayersControl, ScaleControl, FullScreenControl, basemaps, MarkerCluster, Polyline, Polygon

# --- Database Connection Details ---
DB_USER = "WOODMAC"
DB_PASSWORD = "c0pp3r"
TNS_ALIAS = "GDC_LINK.geologic.com"

# SQLAlchemy connection string format for Oracle with the python-oracledb driver.
# The DSN (Data Source Name) is the TNS alias.
CONNECTION_STRING = f"oracle+oracledb://{DB_USER}:{DB_PASSWORD}@{TNS_ALIAS}"

# Global engine object to hold the database connection pool.
engine = None

# --- Function to Establish Database Connection ---
def connect_to_db():
    """
    Establishes and returns a SQLAlchemy engine object.
    This replaces the R function of the same name.
    """
    global engine
    # If engine exists, test the connection to see if it's still valid.
    if engine is not None:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1 FROM DUAL"))
            print("Database connection is still valid.")
            return engine
        except Exception as e:
            print(f"Connection lost, attempting to reconnect. Error: {e}")
            engine = None

    try:
        print(f"Attempting to connect to Oracle: {TNS_ALIAS} as user: {DB_USER}")
        
        # Note: The oracledb.init_oracle_client() might be needed if the Oracle Instant Client
        # is not in your system's PATH or LD_LIBRARY_PATH. For example:
        # oracledb.init_oracle_client(lib_dir="C:/Users/I37643/OneDrive - Wood Mackenzie Limited/Documents/InstantClient_64bit/instantclient_23_7")
        
        engine = create_engine(
            CONNECTION_STRING,
            connect_args={"timeout": 10},
            echo=False  # Set to True for debugging SQL queries
        )
        # Test the connection to confirm it was established.
        with engine.connect() as conn:
            print("SUCCESS: Database connection established.")
        return engine
    except Exception as e:
        print(f"ERROR during create_engine or connect call: Failed to connect to Oracle.")
        print(f"Detailed Python error: {e}")
        engine = None
        return None

# Initial connection attempt when the app starts.
engine = connect_to_db()

# Note on app shutdown: SQLAlchemy's connection pooling is designed to manage
# connections automatically. Unlike the R Shiny 'onStop', it's not typical
# in Python web apps to manually close the engine on shutdown. The pool handles
# stale connections and recycling.

# --- 1. Define File Paths and Constants ---
# UPDATE THIS PATH TO YOUR LOCAL DIRECTORY
BASE_PATH = "C:/Users/I37643/OneDrive - Wood Mackenzie Limited/Documents/WoodMac/APP" 
# Using .pkl (pickle) is a common Python format for saving data structures like DataFrames
PROCESSED_DATA_FILE = os.path.join(BASE_PATH, "processed_app_data.pkl")

WOODMACK_COVERAGE_FILE_XLSX = os.path.join(BASE_PATH, "Woodmack.Coverage.2024.xlsx")
PLAY_SUBPLAY_SHAPEFILE_DIR = os.path.join(BASE_PATH, "SubplayShapefile")
COMPANY_SHAPEFILES_DIR = os.path.join(BASE_PATH, "Shapefile")

# Conversion factors
E3M3_TO_MCF = 35.3147
M3_TO_BBL = 6.28981
AVG_DAYS_PER_MONTH = 30.4375
MCF_PER_BOE = 6  # Standard conversion for BOE calculations

# Custom Color Palette
CUSTOM_PALETTE = [
  "#53143F", "#F94355", "#058F96", "#F57D01", "#205B2E", "#A8011E",
  "#5A63E3", "#FFD31A", "#4E207F", "#CC9900", "#A6A6A6", "#9ACEE2",
  "#A9899E", "#FFA3AA", "#92C5C9", "#F9BE96", "#92AC96", "#D6890B",
  "#ABAFF0", "#FFE89C", "#A68EBF", "#D7C68E", "#D1D1D1", "#CBE5EF"
]

# --- 2. Helper Functions ---

def standardize_uwi(uwi_series: pd.Series) -> pd.Series:
    """ Replaces R 'standardize_uwi' function. """
    if not isinstance(uwi_series, pd.Series):
        uwi_series = pd.Series(uwi_series)
    return uwi_series.astype(str).str.replace(r'[^A-Za-z0-9]', '', regex=True).str.upper()

def safe_read_excel(file_path: str, sheet_name: str, file_description: str = None) -> pd.DataFrame:
    """ Replaces R 'safe_read_excel' function. """
    if file_description is None:
        file_description = file_path
    print(f"Attempting to load Excel sheet: {sheet_name} from {file_description}")
    if not os.path.exists(file_path):
        warnings.warn(f"Excel file not found: {file_path} for {file_description}")
        return pd.DataFrame()
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"Successfully loaded Excel sheet: {sheet_name} - Rows: {len(df)}, Cols: {len(df.columns)}")
        return df
    except Exception as e:
        warnings.warn(f"Error loading Excel sheet {sheet_name} from {file_path}: {e}")
        return pd.DataFrame()

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
    
    # Define final column names and create an empty GeoDataFrame structure
    final_column_names = [
        "UWI", "GSL_UWI", "SurfaceLatitude", "SurfaceLongitude",
        "BH_Latitude", "BH_Longitude", "LateralLength",
        "AbandonmentDate", "WellName", "CurrentStatus", "OperatorCode", "StratUnitID",
        "SpudDate", "FirstProdDate", "FinalTD", "ProvinceState", "Country",
        "UWI_Std", "GSL_UWI_Std", "OperatorName", "Formation", "FieldName",
        "ConfidentialType"
    ]
    empty_gdf = gpd.GeoDataFrame(columns=final_column_names, geometry=[], crs=f"EPSG:4326")

    app_data = {
        'wells_gdf': empty_gdf,
        'play_subplay_layers_list': [],
        'company_layers_list': []
    }
    load_from_db = True

    # Try loading from pre-processed file first
    if os.path.exists(PROCESSED_DATA_FILE):
        print(f"Attempting to load pre-processed data from: {PROCESSED_DATA_FILE}")
        try:
            with open(PROCESSED_DATA_FILE, 'rb') as f:
                loaded_data = pickle.load(f)

            wells_ok = ('wells_gdf' in loaded_data and 
                        isinstance(loaded_data['wells_gdf'], gpd.GeoDataFrame) and 
                        not loaded_data['wells_gdf'].empty)
            play_layers_ok = 'play_subplay_layers_list' in loaded_data
            company_layers_ok = 'company_layers_list' in loaded_data
            
            if wells_ok and play_layers_ok and company_layers_ok:
                app_data = loaded_data
                print("SUCCESS: Pre-processed data loaded and validated from .pkl file.")
                load_from_db = False
            else:
                print("WARNING: .pkl file loaded, but data is invalid/incomplete. Will reload from DB.")
        
        except Exception as e:
            print(f"ERROR loading .pkl file: {e}. Will reload from DB.")

    if load_from_db:
        print("INITIATING DATA LOAD FROM DATABASE AND/OR SHAPEFILES...")
        global engine
        if engine is None:
            engine = connect_to_db()
        
        if engine is None:
            raise Exception("FATAL: Database connection failed. Cannot load primary data.")

        sql_well_master = """
        SELECT W.UWI, W.GSL_UWI, W.SURFACE_LATITUDE, W.SURFACE_LONGITUDE,
               W.BOTTOM_HOLE_LATITUDE, W.BOTTOM_HOLE_LONGITUDE, W.GSL_FULL_LATERAL_LENGTH,
               W.ABANDONMENT_DATE, W.WELL_NAME, W.CURRENT_STATUS, W.OPERATOR AS OPERATOR_CODE, W.CONFIDENTIAL_TYPE,
               P.STRAT_UNIT_ID, W.SPUD_DATE, PFS.FIRST_PROD_DATE, W.FINAL_TD, W.PROVINCE_STATE, W.COUNTRY, FL.FIELD_NAME
        FROM WELL W
        LEFT JOIN PDEN P ON W.GSL_UWI = P.GSL_UWI
        LEFT JOIN FIELD FL ON W.ASSIGNED_FIELD = FL.FIELD_ID
        LEFT JOIN PDEN_FIRST_SUM PFS ON W.GSL_UWI = PFS.GSL_UWI
        WHERE W.SURFACE_LATITUDE IS NOT NULL AND W.SURFACE_LONGITUDE IS NOT NULL
          AND (W.ABANDONMENT_DATE IS NULL OR W.ABANDONMENT_DATE > SYSDATE - (365*20))
        """
        print("Fetching well master data from Oracle...")
        wells_master_df = pd.read_sql(sql_well_master, engine)
        print(f"DB Load: Successfully loaded {len(wells_master_df)} base well rows from DB.")

        # --- Data Processing (Pandas equivalent of data.table steps) ---
        if not wells_master_df.empty:
            wells_master_df['UWI_Std'] = standardize_uwi(wells_master_df['uwi'])
            wells_master_df['GSL_UWI_Std'] = standardize_uwi(wells_master_df['gsl_uwi'])

            # Fetch Strat names
            unique_strat_ids = wells_master_df['strat_unit_id'].dropna().unique()
            if len(unique_strat_ids) > 0:
                strat_names_df = pd.DataFrame()
                # Batch fetch in chunks of 500
                for i in range(0, len(unique_strat_ids), 500):
                    batch_ids = unique_strat_ids[i:i + 500]
                    sql_strat_names = f"SELECT STRAT_UNIT_ID, SHORT_NAME FROM STRAT_UNIT WHERE STRAT_UNIT_ID IN ({','.join([str(i) for i in batch_ids])})"
                    strat_names_batch_df = pd.read_sql(sql_strat_names, engine)
                    strat_names_df = pd.concat([strat_names_df, strat_names_batch_df], ignore_index=True)
                
                strat_names_df = strat_names_df.rename(columns={'short_name': 'Formation'})
                wells_master_df = pd.merge(wells_master_df, strat_names_df, on='strat_unit_id', how='left')
            else:
                wells_master_df['Formation'] = None

            # Load and merge Operator names
            operator_codes_df = safe_read_excel(WOODMACK_COVERAGE_FILE_XLSX, sheet_name="Operator")
            if not operator_codes_df.empty:
                operator_codes_df = clean_df_colnames(operator_codes_df)
                operator_codes_df = operator_codes_df.rename(columns={'OPERATOR': 'WoodmackJoinOperatorCode', 'GSL_PARENT_BA_NAME': 'OperatorNameDisplay'})
                wells_master_df['operator_code'] = wells_master_df['operator_code'].astype(str)
                operator_codes_df['WoodmackJoinOperatorCode'] = operator_codes_df['WoodmackJoinOperatorCode'].astype(str)
                wells_master_df = pd.merge(wells_master_df, operator_codes_df[['WoodmackJoinOperatorCode', 'OperatorNameDisplay']], 
                                           left_on='operator_code', right_on='WoodmackJoinOperatorCode', how='left')

            # Rename columns to match target schema
            rename_map = {
                'uwi': 'UWI', 'gsl_uwi': 'GSL_UWI', 'surface_latitude': 'SurfaceLatitude',
                'surface_longitude': 'SurfaceLongitude', 'bottom_hole_latitude': 'BH_Latitude',
                'bottom_hole_longitude': 'BH_Longitude', 'gsl_full_lateral_length': 'LateralLength',
                'abandonment_date': 'AbandonmentDate', 'well_name': 'WellName', 'current_status': 'CurrentStatus',
                'operator_code': 'OperatorCode', 'strat_unit_id': 'StratUnitID', 'spud_date': 'SpudDate',
                'first_prod_date': 'FirstProdDate', 'final_td': 'FinalTD', 'province_state': 'ProvinceState',
                'country': 'Country', 'field_name': 'FieldName', 'confidential_type': 'ConfidentialType',
                'OperatorNameDisplay': 'OperatorName'
            }
            wells_master_df = wells_master_df.rename(columns=rename_map)

            # Ensure all final columns exist and have correct types
            for col in final_column_names:
                if col not in wells_master_df.columns:
                    wells_master_df[col] = None
            
            date_cols = ['AbandonmentDate', 'SpudDate', 'FirstProdDate']
            for col in date_cols:
                wells_master_df[col] = pd.to_datetime(wells_master_df[col], errors='coerce')
            
            numeric_cols = ['SurfaceLatitude', 'SurfaceLongitude', 'BH_Latitude', 'BH_Longitude', 'LateralLength', 'FinalTD']
            for col in numeric_cols:
                wells_master_df[col] = pd.to_numeric(wells_master_df[col], errors='coerce')

            # Create GeoDataFrame
            wells_for_gdf = wells_master_df.dropna(subset=['SurfaceLongitude', 'SurfaceLatitude'])
            app_data['wells_gdf'] = gpd.GeoDataFrame(
                wells_for_gdf[final_column_names],
                geometry=gpd.points_from_xy(wells_for_gdf.SurfaceLongitude, wells_for_gdf.SurfaceLatitude),
                crs="EPSG:4269" # Assuming NAD83 initially like the R code
            ).to_crs("EPSG:4326")

        # --- Load Shapefiles ---
        print("--- Loading Play/Subplay Acreage ---")
        play_layers = []
        if os.path.isdir(PLAY_SUBPLAY_SHAPEFILE_DIR):
            for filename in os.listdir(PLAY_SUBPLAY_SHAPEFILE_DIR):
                if filename.lower().endswith('.shp'):
                    shp_path = os.path.join(PLAY_SUBPLAY_SHAPEFILE_DIR, filename)
                    layer_name = os.path.splitext(filename)[0]
                    gdf = load_process_spatial_layer(shp_path, layer_name, simplify=True, tolerance=100, make_valid_geom=True)
                    if gdf is not None:
                        play_layers.append({'name': layer_name, 'data': gdf})
        app_data['play_subplay_layers_list'] = play_layers

        print("--- Loading DISSOLVED Company Acreage ---")
        company_layers = []
        if os.path.isdir(COMPANY_SHAPEFILES_DIR):
            for filename in os.listdir(COMPANY_SHAPEFILES_DIR):
                if filename.lower().endswith('.shp'):
                    shp_path = os.path.join(COMPANY_SHAPEFILES_DIR, filename)
                    layer_name = os.path.splitext(filename)[0]
                    gdf = load_process_spatial_layer(shp_path, layer_name, simplify=False, make_valid_geom=False)
                    if gdf is not None:
                        company_layers.append({'name': layer_name, 'data': gdf})
        app_data['company_layers_list'] = company_layers

        # Save processed data to pickle file
        try:
            print(f"Saving processed data to: {PROCESSED_DATA_FILE}")
            with open(PROCESSED_DATA_FILE, 'wb') as f:
                pickle.dump(app_data, f)
            print("Processed data saved to .pkl file.")
        except Exception as e:
            print(f"Error saving .pkl file: {e}")

    return app_data

# --- Load data into global variables for the app ---
APP_DATA = load_and_process_all_data()
wells_gdf_global = APP_DATA['wells_gdf']
play_subplay_layers_list_global = APP_DATA['play_subplay_layers_list']
company_layers_list_global = APP_DATA['company_layers_list']

# Prepare initial choices for UI dropdowns
initial_play_subplay_layer_names = sorted([layer['name'] for layer in play_subplay_layers_list_global])
initial_company_layer_names = sorted([layer['name'] for layer in company_layers_list_global])


# Final checks before starting server
print("--- FINAL CHECK OF wells_gdf_global BEFORE SERVER ---")
if wells_gdf_global is not None and not wells_gdf_global.empty:
    print(f"  nrow(wells_gdf_global): {len(wells_gdf_global)}")
    for col in ['ConfidentialType', 'BH_Latitude', 'BH_Longitude', 'LateralLength', 'FirstProdDate']:
        if col in wells_gdf_global.columns:
            print(f"  '{col}' column IS present. Sample: {wells_gdf_global[col].dropna().head(3).to_list()}")
        else:
            print(f"  '{col}' column IS NOT present.")
else:
    print("  wells_gdf_global is None or empty before server starts.")

# --- 4. Define User Interface (UI) ---
app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.style(
            """
            .shiny-notification { 
                position:fixed; 
                top: calc(5%); 
                left: calc(50% - 150px); 
                width: 300px; 
                z-index: 2000 !important; 
            }
            """
        )
    ),
    ui.panel_title("Interactive Well and Acreage Map Application (Python Conversion)"),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.h4("Well Selection Criteria"),
            # Note: shiny.ui.input_selectize is used as a replacement for shinyWidgets::pickerInput
            ui.input_selectize("operator_filter", "Operator:", choices=[], multiple=True),
            ui.input_selectize("formation_filter", "Formation:", choices=[], multiple=True),
            ui.input_selectize("field_filter", "Field:", choices=[], multiple=True),
            ui.input_selectize("province_filter", "Province/State:", choices=[], multiple=True),
            ui.input_date_range("well_date_filter", "Filter by First Production Date:",
                                start=date.today() - pd.Timedelta(days=365*10), end=date.today()),
            ui.input_action_button("update_map", "Apply Filters & Update Map", class_="btn-primary btn-block"),
            ui.input_action_button("reset_filters", "Reset All Filters", class_="btn-block"),
            ui.input_action_button("reconnect_db_button", "Reconnect to Database", class_="btn-warning btn-block", style="margin-top: 10px;"),
            ui.hr(),
            ui.h5(ui.strong(ui.output_ui("well_count_display"))),
            ui.hr(),
            ui.h5("Map Layers:"),
            ui.input_selectize("play_subplay_filter", "Play/Subplay Boundaries:", 
                               choices=initial_play_subplay_layer_names, multiple=True),
            ui.input_selectize("company_acreage_filter", "Company Acreage:", 
                               choices=initial_company_layer_names, multiple=True),
            width=3
        ),
        ui.panel_main(
            ui.navset_tab(
                ui.nav("Well Map", output_widget("well_map", height="85vh")),
                ui.nav("Production Analysis",
                    ui.row(
                        ui.column(12,
                            ui.input_selectize("product_type_filter_analysis", "Filter Analyses by Product Type:",
                                              choices={"Oil": "OIL", "Condensate": "CND", "Gas": "GAS", "BOE (Oil+Cnd+Gas)": "BOE"},
                                              selected=["OIL", "CND", "GAS", "BOE"],
                                              multiple=True)
                        )
                    ),
                    ui.hr(),
                    ui.navset_tab(
                        ui.nav("Single Well Analysis",
                            ui.h4("Daily Production Rate"),
                            ui.input_select("selected_well_for_prod", "Select Well for Production:", choices=["Apply filters and click a well..."]),
                            ui.output_ui("production_date_slider_ui"),
                            ui.output_plot("production_plot", height="45vh"),
                            ui.hr(),
                            ui.h5("Production Data Table"),
                            ui.download_button("download_prod_data", "Download Table as CSV"),
                            ui.output_data_frame("production_table")
                        ),
                        ui.nav("Filtered Group Cumulative",
                            ui.h4("Time Normalized Production by Group (Map-Filtered Wells)"),
                            ui.p("This analysis uses wells currently displayed on the map. Production is normalized by reported lateral length (if available)."),
                            ui.row(
                                ui.column(6, ui.input_select("filtered_group_breakout_by", "Normalize & Group By:",
                                                             choices={"Operator": "OperatorName", "Formation": "Formation", "Field": "FieldName",
                                                                      "Province/State": "ProvinceState", "First Prod Year": "FirstProdYear"},
                                                             selected="OperatorName")),
                                ui.column(6, ui.input_action_button("calculate_filtered_cumulative", "Calculate Rates", class_="btn-info btn-block", style="margin-top: 25px;"))
                            ),
                            ui.hr(),
                            ui.output_ui("filtered_group_plot_title_normalized"),
                            ui.output_plot("filtered_group_cumulative_plot_normalized", height="45vh"),
                            ui.hr(),
                            ui.output_ui("filtered_group_plot_title_cumulative_boe"),
                            ui.output_plot("filtered_group_cumulative_plot_cumulative_boe", height="45vh"),
                            ui.hr(),
                            ui.output_ui("filtered_group_plot_title_calendar_rate"),
                            ui.output_plot("filtered_group_calendar_rate_plot", height="45vh"),
                            ui.hr(),
                            ui.h5("Filtered Group Production Data Summary"),
                            ui.download_button("download_filtered_group_prod_data", "Download Summary as CSV"),
                            ui.output_data_frame("filtered_group_production_table")
                        ),
                        ui.nav("Type Curve Analysis (Arps)",
                            ui.h4("Arps Decline Curve (Peak Normalized) for Map-Filtered Wells"),
                            ui.row(
                                ui.column(6, ui.input_select("arps_product_type", "Select Product for Arps:",
                                                             choices={"Oil/Condensate": "Oil", "Gas": "Gas"})),
                                ui.column(6, ui.input_select("arps_model_type", "Select Arps Model:",
                                                             choices={"Hyperbolic": "hyperbolic", "Exponential": "exponential", "Harmonic": "harmonic"}))
                            ),
                            ui.input_action_button("generate_type_curve", "Generate Type Curve", class_="btn-info btn-block"),
                            ui.hr(),
                            ui.output_plot("arps_type_curve_plot", height="50vh"),
                            ui.h5("Fitted Arps Parameters, EUR & Decline Summary:"),
                            ui.output_text_verbatim("arps_parameters_output"),
                            ui.hr(),
                            ui.h5("Aggregated Data Used for Type Curve"),
                            ui.output_data_frame("arps_data_table")
                        ),
                        ui.nav("Operator Group Cumulative",
                            ui.h4("Operator Group Average Daily Production Rate"),
                            ui.p(ui.strong("Note:"), " This tab shows gross production for selected operators over a specific date range, independent of map filters."),
                            ui.input_selectize("group_operator_filter", "Select Operator(s) to Group:", choices=[], multiple=True),
                            ui.input_date_range("group_prod_date_range", "Select Date Range:",
                                                start=date.today() - pd.Timedelta(days=365*5), end=date.today()),
                            ui.input_action_button("update_group_plot", "Update Operator Group Plot", class_="btn-info"),
                            ui.hr(),
                            ui.output_plot("grouped_cumulative_plot", height="50vh"),
                            ui.h5("Operator Group Production Data Summary"),
                            ui.download_button("download_group_prod_data", "Download Operator Group Summary as CSV"),
                            ui.output_data_frame("grouped_production_table")
                        )
                    )
                )
            )
        )
    )
)
def server(input, output, session):
    
    # Reactive values to hold state
    wells_to_display = reactive.Value(gpd.GeoDataFrame(columns=wells_gdf_global.columns, crs="EPSG:4326"))
    has_map_been_updated_once = reactive.Value(False)
    current_selected_gsl_uwi_std = reactive.Value(None)
    
    # Initialize the map object. This will be updated reactively.
    map_object = Map(center=(55, -106), zoom=4, basemap=basemaps.CartoDB.Positron, scroll_wheel_zoom=True)
    map_object.add_control(LayersControl())
    map_object.add_control(ScaleControl(position='bottomleft'))
    map_object.add_control(FullScreenControl())
    
    # Store map layers to manage them easily
    map_layers = {
        "wells": MarkerCluster(markers=[], name="Wells"),
        "well_sticks": [], # This will be a list of Polyline layers
        "play_subplay": [], # This will be a list of GeoData layers
        "company": [] # This will be a list of GeoData layers
    }
    
    @render_widget
    def well_map():
        return map_object
    
    # Initial population of filters
    @reactive.Effect
    @reactive.event(wells_gdf_global) # Trigger once when data is loaded
    def _():
        if wells_gdf_global is None or wells_gdf_global.empty:
            return
        
        print("SERVER: Initializing filter choices.")
        op_choices = prepare_filter_choices(wells_gdf_global["OperatorName"], "OperatorName")
        form_choices = prepare_filter_choices(wells_gdf_global["Formation"], "Formation")
        fld_choices = prepare_filter_choices(wells_gdf_global["FieldName"], "FieldName")
        prov_choices = prepare_filter_choices(wells_gdf_global["ProvinceState"], "ProvinceState")
        
        ui.update_selectize("operator_filter", choices=op_choices, selected=[])
        ui.update_selectize("group_operator_filter", choices=op_choices, selected=[])
        ui.update_selectize("formation_filter", choices=form_choices, selected=[])
        ui.update_selectize("field_filter", choices=fld_choices, selected=[])
        ui.update_selectize("province_filter", choices=prov_choices, selected=[])

        if "FirstProdDate" in wells_gdf_global.columns:
            min_date = wells_gdf_global["FirstProdDate"].min()
            max_date = wells_gdf_global["FirstProdDate"].max()
            if pd.notna(min_date) and pd.notna(max_date):
                ui.update_date_range("well_date_filter", min=min_date, max=max_date, 
                                     start=max_date - pd.Timedelta(days=365*10), end=max_date)

    # --- Event Handlers for Buttons ---
    @reactive.Effect
    @reactive.event(input.reset_filters)
    def _():
        ui.notification_show("Resetting all filters...", type="message", duration=2)
        ui.update_selectize("operator_filter", selected=[])
        ui.update_selectize("formation_filter", selected=[])
        ui.update_selectize("field_filter", selected=[])
        ui.update_selectize("province_filter", selected=[])
        ui.update_selectize("play_subplay_filter", selected=[])
        ui.update_selectize("company_acreage_filter", selected=[])
        ui.update_selectize("group_operator_filter", selected=[])
        # Reset other inputs as needed
        wells_to_display.set(gpd.GeoDataFrame(columns=wells_gdf_global.columns, crs="EPSG:4326"))
        has_map_been_updated_once.set(False)

    @reactive.Effect
    @reactive.event(input.reconnect_db_button)
    def _():
        ui.notification_show("Attempting to reconnect...", type="message", duration=None, id="db_reconnect")
        global engine
        engine = None # Force re-creation
        new_engine = connect_to_db()
        ui.notification_remove("db_reconnect")
        if new_engine:
            ui.notification_show("Successfully reconnected to database.", type="message", duration=3)
        else:
            ui.notification_show("Failed to reconnect to database.", type="error", duration=5)

    @reactive.Effect
    @reactive.event(input.update_map)
    def _():
        ui.notification_show("Applying filters...", type="message", duration=None, id="map_update")
        
        df = wells_gdf_global.copy()
        
        # Apply filters
        if input.operator_filter():
            df = df[df['OperatorName'].isin(input.operator_filter())]
        if input.formation_filter():
            df = df[df['Formation'].isin(input.formation_filter())]
        if input.field_filter():
            df = df[df['FieldName'].isin(input.field_filter())]
        if input.province_filter():
            df = df[df['ProvinceState'].isin(input.province_filter())]
        
        start_date, end_date = input.well_date_filter()
        df = df[df['FirstProdDate'].between(pd.to_datetime(start_date), pd.to_datetime(end_date))]
        
        wells_to_display.set(df)
        has_map_been_updated_once.set(True)
        
        ui.notification_remove("map_update")
        ui.notification_show(f"Map updated with {len(df)} wells.", type="message", duration=3)
        
        # Update well selector for production analysis
        if not df.empty:
            choices = dict(zip(df['GSL_UWI_Std'], df['WellName'] + " - " + df['UWI']))
            ui.update_select("selected_well_for_prod", choices=choices)
        else:
            ui.update_select("selected_well_for_prod", choices=["No wells match filters"])

    # --- Reactive UI Outputs ---
    @render.ui
    def well_count_display():
        if not has_map_been_updated_once():
            return ui.HTML("Apply filters to see well counts.")
        
        displayed_wells = wells_to_display.get()
        total_count = len(displayed_wells)
        confidential_count = 0
        if "ConfidentialType" in displayed_wells.columns:
            confidential_count = (displayed_wells["ConfidentialType"].str.upper() == "CONFIDENTIAL").sum()
            
        return ui.HTML(f"Total Wells Displayed: {total_count:,}<br/>Confidential Wells: {confidential_count:,}")

    # --- Map Update Logic ---
    @reactive.Effect
    def _update_map_layers():
        # This effect will run whenever the filtered data or layer selections change
        df_map = wells_to_display()
        play_layers_selection = input.play_subplay_filter()
        company_layers_selection = input.company_acreage_filter()

        # Clear existing layers
        map_object.clear_layers()
        # Re-add basemaps
        map_object.add_layer(basemaps.CartoDB.Positron)
        map_object.add_layer(basemaps.OpenStreetMap.Mapnik)
        map_object.add_layer(basemaps.Esri.WorldImagery)


        # Add well markers and sticks
        if not df_map.empty:
            # Well Markers
            markers = []
            for idx, row in df_map.iterrows():
                popup_html = f"""
                <b>UWI:</b> {row['UWI']}<br>
                <b>Well Name:</b> {row['WellName']}<br>
                <b>Operator:</b> {row['OperatorName']}<br>
                <b>Formation:</b> {row['Formation']}
                """
                marker = Marker(location=(row['SurfaceLatitude'], row['SurfaceLongitude']), 
                                draggable=False,
                                popup=Popup(child=htmltools.HTML(popup_html)))
                markers.append(marker)
            marker_cluster = MarkerCluster(markers=markers, name="Wells")
            map_object.add_layer(marker_cluster)

            # Well Sticks
            wells_with_bh = df_map.dropna(subset=['BH_Latitude', 'BH_Longitude'])
            if not wells_with_bh.empty:
                for idx, row in wells_with_bh.iterrows():
                    stick = Polyline(
                        locations=[
                            (row['SurfaceLatitude'], row['SurfaceLongitude']),
                            (row['BH_Latitude'], row['BH_Longitude'])
                        ],
                        color="red",
                        weight=2,
                        fill=False
                    )
                    map_object.add_layer(stick)
        
        # Add polygon layers
        # Play/Subplay Layers
        for layer_info in play_subplay_layers_list_global:
            if layer_info['name'] in play_layers_selection:
                gdf = layer_info['data']
                if not gdf.empty:
                    geo_data = GeoData(geo_dataframe=gdf, name=layer_info['name'],
                                       style={'color': 'blue', 'weight': 1, 'fillOpacity': 0.1})
                    map_object.add_layer(geo_data)

        # Company Acreage Layers
        for layer_info in company_layers_list_global:
            if layer_info['name'] in company_layers_selection:
                gdf = layer_info['data']
                if not gdf.empty:
                    geo_data = GeoData(geo_dataframe=gdf, name=layer_info['name'],
                                       style={'color': 'green', 'weight': 1.5, 'fillOpacity': 0.2})
                    map_object.add_layer(geo_data)
    
    # --- Single Well Production Analysis ---

    @reactive.Calc
    def fetched_production_data():
        """
        Fetches production data from the database for the currently selected well.
        This is a reactive calculation that re-runs when its dependencies change.
        """
        req(input.selected_well_for_prod(), input.product_type_filter_analysis())
        selected_uwi = input.selected_well_for_prod()
        
        if not selected_uwi or "..." in selected_uwi:
            return pd.DataFrame()

        print(f"--- fetched_production_data: Fetching for GSL_UWI_Std: '{selected_uwi}' ---")
        if engine is None:
            ui.notification_show("Database connection not available.", type="error")
            return pd.DataFrame()

        sql_query = text(f"""
            SELECT GSL_UWI, YEAR, PRODUCT_TYPE, ACTIVITY_TYPE,
                   JAN_VOLUME, FEB_VOLUME, MAR_VOLUME, APR_VOLUME,
                   MAY_VOLUME, JUN_VOLUME, JUL_VOLUME, AUG_VOLUME,
                   SEP_VOLUME, OCT_VOLUME, NOV_VOLUME, DEC_VOLUME
            FROM PDEN_VOL_BY_MONTH 
            WHERE GSL_UWI = :uwi
            AND ACTIVITY_TYPE = 'PRODUCTION' 
            AND PRODUCT_TYPE IN ('OIL', 'CND', 'GAS')
        """)

        try:
            with engine.connect() as conn:
                well_prod_raw = pd.read_sql(sql_query, conn, params={"uwi": selected_uwi})
        except Exception as e:
            ui.notification_show(f"Error fetching production data: {e}", type="error")
            return pd.DataFrame()

        if well_prod_raw.empty:
            return pd.DataFrame()

        # Reshape data from wide to long format
        month_cols = {f"{m.upper()}_VOLUME": i+1 for i, m in enumerate(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])}
        prod_long = pd.melt(well_prod_raw,
                            id_vars=['gsl_uwi', 'year', 'product_type'],
                            value_vars=month_cols.keys(),
                            var_name='month_col', value_name='volume')
        
        prod_long['month'] = prod_long['month_col'].str.extract(r'(\w+)_').map(lambda x: month_cols[f"{x}_VOLUME"])
        prod_long['prod_date'] = pd.to_datetime(prod_long['year'].astype(str) + '-' + prod_long['month'].astype(str) + '-01')

        # Unit Conversions
        is_liquid = prod_long['product_type'].isin(['OIL', 'CND'])
        is_gas = prod_long['product_type'] == 'GAS'
        prod_long.loc[is_liquid, 'volume_converted'] = prod_long['volume'] * M3_TO_BBL
        prod_long.loc[is_gas, 'volume_converted'] = prod_long['volume'] * E3M3_TO_MCF
        
        prod_long = prod_long.dropna(subset=['volume_converted'])

        # Pivot to have one row per month with different products as columns
        prod_pivot = prod_long.pivot_table(index=['gsl_uwi', 'prod_date'],
                                           columns='product_type',
                                           values='volume_converted').reset_index()
        prod_pivot = prod_pivot.rename(columns={'OIL': 'MonthlyOilTrueBBL', 'CND': 'MonthlyCondensateBBL', 'GAS': 'MonthlyGasMCF'})
        
        return prod_pivot.fillna(0)

    @render.ui
    def production_date_slider_ui():
        prod_data = fetched_production_data()
        if prod_data.empty:
            return None
        
        min_date = prod_data['prod_date'].min().date()
        max_date = prod_data['prod_date'].max().date()
        
        return ui.input_slider("production_date_filter", "Filter Production Dates:", 
                               min=min_date, max=max_date, value=(min_date, max_date))

    @reactive.Calc
    def processed_production_for_plotting():
        prod_data = fetched_production_data()
        req(input.production_date_filter())
        
        if prod_data.empty:
            return pd.DataFrame()
            
        start_date, end_date = input.production_date_filter()
        
        df = prod_data[prod_data['prod_date'].between(pd.to_datetime(start_date), pd.to_datetime(end_date))].copy()
        df = df.sort_values('prod_date')

        # Calculate rates and cumulative values
        df['DaysInMonth'] = df['prod_date'].dt.days_in_month
        
        selected_prods = input.product_type_filter_analysis()
        
        df['MonthlyOilTrueBBL'] = df.get('MonthlyOilTrueBBL', 0)
        df['MonthlyCondensateBBL'] = df.get('MonthlyCondensateBBL', 0)
        df['MonthlyGasMCF'] = df.get('MonthlyGasMCF', 0)
        
        df['OilTrueRateBBLD'] = df['MonthlyOilTrueBBL'] / df['DaysInMonth']
        df['CondensateRateBBLD'] = df['MonthlyCondensateBBL'] / df['DaysInMonth']
        df['GasRateMCFD'] = df['MonthlyGasMCF'] / df['DaysInMonth']
        
        oil_for_boe = df['MonthlyOilTrueBBL'] if "OIL" in selected_prods or "BOE" in selected_prods else 0
        cnd_for_boe = df['MonthlyCondensateBBL'] if "CND" in selected_prods or "BOE" in selected_prods else 0
        gas_for_boe = df['MonthlyGasMCF'] if "GAS" in selected_prods or "BOE" in selected_prods else 0
        
        df['BOERateBBLD'] = (oil_for_boe + cnd_for_boe + (gas_for_boe / MCF_PER_BOE)) / df['DaysInMonth']

        df['CumOilTrueBBL'] = df['MonthlyOilTrueBBL'].cumsum()
        df['CumCondensateBBL'] = df['MonthlyCondensateBBL'].cumsum()
        df['CumGasMCF'] = df['MonthlyGasMCF'].cumsum()
        
        return df

    @render.plot
    def production_plot():
        plot_data = processed_production_for_plotting()
        
        if plot_data.empty:
            return p9.ggplot() + p9.labs(title="No production data for selected well/product(s).")

        # Melt data for plotting with plotnine/ggplot
        cols_to_plot = []
        if "OIL" in input.product_type_filter_analysis(): cols_to_plot.append("OilTrueRateBBLD")
        if "CND" in input.product_type_filter_analysis(): cols_to_plot.append("CondensateRateBBLD")
        if "GAS" in input.product_type_filter_analysis(): cols_to_plot.append("GasRateMCFD")
        if "BOE" in input.product_type_filter_analysis(): cols_to_plot.append("BOERateBBLD")

        if not cols_to_plot:
            return p9.ggplot() + p9.labs(title="Please select a product type for analysis.")

        plot_data_long = pd.melt(plot_data, id_vars=['prod_date'], value_vars=cols_to_plot,
                                 var_name='ProductRateType', value_name='Rate')
        
        plot_data_long = plot_data_long[plot_data_long['Rate'] > 0]
        
        if plot_data_long.empty:
            return p9.ggplot() + p9.labs(title="No positive production rates to display.")
        
        well_info = wells_gdf_global[wells_gdf_global['GSL_UWI_Std'] == input.selected_well_for_prod()]
        title = f"Daily Production Rate: {well_info['WellName'].iloc[0]}" if not well_info.empty else "Daily Production Rate"

        return (
            p9.ggplot(plot_data_long, p9.aes(x='prod_date', y='Rate', color='ProductRateType'))
            + p9.geom_line()
            + p9.geom_point()
            + p9.scale_y_continuous(labels=lambda l: [f'{int(v):,}' for v in l])
            + p9.labs(title=title, x="Date", y="Daily Rate", color="Product Type")
            + p9.theme_minimal()
            + p9.theme(axis_text_x=p9.element_text(angle=45, hjust=1))
        )

    @render.data_frame
    def production_table():
        df = processed_production_for_plotting()
        if df.empty:
            return pd.DataFrame({"Message": ["No production data to display."]})
            
        cols_to_show = {
            "prod_date": "Prod. Month",
            "MonthlyOilTrueBBL": "Oil (BBL/Month)",
            "MonthlyCondensateBBL": "Cond. (BBL/Month)",
            "MonthlyGasMCF": "Gas (MCF/Month)",
            "OilTrueRateBBLD": "Oil Rate (BBL/day)",
            "CondensateRateBBLD": "Cond. Rate (BBL/day)",
            "GasRateMCFD": "Gas Rate (MCF/day)",
            "BOERateBBLD": "BOE Rate (BBL/day)",
            "CumOilTrueBBL": "Cum. Oil (BBL)",
            "CumCondensateBBL": "Cum. Cond. (BBL)",
            "CumGasMCF": "Cum. Gas (MCF)",
        }
        
        df_display = df[list(cols_to_show.keys())].copy()
        df_display.rename(columns=cols_to_show, inplace=True)
        df_display['Prod. Month'] = df_display['Prod. Month'].dt.strftime('%Y-%m')
        
        return df_display.round(2)
        
    @session.download(filename="production_data.csv")
    def download_prod_data():
        df_to_download = processed_production_for_plotting().copy()
        if not df_to_download.empty:
            yield df_to_download.to_csv(index=False)
        else:
            yield "No data available to download."


 # --- Filtered Group Cumulative Logic ---

    @reactive.Calc
    @reactive.event(input.calculate_filtered_cumulative)
    def filtered_group_cumulative_data():
        ui.notification_show("Calculating normalized rates...", duration=None, id="filt_rate_msg")
        
        filtered_wells = wells_to_display.get()
        if filtered_wells.empty:
            ui.notification_remove("filt_rate_msg")
            ui.notification_show("No wells selected by current map filters.", type="warning")
            return None

        target_uwis_df = filtered_wells[["GSL_UWI_Std", "OperatorName", "LateralLength", "Formation", "FieldName", "ProvinceState", "FirstProdDate"]].copy()
        target_uwis_df["FirstProdYear"] = target_uwis_df["FirstProdDate"].dt.year.astype(str)
        target_uwis = target_uwis_df["GSL_UWI_Std"].dropna().unique().tolist()

        if not target_uwis:
            ui.notification_remove("filt_rate_msg")
            ui.notification_show("No valid wells to fetch production for.", type="warning")
            return None
        
        # Fetch production data in batches
        all_prod_data = []
        for i in range(0, len(target_uwis), 300):
            batch_uwis = target_uwis[i:i + 300]
            sql_prod = text(f"""
                SELECT GSL_UWI, YEAR, PRODUCT_TYPE, JAN_VOLUME, FEB_VOLUME, MAR_VOLUME, APR_VOLUME, 
                       MAY_VOLUME, JUN_VOLUME, JUL_VOLUME, AUG_VOLUME, SEP_VOLUME, OCT_VOLUME, 
                       NOV_VOLUME, DEC_VOLUME 
                FROM PDEN_VOL_BY_MONTH 
                WHERE GSL_UWI IN :uwis AND ACTIVITY_TYPE = 'PRODUCTION'
            """)
            try:
                with engine.connect() as conn:
                    batch_df = pd.read_sql(sql_prod, conn, params={"uwis": tuple(batch_uwis)})
                    all_prod_data.append(batch_df)
            except Exception as e:
                print(f"Error fetching batch prod data: {e}")
        
        if not all_prod_data:
            ui.notification_remove("filt_rate_msg")
            ui.notification_show("No production data found for filtered wells.", type="warning")
            return None

        full_prod_df = pd.concat(all_prod_data, ignore_index=True)
        full_prod_df['gsl_uwi'] = standardize_uwi(full_prod_df['gsl_uwi'])

        # Melt, convert units, and create dates
        month_cols = {f"{m.upper()}_VOLUME": i+1 for i, m in enumerate(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])}
        prod_long = pd.melt(full_prod_df, id_vars=['gsl_uwi', 'year', 'product_type'], value_vars=month_cols.keys(),
                            var_name='month_col', value_name='volume')
        prod_long['month'] = prod_long['month_col'].map(lambda x: month_cols.get(x.upper()))
        prod_long['prod_date'] = pd.to_datetime(prod_long['year'].astype(str) + '-' + prod_long['month'].astype(str) + '-01')
        
        prod_long.loc[prod_long['product_type'].isin(['OIL', 'CND']), 'vol_converted'] = prod_long['volume'] * M3_TO_BBL
        prod_long.loc[prod_long['product_type'] == 'GAS', 'vol_converted'] = prod_long['volume'] * E3M3_TO_MCF
        
        # Merge with well attributes
        prod_long = pd.merge(prod_long, target_uwis_df, left_on='gsl_uwi', right_on='GSL_UWI_Std')

        # Calculate daily rates and BOE per well
        prod_long['DaysInMonth'] = prod_long['prod_date'].dt.days_in_month
        prod_long['DailyRate'] = prod_long['vol_converted'] / prod_long['DaysInMonth']
        
        prod_pivot = prod_long.pivot_table(index=['GSL_UWI_Std', 'prod_date', 'DaysInMonth', 'LateralLength', input.filtered_group_breakout_by()],
                                           columns='product_type', values='DailyRate').reset_index()
        prod_pivot.fillna(0, inplace=True)
        
        prod_pivot['DailyBOE_well'] = prod_pivot.get('OIL', 0) + prod_pivot.get('CND', 0) + (prod_pivot.get('GAS', 0) / MCF_PER_BOE)
        prod_pivot['DailyBOE_per_1000ft_well'] = np.where(prod_pivot['LateralLength'] > 0, 
                                                         prod_pivot['DailyBOE_well'] / (prod_pivot['LateralLength'] / 1000), 
                                                         np.nan)

        # Time normalization (since peak)
        peak_dates = prod_pivot.loc[prod_pivot.groupby('GSL_UWI_Std')['DailyBOE_well'].idxmax()][['GSL_UWI_Std', 'prod_date']].rename(columns={'prod_date': 'PeakProdDate'})
        prod_from_peak = pd.merge(prod_pivot, peak_dates, on='GSL_UWI_Std')
        prod_from_peak = prod_from_peak[prod_from_peak['prod_date'] >= prod_from_peak['PeakProdDate']]
        prod_from_peak['MonthOnProd'] = ((prod_from_peak['prod_date'] - prod_from_peak['PeakProdDate']).dt.days / AVG_DAYS_PER_MONTH).round() + 1
        
        breakout_col = input.filtered_group_breakout_by()
        normalized_data = prod_from_peak.groupby([breakout_col, 'MonthOnProd']).agg(
            AvgNormBOERate_per_1000ft=('DailyBOE_per_1000ft_well', 'mean'),
            WellCount=('GSL_UWI_Std', 'nunique')
        ).reset_index()
        normalized_data['CumNormBOE_per_1000ft'] = normalized_data.groupby(breakout_col)['AvgNormBOERate_per_1000ft'].transform(lambda x: (x * AVG_DAYS_PER_MONTH).cumsum())

        # Calendar time aggregation
        calendar_data = prod_pivot.groupby([breakout_col, 'prod_date']).agg(
            AvgDailyBOERate_Cal=('DailyBOE_well', 'mean')
        ).reset_index()
        calendar_data = calendar_data.sort_values(['prod_date'])
        calendar_data['TotalMonthlyBOE_Cal'] = calendar_data['AvgDailyBOERate_Cal'] * calendar_data['prod_date'].dt.days_in_month
        calendar_data['CumBOE_Calendar'] = calendar_data.groupby(breakout_col)['TotalMonthlyBOE_Cal'].cumsum()

        ui.notification_remove("filt_rate_msg")
        return {
            "normalized_data": normalized_data,
            "calendar_data": calendar_data,
            "table_data": prod_pivot 
        }

    @render.plot
    def filtered_group_cumulative_plot_normalized():
        results = filtered_group_cumulative_data()
        if results is None or results["normalized_data"].empty:
            return p9.ggplot() + p9.labs(title="No data for normalized rate plot.")
        
        df = results["normalized_data"]
        breakout_col = input.filtered_group_breakout_by()
        
        return (
            p9.ggplot(df, p9.aes(x='MonthOnProd', y='AvgNormBOERate_per_1000ft', color=breakout_col))
            + p9.geom_line()
            + p9.labs(title="Avg Daily BOE Rate per 1000ft Lateral (Time Normalized)",
                      x="Months on Production", y="Avg Daily BOE Rate / 1000ft")
            + p9.theme_minimal()
        )

    @render.plot
    def filtered_group_cumulative_plot_cumulative_boe():
        results = filtered_group_cumulative_data()
        if results is None or results["calendar_data"].empty:
            return p9.ggplot() + p9.labs(title="No data for cumulative BOE plot.")
            
        df = results["calendar_data"]
        breakout_col = input.filtered_group_breakout_by()

        return (
            p9.ggplot(df, p9.aes(x='prod_date', y='CumBOE_Calendar', color=breakout_col))
            + p9.geom_line()
            + p9.scale_y_continuous(labels=lambda l: [f'{int(v/1000):,} MBOE' for v in l])
            + p9.labs(title="Cumulative BOE Production (Calendar Time)",
                      x="Date", y="Cumulative BOE")
            + p9.theme_minimal()
        )

    @render.plot
    def filtered_group_calendar_rate_plot():
        results = filtered_group_cumulative_data()
        if results is None or results["calendar_data"].empty:
            return p9.ggplot() + p9.labs(title="No data for calendar rate plot.")
        
        df = results["calendar_data"]
        breakout_col = input.filtered_group_breakout_by()

        return (
            p9.ggplot(df, p9.aes(x='prod_date', y='AvgDailyBOERate_Cal', color=breakout_col))
            + p9.geom_line()
            + p9.labs(title="Average Daily BOE Rate (Calendar Time)",
                      x="Date", y="Average Daily BOE Rate")
            + p9.theme_minimal()
        )

    @render.data_frame
    def filtered_group_production_table():
        results = filtered_group_cumulative_data()
        if results is None or results["table_data"].empty:
            return pd.DataFrame({"Message": ["Click 'Calculate Rates' to generate data."]})
        
        return results["table_data"].round(2)
        
    @session.download(filename="filtered_group_data.csv")
    def download_filtered_group_prod_data():
        results = filtered_group_cumulative_data()
        if results is not None:
            yield results["table_data"].to_csv(index=False)
        else:
            yield "No data to download."

# --- Arps Logic ---

    def arps_hyperbolic(t, qi, di, b):
        return qi / (1 + b * di * t)**(1/b)

    def arps_exponential(t, qi, di):
        return qi * np.exp(-di * t)

    def arps_harmonic(t, qi, di):
        return qi / (1 + di * t)

    @reactive.Calc
    @reactive.event(input.generate_type_curve)
    def type_curve_analysis_data():
        ui.notification_show("Generating Arps type curve...", duration=None, id="arps_msg")
        filtered_wells = wells_to_display.get()

        if filtered_wells.empty:
            ui.notification_remove("arps_msg")
            ui.notification_show("No wells selected for type curve.", type="warning")
            return None

        target_uwis = filtered_wells["GSL_UWI_Std"].dropna().unique().tolist()
        
        if not target_uwis:
            ui.notification_remove("arps_msg")
            ui.notification_show("No valid wells to fetch production for.", type="warning")
            return None

        # Fetch production data in batches
        all_prod_data = []
        for i in range(0, len(target_uwis), 300):
            batch_uwis = target_uwis[i:i + 300]
            sql_prod = text(f"""
                SELECT GSL_UWI, YEAR, PRODUCT_TYPE, JAN_VOLUME, FEB_VOLUME, MAR_VOLUME, APR_VOLUME, 
                       MAY_VOLUME, JUN_VOLUME, JUL_VOLUME, AUG_VOLUME, SEP_VOLUME, OCT_VOLUME, 
                       NOV_VOLUME, DEC_VOLUME 
                FROM PDEN_VOL_BY_MONTH 
                WHERE GSL_UWI IN :uwis AND ACTIVITY_TYPE = 'PRODUCTION'
            """)
            try:
                with engine.connect() as conn:
                    batch_df = pd.read_sql(sql_prod, conn, params={"uwis": tuple(batch_uwis)})
                    all_prod_data.append(batch_df)
            except Exception as e:
                print(f"Error fetching batch prod data for Arps: {e}")
        
        if not all_prod_data:
            ui.notification_remove("arps_msg")
            ui.notification_show("No production data found for the selected wells.", type="warning")
            return None

        full_prod_df = pd.concat(all_prod_data, ignore_index=True)

        # Melt, convert, and get daily rates
        month_cols = {f"{m.upper()}_VOLUME": i+1 for i, m in enumerate(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])}
        prod_long = pd.melt(full_prod_df, id_vars=['gsl_uwi', 'year', 'product_type'], value_vars=month_cols.keys(),
                            var_name='month_col', value_name='volume')
        prod_long['month'] = prod_long['month_col'].map(lambda x: month_cols.get(x.upper()))
        prod_long['prod_date'] = pd.to_datetime(prod_long['year'].astype(str) + '-' + prod_long['month'].astype(str) + '-01')
        prod_long['gsl_uwi'] = standardize_uwi(prod_long['gsl_uwi'])

        arps_prod = input.arps_product_type()
        if arps_prod == "Oil":
            prod_long = prod_long[prod_long['product_type'].isin(['OIL', 'CND'])]
            prod_long.loc[:, 'vol_converted'] = prod_long['volume'] * M3_TO_BBL
        else: # Gas
            prod_long = prod_long[prod_long['product_type'] == 'GAS']
            prod_long.loc[:, 'vol_converted'] = prod_long['volume'] * E3M3_TO_MCF

        prod_long_agg = prod_long.groupby(['gsl_uwi', 'prod_date'])['vol_converted'].sum().reset_index()
        prod_long_agg['DaysInMonth'] = prod_long_agg['prod_date'].dt.days_in_month
        prod_long_agg['DailyRate'] = prod_long_agg['vol_converted'] / prod_long_agg['DaysInMonth']
        
        # Normalize by peak
        peak_dates = prod_long_agg.loc[prod_long_agg.groupby('gsl_uwi')['DailyRate'].idxmax()][['gsl_uwi', 'prod_date']].rename(columns={'prod_date': 'PeakProdDate'})
        prod_from_peak = pd.merge(prod_long_agg, peak_dates, on='gsl_uwi')
        prod_from_peak = prod_from_peak[prod_from_peak['prod_date'] >= prod_from_peak['PeakProdDate']]
        prod_from_peak['MonthOnProd'] = ((prod_from_peak['prod_date'] - prod_from_peak['PeakProdDate']).dt.days / AVG_DAYS_PER_MONTH).round() + 1

        # Aggregate for type curve
        type_curve_df = prod_from_peak.groupby('MonthOnProd').agg(
            AvgDailyRate=('DailyRate', 'mean'),
            WellCount=('gsl_uwi', 'nunique')
        ).reset_index()
        
        type_curve_df = type_curve_df[type_curve_df['WellCount'] >= 3] # Filter for minimum well count
        if len(type_curve_df) < 3:
            ui.notification_show("Not enough data points to fit a curve.", type="warning")
            return {"data": type_curve_df, "fit_params": None, "fit_curve": None}

        # Fit curve
        model = input.arps_model_type()
        t_data = type_curve_df['MonthOnProd']
        rate_data = type_curve_df['AvgDailyRate']
        
        try:
            if model == "hyperbolic":
                p0 = [rate_data.iloc[0], 0.1, 1.0] # Initial guess for qi, di, b
                params, _ = curve_fit(arps_hyperbolic, t_data, rate_data, p0=p0, bounds=([0, 0, 0], [np.inf, 1, 2]))
            elif model == "exponential":
                p0 = [rate_data.iloc[0], 0.1]
                params, _ = curve_fit(arps_exponential, t_data, rate_data, p0=p0, bounds=([0, 0], [np.inf, 1]))
            else: # Harmonic
                p0 = [rate_data.iloc[0], 0.1]
                params, _ = curve_fit(arps_harmonic, t_data, rate_data, p0=p0, bounds=([0, 0], [np.inf, 1]))

            # Create fitted curve for plotting
            t_fit = np.linspace(t_data.min(), t_data.max(), 100)
            if model == "hyperbolic":
                rate_fit = arps_hyperbolic(t_fit, *params)
            elif model == "exponential":
                rate_fit = arps_exponential(t_fit, *params)
            else:
                rate_fit = arps_harmonic(t_fit, *params)
            fit_curve = pd.DataFrame({'MonthOnProd': t_fit, 'PredictedRate': rate_fit})

        except RuntimeError:
            params = None
            fit_curve = None
            ui.notification_show("Arps curve fitting failed.", type="error")

        ui.notification_remove("arps_msg")
        return {"data": type_curve_df, "fit_params": params, "fit_curve": fit_curve}

    @render.plot
    def arps_type_curve_plot():
        results = type_curve_analysis_data()
        if results is None or results["data"].empty:
            return p9.ggplot() + p9.labs(title="Generate Type Curve to see plot.")
        
        df = results["data"]
        p = (
            p9.ggplot(df, p9.aes(x='MonthOnProd', y='AvgDailyRate'))
            + p9.geom_point(p9.aes(size='WellCount'), alpha=0.6)
            + p9.labs(title="Arps Decline Curve", x="Months Since Peak", y="Average Daily Rate")
            + p9.theme_minimal()
        )
        
        if results["fit_curve"] is not None:
            p += p9.geom_line(data=results["fit_curve"], mapping=p9.aes(x='MonthOnProd', y='PredictedRate'), color='red', linetype='dashed')

        return p
    
    @render.text
    def arps_parameters_output():
        results = type_curve_analysis_data()
        if results is None or results["fit_params"] is None:
            return "Fit parameters will be displayed here."
        
        params = results["fit_params"]
        model = input.arps_model_type()
        
        text = f"Model: {model}\n"
        text += f"Qi (Initial Rate): {params[0]:,.2f}\n"
        text += f"Di (Initial Decline): {params[1]:.4f}\n"
        if model == "hyperbolic":
            text += f"b (Exponent): {params[2]:.4f}\n"
            
        return text
    
    @render.data_frame
    def arps_data_table():
        results = type_curve_analysis_data()
        if results is None or results["data"].empty:
            return pd.DataFrame({"Message": ["Click 'Generate Type Curve' to see data."]})
        return results["data"]
        
    # --- Operator Group Logic ---
    @reactive.Calc
    @reactive.event(input.update_group_plot)
    def operator_group_prod_data():
        req(input.group_operator_filter(), input.group_prod_date_range())
        ui.notification_show("Fetching operator group data...", id="op_group_msg", duration=None)

        selected_operators = input.group_operator_filter()
        if not selected_operators:
            ui.notification_remove("op_group_msg")
            ui.notification_show("Please select at least one operator.", type="warning")
            return None

        uwis_df = wells_gdf_global[wells_gdf_global['OperatorName'].isin(selected_operators)][['GSL_UWI_Std', 'OperatorName']]
        target_uwis = uwis_df['GSL_UWI_Std'].dropna().unique().tolist()

        if not target_uwis:
            ui.notification_remove("op_group_msg")
            ui.notification_show("No wells found for selected operators.", type="warning")
            return None
        
        all_prod_data = []
        for i in range(0, len(target_uwis), 300):
            batch_uwis = target_uwis[i:i + 300]
            sql_prod = text(f"""
                SELECT GSL_UWI, YEAR, PRODUCT_TYPE, JAN_VOLUME, FEB_VOLUME, MAR_VOLUME, APR_VOLUME, 
                       MAY_VOLUME, JUN_VOLUME, JUL_VOLUME, AUG_VOLUME, SEP_VOLUME, OCT_VOLUME, 
                       NOV_VOLUME, DEC_VOLUME 
                FROM PDEN_VOL_BY_MONTH 
                WHERE GSL_UWI IN :uwis AND ACTIVITY_TYPE = 'PRODUCTION'
            """)
            try:
                with engine.connect() as conn:
                    batch_df = pd.read_sql(sql_prod, conn, params={"uwis": tuple(batch_uwis)})
                    all_prod_data.append(batch_df)
            except Exception as e:
                print(f"Error fetching operator group prod data: {e}")

        if not all_prod_data:
            ui.notification_remove("op_group_msg")
            ui.notification_show("No production data found for this operator group.", type="warning")
            return None

        full_prod_df = pd.concat(all_prod_data, ignore_index=True)
        full_prod_df['GSL_UWI_Std'] = standardize_uwi(full_prod_df['gsl_uwi'])

        month_cols = {f"{m.upper()}_VOLUME": i+1 for i, m in enumerate(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])}
        prod_long = pd.melt(full_prod_df, id_vars=['GSL_UWI_Std', 'year', 'product_type'], value_vars=month_cols.keys(),
                            var_name='month_col', value_name='volume')
        prod_long['month'] = prod_long['month_col'].map(lambda x: month_cols.get(x.upper()))
        prod_long['prod_date'] = pd.to_datetime(prod_long['year'].astype(str) + '-' + prod_long['month'].astype(str) + '-01')
        
        prod_long = pd.merge(prod_long, uwis_df.drop_duplicates(), on='GSL_UWI_Std')

        prod_long.loc[prod_long['product_type'].isin(['OIL', 'CND']), 'vol_converted'] = prod_long['volume'] * M3_TO_BBL
        prod_long.loc[prod_long['product_type'] == 'GAS', 'vol_converted'] = prod_long['volume'] * E3M3_TO_MCF
        
        start_date, end_date = input.group_prod_date_range()
        prod_long = prod_long[prod_long['prod_date'].between(pd.to_datetime(start_date), pd.to_datetime(end_date))]
        
        agg_by_operator_month = prod_long.groupby(['OperatorName', 'prod_date', 'product_type'])['vol_converted'].sum().reset_index()
        
        agg_pivot = agg_by_operator_month.pivot_table(index=['OperatorName', 'prod_date'], columns='product_type', values='vol_converted').reset_index()
        agg_pivot.fillna(0, inplace=True)
        agg_pivot = agg_pivot.rename(columns={'OIL':'MonthlyOil', 'CND':'MonthlyCnd', 'GAS':'MonthlyGas'})

        agg_pivot['DaysInMonth'] = agg_pivot['prod_date'].dt.days_in_month
        agg_pivot['AvgOilRateBBLD'] = agg_pivot.get('MonthlyOil', 0) / agg_pivot['DaysInMonth']
        agg_pivot['AvgCndRateBBLD'] = agg_pivot.get('MonthlyCnd', 0) / agg_pivot['DaysInMonth']
        agg_pivot['AvgGasRateMCFD'] = agg_pivot.get('MonthlyGas', 0) / agg_pivot['DaysInMonth']
        agg_pivot['AvgBOERateBBLD'] = (agg_pivot.get('MonthlyOil', 0) + agg_pivot.get('MonthlyCnd', 0) + (agg_pivot.get('MonthlyGas', 0) / MCF_PER_BOE)) / agg_pivot['DaysInMonth']
        
        agg_pivot = agg_pivot.sort_values(['OperatorName', 'prod_date'])
        agg_pivot['CumOilBBL'] = agg_pivot.groupby('OperatorName')['MonthlyOil'].cumsum()
        agg_pivot['CumCndBBL'] = agg_pivot.groupby('OperatorName')['MonthlyCnd'].cumsum()
        agg_pivot['CumGasMCF'] = agg_pivot.groupby('OperatorName')['MonthlyGas'].cumsum()

        ui.notification_remove("op_group_msg")
        return agg_pivot

    @render.plot
    def grouped_cumulative_plot():
        plot_data = operator_group_prod_data()
        if plot_data is None or plot_data.empty:
            return p9.ggplot() + p9.labs(title="Click 'Update Operator Group Plot' to see data.")

        cols_to_plot = []
        selected_prods = input.product_type_filter_analysis()
        if "OIL" in selected_prods: cols_to_plot.append("AvgOilRateBBLD")
        if "CND" in selected_prods: cols_to_plot.append("AvgCndRateBBLD")
        if "GAS" in selected_prods: cols_to_plot.append("AvgGasRateMCFD")
        if "BOE" in selected_prods: cols_to_plot.append("AvgBOERateBBLD")
        
        if not cols_to_plot:
            return p9.ggplot() + p9.labs(title="Select at least one product type.")
            
        plot_data_long = pd.melt(plot_data, id_vars=['OperatorName', 'prod_date'], value_vars=cols_to_plot,
                                 var_name='ProductRateType', value_name='Rate')

        return (
            p9.ggplot(plot_data_long, p9.aes(x='prod_date', y='Rate', color='OperatorName', linetype='ProductRateType'))
            + p9.geom_line()
            + p9.labs(title="Operator Group Average Daily Production Rate", x="Date", y="Average Daily Rate")
            + p9.theme_minimal()
        )
        
    @render.data_frame
    def grouped_production_table():
        table_data = operator_group_prod_data()
        if table_data is None or table_data.empty:
            return pd.DataFrame({"Message": "Click 'Update Operator Group Plot' to see data."})
        return table_data.round(2)

    @session.download(filename="operator_group_data.csv")
    def download_group_prod_data():
        df = operator_group_prod_data()
        if df is not None and not df.empty:
            yield df.to_csv(index=False)
        else:
            yield "No data available."


# --- 6. Run the Application ---
app = App(app_ui, server)