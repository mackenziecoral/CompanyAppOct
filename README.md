# CompanyApp

## Streamlit Port

The original application shipped as an R/Shiny dashboard. This repository
now contains a Python/Streamlit reimplementation that uses Polars and
GeoPandas to work with larger datasets without crashing.

### Getting started

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   If you are using Anaconda/Spyder instead of a virtual environment,
   install the required packages with:
   ```bash
   conda install -c conda-forge polars pyarrow streamlit geopandas plotly pyreadr
   ```
   When the legacy `processed_app_data_vMERGED_FINAL_v24_DB_sticks_latlen.rds`
   snapshot is present alongside the app, the dashboard will automatically
   convert it to parquet on first launch. Ensure the optional `pyreadr`
   dependency is installed if you plan to rely on this fallback.
2. Configure environment variables if you need to connect to the live
   Oracle database:
   - `ORACLE_USER`
   - `ORACLE_PASSWORD`
   - `ORACLE_DSN`

   The Streamlit app also understands optional Oracle client settings:
   - `ORACLE_TNS_ADMIN` or `TNS_ADMIN` – folder containing `tnsnames.ora`
   - `ORACLE_CONFIG_DIR` – alternate name for the same directory
   - `ORACLE_WALLET_DIR` – wallet location when database access requires it
   - `ORACLE_CLIENT_LIB_DIR` – Instant Client directory if thick mode is needed

   A sample `tnsnames.ora` matching the legacy Shiny setup ships with the
   repository. Point `ORACLE_TNS_ADMIN` at the repository root or copy the
   file into your corporate Oracle client directory. Set `ORACLE_DSN` to the
   alias defined inside `tnsnames.ora` (for example `GDC_LINK`).

   Credentials and paths can also be supplied via `~/.streamlit/secrets.toml`:
   ```toml
   [oracle]
   user = "WOODMAC"
   password = "c0pp3r"
   dsn = "GDC_LINK"
   tns_admin = "C:/oracle/network/admin"
   ```

   You can override the default file locations with the variables
   `APP_BASE_PATH`, `APP_PROCESSED_PARQUET`, `APP_WOODMACK_COVERAGE_XLSX`,
   `APP_SUBPLAY_SHAPEFILES` and `APP_COMPANY_SHAPEFILES`.
3. Launch the app:
   ```bash
   streamlit run app.py
   ```

When a processed parquet file is present it is used as the primary data
source. If that file is absent the application will connect to the
Oracle database and stream the well master data. Spatial overlays are
loaded lazily from the shapefile directories.

If neither a parquet snapshot nor a live database is available, placing
the original RDS export in the application directory provides a
zero-configuration offline option. The Streamlit port reads the RDS file
with `pyreadr`, writes a parquet copy for subsequent runs, and exposes
the same well metadata used in the Shiny application.

### Production analytics

The Streamlit port now mirrors the analytical workflows offered in the
Shiny implementation:

- **Single well analysis** – inspect converted monthly volumes, daily
  rates, and cumulative totals for a selected well and download the
  results.
- **Filtered group cumulative** – aggregate production for the set of
  wells currently visible on the map, respecting the product filters and
  custom date window.
- **Type curve analysis** – align wells by months on production to build
  average and percentile curves using BOE rates.
- **Operator group cumulative** – compare operator-level production over
  a chosen period independent of the spatial filters.

All production queries stream monthly PDEN data from Oracle in batches
and cache the result to `.parquet` files inside the `cache/` directory
to avoid repeat round-trips during a session.

### Why Polars?

Polars executes most transformations in native Rust and automatically
streams larger-than-memory tables. This proved to be more robust than
R/data.table in production where the dashboard would frequently crash.
