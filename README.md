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
2. Configure environment variables if you need to connect to the live
   Oracle database:
   - `ORACLE_USER`
   - `ORACLE_PASSWORD`
   - `ORACLE_DSN`
   You can also override the default file locations with the variables
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

### Why Polars?

Polars executes most transformations in native Rust and automatically
streams larger-than-memory tables. This proved to be more robust than
R/data.table in production where the dashboard would frequently crash.
