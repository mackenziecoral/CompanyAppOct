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
   Windows and Anaconda users can skip the manual steps by invoking the helper
   script from an Anaconda Prompt:
   ```bat
   run_build_and_app.bat YourCondaEnvName
   ```
   The batch file activates the requested environment, installs the Python
   dependencies, triggers a one-time Oracle cache build, and finally launches
   `streamlit run app.py`.
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

   When connectivity fails, open the **Data source diagnostics** expander at
   the top of the Streamlit UI. It lists which files were discovered, whether
   the Oracle client configuration is valid (including `tnsnames.ora` lookup),
   and which environment variables still need attention. The diagnostics also
   report which Python module (`python-oracledb` or a compatibility layer on
   top of `cx_Oracle`) satisfied the database dependency so you can verify
   Streamlit is using the same interpreter where you installed the package.
3. Launch the app:
   ```bash
   streamlit run app.py
   ```

The Streamlit port now prefers **Oracle → parquet cache → legacy RDS** when
bootstrapping the well master data:

1. If Oracle credentials are configured, the app connects with
   `python-oracledb`, streams the latest well metadata, and uses it directly.
   When the local parquet snapshot is missing the connection response is
   immediately written to disk via `build_cache_from_oracle()` so future runs
   can operate offline.
2. When Oracle access is unavailable, the previously generated
   `processed_app_data*.parquet` snapshot is loaded instead.
3. If both options fail and the optional `pyreadr` dependency is installed,
   the legacy RDS export is treated as a last-resort fallback.

Spatial overlays are still loaded lazily from the shapefile directories.

The top of the Streamlit page includes a compact runtime diagnostics panel
that lists the interpreter path, whether `python-oracledb` was imported
successfully, the status of the parquet cache, and the active DSN/mode. Use it
to verify that Streamlit is using the same environment where Oracle
dependencies were installed.

### Troubleshooting Oracle connectivity

- Ensure the driver is installed in the same Python environment that launches
  Streamlit by running:
  ```bash
  "$(python -c "import sys; print(sys.executable)")" -m pip install oracledb
  ```
  The diagnostics panel shows the interpreter path of the loaded module. If
  Streamlit reports that `python-oracledb` is missing but you already
  installed it elsewhere, re-run the command above inside the environment that
  runs `streamlit`.
- The app automatically falls back to a compatibility layer when only
  `cx_Oracle` is available. While this provides the legacy behaviour, the
  recommended setup uses `python-oracledb` because it supports both thin and
  thick modes in a single package. `build_cache_from_oracle()` will honour the
  thick client when `ORACLE_CLIENT_LIB_DIR` is set and defaults to the thin
  easy-connect mode otherwise.
- When you rely on a TNS alias, confirm that the alias exists in the reported
  `tnsnames.ora` file and that `ORACLE_TNS_ADMIN` (or `TNS_ADMIN`) points to
  the correct directory. The diagnostics expander will highlight missing or
  unreadable files.

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
