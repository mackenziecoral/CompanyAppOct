@echo off
SETLOCAL ENABLEDELAYEDEXPANSION
IF "%~1"=="" (
    ECHO Usage: %~nx0 ^<conda-environment-name^>
    ECHO.
    ECHO Activates the target conda environment, installs dependencies,
    ECHO builds the parquet cache from Oracle, and launches Streamlit.
    EXIT /B 1
)
SET TARGET_ENV=%~1
CALL conda activate %TARGET_ENV%
IF ERRORLEVEL 1 (
    ECHO Failed to activate conda environment %TARGET_ENV%.
    EXIT /B 1
)
python -m pip install --upgrade pip >NUL 2>&1
pip install -r requirements.txt
python -c "from app import build_cache_from_oracle; success, msg = build_cache_from_oracle(); print(msg); import sys; sys.exit(0 if success else 1)"
IF ERRORLEVEL 1 (
    ECHO Unable to build the parquet cache from Oracle. Review the message above.
    EXIT /B 1
)
ECHO Oracle cache build complete.
streamlit run app.py
ENDLOCAL
