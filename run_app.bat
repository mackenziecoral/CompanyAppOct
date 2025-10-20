@echo off
setlocal
cd /d "%~dp0"
call conda activate companyapp_py
streamlit run app.py
endlocal
