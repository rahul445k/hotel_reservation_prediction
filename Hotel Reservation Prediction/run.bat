@echo off
REM Hotel Reservation Prediction - Run Scripts Helper
REM This script helps you run the ML pipeline components easily

echo ========================================
echo Hotel Reservation Prediction MLOps
echo ========================================
echo.

:menu
echo Please select an option:
echo.
echo 1. Run Complete Training Pipeline
echo 2. Run Data Ingestion Only
echo 3. Run Data Preprocessing Only
echo 4. Run Model Training Only
echo 5. Start Web Application
echo 6. Test Imports
echo 7. View Latest Logs
echo 8. Set GCP Credentials
echo 9. Exit
echo.

set /p choice="Enter your choice (1-9): "

if "%choice%"=="1" goto pipeline
if "%choice%"=="2" goto ingestion
if "%choice%"=="3" goto preprocessing
if "%choice%"=="4" goto training
if "%choice%"=="5" goto webapp
if "%choice%"=="6" goto test
if "%choice%"=="7" goto logs
if "%choice%"=="8" goto credentials
if "%choice%"=="9" goto end

echo Invalid choice! Please try again.
echo.
goto menu

:pipeline
echo.
echo Running Complete Training Pipeline...
echo ========================================
python pipeline/training_pipeline.py
echo.
echo Pipeline execution completed!
pause
goto menu

:ingestion
echo.
echo Running Data Ingestion...
echo ========================================
python run_data_ingestion.py
echo.
echo Data ingestion completed!
pause
goto menu

:preprocessing
echo.
echo Running Data Preprocessing...
echo ========================================
python run_data_preprocessing.py
echo.
echo Data preprocessing completed!
pause
goto menu

:training
echo.
echo Running Model Training...
echo ========================================
python run_model_training.py
echo.
echo Model training completed!
pause
goto menu

:webapp
echo.
echo Starting Flask Web Application...
echo ========================================
echo The app will be available at: http://localhost:8080
echo Press Ctrl+C to stop the server
echo.
python application.py
pause
goto menu

:test
echo.
echo Testing Imports...
echo ========================================
python -c "from src.logger import get_logger; print('✅ Logger import: OK')"
python -c "from src.data_ingestion import DataIngestion; print('✅ Data Ingestion import: OK')"
python -c "from src.data_preprocessing import DataProcessor; print('✅ Data Preprocessing import: OK')"
python -c "from src.model_training import ModelTraining; print('✅ Model Training import: OK')"
python -c "from utils.common_functions import read_yaml; print('✅ Utils import: OK')"
echo.
echo All imports successful!
pause
goto menu

:logs
echo.
echo Latest Log Entries:
echo ========================================
for /f %%i in ('dir /b /o-d logs\*.log') do (
    echo Showing last 30 lines from: %%i
    echo.
    powershell -Command "Get-Content logs\%%i -Tail 30"
    goto :logsdone
)
:logsdone
echo.
pause
goto menu

:credentials
echo.
echo Setting GCP Credentials...
echo ========================================
echo.
echo Available credential files:
dir /b gcp_credentials\*.json 2>nul | findstr /v "template"
echo.
set /p credfile="Enter the credential filename (or press Enter to skip): "
if "%credfile%"=="" goto menu
set GOOGLE_APPLICATION_CREDENTIALS=gcp_credentials\%credfile%
echo.
echo ✅ Credentials set to: %GOOGLE_APPLICATION_CREDENTIALS%
echo.
echo Note: This is only set for this session.
echo To make it permanent, add it to your system environment variables.
pause
goto menu

:end
echo.
echo Thank you for using Hotel Reservation Prediction MLOps!
echo.
exit /b 0
