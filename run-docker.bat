@echo off
REM Simple startup script for Windows

echo.
echo ====================================
echo Fraud Detection API - Docker Setup
echo ====================================
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not installed or not in PATH
    echo Please download Docker Desktop from https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

echo [1/3] Building Docker image...
docker build -t fraud-detection:latest .
if errorlevel 1 (
    echo ERROR: Failed to build image
    pause
    exit /b 1
)

echo.
echo [2/3] Starting container with docker-compose...
docker-compose up

echo.
echo [3/3] Container is running!
echo.
echo Your API is available at: http://localhost:5000
echo.
pause
