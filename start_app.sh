#!/bin/bash

# Autonomous Vehicle Application Startup Script
# Automatically sets the Qt plugin path and starts the application

echo "Starting Autonomous Vehicle Application..."

# Check if the virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Error: Virtual environment does not exist, please create it first"
    echo "Run: python -m venv .venv"
    exit 1
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Check if PySide6 is installed
if ! python -c "import PySide6" 2>/dev/null; then
    echo "❌ Error: PySide6 is not installed, installing now..."
    pip install PySide6==6.8.0
fi

# Get Python version
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "🐍 Python Version: $PYTHON_VERSION"

# Set Qt plugin path
QT_PLUGIN_PATH="$(pwd)/.venv/lib/python$PYTHON_VERSION/site-packages/PySide6/Qt/plugins"
export QT_PLUGIN_PATH

echo "🔧 Set Qt plugin path: $QT_PLUGIN_PATH"

# Check if Qt plugin file exists
if [ ! -f "$QT_PLUGIN_PATH/platforms/libqcocoa.dylib" ]; then
    echo "❌ Error: Qt plugin file does not exist"
    exit 1
fi

# Fix Qt plugin rpath (if needed)
echo "Fixing Qt plugin rpath..."
install_name_tool -add_rpath "$(pwd)/.venv/lib/python$PYTHON_VERSION/site-packages/PySide6/Qt/lib" "$QT_PLUGIN_PATH/platforms/libqcocoa.dylib" 2>/dev/null || true

# Start the application
echo "Starting the application..."
python -m AutonomousVehicle

echo "Application has exited"
