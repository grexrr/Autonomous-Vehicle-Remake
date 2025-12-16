#!/usr/bin/env python3
"""
Autonomous Vehicle Application Startup Script
Automatically sets the Qt plugin path and starts the application
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("Starting Autonomous Vehicle Application...")
    
    # Check virtual environment
    venv_path = Path(".venv")
    if not venv_path.exists():
        print("❌ Error: Virtual environment does not exist, please create it first")
        print("Run: python -m venv .venv")
        sys.exit(1)
    
    # Get Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"🐍 Python Version: {python_version}")
    
    # Set Qt paths
    qt_plugin_path = f".venv/lib/python{python_version}/site-packages/PySide6/Qt/plugins"
    qt_lib_path = f".venv/lib/python{python_version}/site-packages/PySide6/Qt/lib"
    qt_plugin_abs_path = os.path.abspath(qt_plugin_path)
    qt_lib_abs_path = os.path.abspath(qt_lib_path)
    
    # Create environment for subprocess
    env = os.environ.copy()
    env["QT_PLUGIN_PATH"] = qt_plugin_abs_path
    env["QT_QPA_PLATFORM_PLUGIN_PATH"] = qt_plugin_abs_path
    env["DYLD_LIBRARY_PATH"] = qt_lib_abs_path
    
    print(f"Set Qt plugin path: {qt_plugin_abs_path}")
    print(f"Set Qt lib path: {qt_lib_abs_path}")
    
    # Check if Qt plugin file exists
    cocoa_plugin = Path(qt_plugin_path) / "platforms" / "libqcocoa.dylib"
    if not cocoa_plugin.exists():
        print("❌ Error: Qt plugin file does not exist")
        sys.exit(1)
    
    # Fix Qt plugin rpath (macOS only)
    if sys.platform == "darwin":
        print("Fixing Qt plugin rpath...")
        try:
            subprocess.run([
                "install_name_tool", 
                "-add_rpath", 
                qt_lib_abs_path,
                str(cocoa_plugin)
            ], check=False, capture_output=True)
        except FileNotFoundError:
            print("⚠️  Warning: install_name_tool not found, skipping rpath fix")
    
    # Start the application using subprocess
    print("Starting the application...")
    try:
        # Use subprocess with the modified environment
        subprocess.run([sys.executable, "-m", "AutonomousVehicle"], check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Application failed to start: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nApplication has exited")
    
    print("Application has exited")

if __name__ == "__main__":
    main()
