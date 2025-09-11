#!/bin/bash
# 快速启动脚本 - 最小化版本

source .venv/bin/activate
export QT_PLUGIN_PATH="$(pwd)/.venv/lib/python$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages/PySide6/Qt/plugins"
python -m AutonomousVehicle
