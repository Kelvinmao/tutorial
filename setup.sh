#!/usr/bin/env bash
set -e

echo "=== AI Compiler Tutorial Setup ==="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Please install Python 3.9+."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Found Python $PYTHON_VERSION"

# Check gcc
if ! command -v gcc &> /dev/null; then
    echo "WARNING: gcc not found. C chapters (ch07, ch13, ch16, ch17) require gcc."
    echo "  Install with: sudo apt-get install build-essential"
fi

# Check graphviz system binary
if ! command -v dot &> /dev/null; then
    echo "WARNING: graphviz 'dot' not found. Visualization scripts require it."
    echo "  Install with: sudo apt-get install graphviz"
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=== Setup complete! ==="
echo "Activate the environment with: source venv/bin/activate"
echo "Start learning:  cd part1_compiler_fundamentals/ch01_what_is_a_compiler && python demo_pipeline.py"
