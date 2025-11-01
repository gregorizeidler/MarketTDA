#!/bin/bash

# MarketTDA Setup Script
# Automates environment setup and dependency installation

echo "=================================="
echo "🌀 MarketTDA Setup"
echo "=================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

required_version="3.8"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.8 or higher is required"
    exit 1
fi

echo "✅ Python version OK"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists"
    read -p "Do you want to recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        python3 -m venv venv
    fi
else
    python3 -m venv venv
fi

echo "✅ Virtual environment created"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
echo "This may take a few minutes..."
echo ""

pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ All dependencies installed successfully!"
else
    echo ""
    echo "❌ Error installing dependencies"
    exit 1
fi

# Create output directory
echo ""
echo "Creating output directory..."
mkdir -p output
mkdir -p data
echo "✅ Directories created"

# Run quick test
echo ""
echo "Running quick test..."
python3 -c "
import yfinance as yf
import numpy as np
from ripser import ripser
print('✅ Core libraries imported successfully')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "✅ Setup Complete!"
    echo "=================================="
    echo ""
    echo "Next steps:"
    echo "  1. Activate environment: source venv/bin/activate"
    echo "  2. Run quick demo: python quick_demo.py"
    echo "  3. Run full analysis: python main.py"
    echo "  4. Read tutorial: cat TUTORIAL.md"
    echo ""
    echo "Examples:"
    echo "  python main.py --tickers 50 --period 1y"
    echo "  python main.py --help"
    echo ""
else
    echo ""
    echo "⚠️  Setup completed with warnings"
    echo "Please check error messages above"
fi

