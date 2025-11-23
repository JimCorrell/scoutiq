#!/bin/bash
# Quick setup script for ScoutIQ

echo "================================================"
echo "ScoutIQ - Quick Setup Script"
echo "================================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Download spaCy model
echo ""
echo "Downloading spaCy language model..."
python -m spacy download en_core_web_lg

# Create necessary directories
echo ""
echo "Creating data directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/models
mkdir -p logs
mkdir -p results

# Generate sample data
echo ""
echo "Generating sample data..."
python scripts/generate_sample_data.py

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run an example:"
echo "  python scripts/run_example.py"
echo ""
echo "To run comprehensive demo:"
echo "  python examples/comprehensive_example.py"
echo ""
echo "To run tests:"
echo "  pytest tests/"
echo ""
