#!/bin/bash
# One-time setup script for NBA Lines Dashboard

echo "🏀 NBA Lines Dashboard Setup"
echo "----------------------------"

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run the dashboard:"
echo "  source venv/bin/activate"
echo "  streamlit run app.py"
