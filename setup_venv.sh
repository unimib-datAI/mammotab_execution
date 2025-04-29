#!/bin/bash

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install other requirements
pip install -r requirements.txt