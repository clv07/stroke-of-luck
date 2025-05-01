#!/bin/bash
set -e

echo "Enter desired virtual environment name: "
read ve

echo "Creating conda virtual environment..."
conda create -n $ve -y

echo "Activating environment..."
conda activate $ve

echo "Installing seaborn..."
pip install seaborn

echo "Installing scikit-learn..."
pip install scikit-learn

echo "Installing tensorflow..."
pip install tensorflow==2.11.0

echo "All required packages installed."