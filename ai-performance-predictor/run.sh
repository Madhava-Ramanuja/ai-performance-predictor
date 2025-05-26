#!/bin/bash

# Make sure models directory exists
mkdir -p models

# Train models if they don't exist
if [ ! -f "models/random_forest_model.pkl" ] || [ ! -f "models/logistic_regression_model.pkl" ]; then
    echo "Training ML models..."
    python train_models.py
fi

# Start the Flask server
echo "Starting Flask server..."
python app.py