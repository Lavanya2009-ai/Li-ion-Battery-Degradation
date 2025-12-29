# Battery Degradation Analysis

A comprehensive pipeline for analyzing battery degradation under mechanical stress using multi-factor transformer models.

## Features

- **Data Loading**: Automatically processes hierarchical battery datasets (288 Excel files)
- **Feature Engineering**: Extracts 50+ features including voltage statistics, current patterns, and capacity trends
- **Multi-Factor Transformer**: Neural network model that accounts for pressure, temperature, and C-rate effects
- **Validation**: Leave-one-battery-out cross-validation for robust evaluation
- **Analysis**: Quantifies pressure effects on battery capacity

## Project Structure
# Battery Degradation Analysis

A comprehensive pipeline for analyzing battery degradation under mechanical stress using multi-factor transformer models.

## Features

- **Data Loading**: Automatically processes hierarchical battery datasets (288 Excel files)
- **Feature Engineering**: Extracts 50+ features including voltage statistics, current patterns, and capacity trends
- **Multi-Factor Transformer**: Neural network model that accounts for pressure, temperature, and C-rate effects
- **Validation**: Leave-one-battery-out cross-validation for robust evaluation
- **Analysis**: Quantifies pressure effects on battery capacity

## Project Structure
battery-analysis/
├── data_loader.py # Script 1: Load and combine Excel files
├── feature_extractor.py # Script 2: Feature engineering
├── transformer_model.py # Script 3: Transformer model with validation
├── requirements.txt # Python dependencies
├── README.md # This file
├── data/ # Battery data (not included in repo)
│ ├── 10d/ # Temperature: 10°C
│ │ ├── 300N/ # Pressure: 300N
│ │ │ ├── 0.5C/ # C-rate: 0.5C
│ │ │ │ ├── Battery No.1.xls
│ │ │ │ └── ...
│ │ │ └── 1C/
│ │ └── 400N/
│ ├── 25d/ # Temperature: 25°C
│ └── 40d/ # Temperature: 40°C
└── results/ # Output files (generated)
├── combined_battery_data.csv
├── enhanced_battery_data_with_features.csv
├── pressure_effect_analysis.png
└── final_results.json
