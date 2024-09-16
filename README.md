# Organoborazine ML Prediction

This project uses machine learning to predict the maximum absorption wavelength (λ_max) of organoborazines based on their structural and chemical properties. We employ a Random Forest Regressor model, optimized through hyperparameter tuning.

## Project Overview

Organoborazines are an important class of compounds with potential applications in optoelectronic devices. This project aims to accelerate the discovery and design of new organoborazines by predicting their λ_max values using machine learning techniques.

## Features

- Data preprocessing and feature scaling
- Random Forest Regression model
- Hyperparameter tuning using GridSearchCV
- Model evaluation using various metrics (MSE, MAE, R2, RMSE, Bias, Correlation Coefficient)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/hmosaffa/organoborazine-ml-prediction.git
   cd organoborazine-ml-prediction
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the model:

```
python main.py
```

This will load the data, preprocess it, train the model, and print the best parameters and evaluation metrics.

