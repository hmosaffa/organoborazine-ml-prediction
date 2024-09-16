
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_data(file_path):
    data = pd.read_excel(file_path)
    X = data.iloc[:, 1:-1]  # Assuming the first column is the name and last column is the target
    y = data.iloc[:, -1]    # The last column is the target (Lambda max)
    return X, y

def preprocess_data(X, y, test_size=0.2, random_state=42, use_pca=False, pca_components=0.95):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if use_pca:
        pca = PCA(n_components=pca_components, random_state=random_state)
        X_train_processed = pca.fit_transform(X_train_scaled)
        X_test_processed = pca.transform(X_test_scaled)
        print(f"Number of components after PCA: {pca.n_components_}")
    else:
        X_train_processed = X_train_scaled
        X_test_processed = X_test_scaled
    
    return X_train_processed, X_test_processed, y_train, y_test

def train_model(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    rf_best = RandomForestRegressor(**best_params, random_state=42)
    rf_best.fit(X_train, y_train)
    
    return rf_best, best_params

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    bias = np.mean(y_pred - y_test)
    cor = np.corrcoef(y_test, y_pred)[0, 1]
    
    metrics = {
        "MSE": mse,
        "MAE": mae,
        "R2": r2,
        "RMSE": rmse,
        "Bias": bias,
        "COR": cor
    }
    
    return metrics

def main():
    # Load and preprocess data
    X, y = load_data('data/bor.xlsx')
    X_train, X_test, y_train, y_test = preprocess_data(X, y, use_pca=True, pca_components=0.95)
    
    # Train model
    model, best_params = train_model(X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Print results
    print("Best Parameters:", best_params)
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

if __name__ == "__main__":
    main()
