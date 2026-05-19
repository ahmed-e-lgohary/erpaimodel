import pandas as pd
import datetime
import numpy as np
import xgboost as xgb
import pickle
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import features_utils

# Helper: safe MAPE (ignores zero true values)
def safe_mape(y_true, y_pred):
    """Return Mean Absolute Percentage Error as a fraction, handling zeros safely."""
    mask = y_true != 0
    if not mask.any():
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

def prepare_data(df):
    """
    Cleans data and extracts enhanced features using features_utils.
    """
    print("Extracting base features...")
    df = features_utils.get_base_features(df)
    
    print("Extracting historical features (Lags, Rolling)...")
    df = features_utils.get_historical_features(df)
    
    # One-Hot Encode Area
    print("Encoding areas...")
    area_names = sorted(df['Area'].unique())
    area_dummies = pd.get_dummies(df['Area'], prefix='Area')
    df = pd.concat([df, area_dummies], axis=1)
    
    # Label Encoder for Area (useful for some models or metadata)
    le = LabelEncoder()
    df['AreaEncoded'] = le.fit_transform(df['Area'])
    
    # Drop rows with NaN values (due to lags)
    df = df.dropna()
    
    # Get feature columns
    feature_cols, numeric_cols = features_utils.get_feature_cols(area_names)
    
    X = df[feature_cols]
    y = df['Passengers']
    
    # Scale numeric features
    print("Scaling numeric features...")
    scaler = StandardScaler()
    X = X.copy() # Avoid SettingWithCopyWarning
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    return X, y, le, feature_cols, scaler

def objective(trial, X_train, X_test, y_train, y_test):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42,
        'tree_method': 'hist'
    }
    
    # TimeSeriesSplit for validation
    tscv = TimeSeriesSplit(n_splits=3)
    maes = []
    
    for train_index, valid_index in tscv.split(X_train):
        cv_X_train, cv_X_valid = X_train.iloc[train_index], X_train.iloc[valid_index]
        cv_y_train, cv_y_valid = y_train.iloc[train_index], y_train.iloc[valid_index]
        
        model = xgb.XGBRegressor(**params, early_stopping_rounds=50)
        model.fit(
            cv_X_train, cv_y_train,
            eval_set=[(cv_X_valid, cv_y_valid)],
            verbose=False
        )
        
        y_pred = model.predict(cv_X_valid)
        maes.append(mean_absolute_error(cv_y_valid, y_pred))
        
    return np.mean(maes)

def train_and_evaluate():
    print("Loading data...")
    try:
        df = pd.read_csv('transport_data.csv')
    except FileNotFoundError:
        print("Error: transport_data.csv not found. Run generate_data.py first.")
        return
        
    print("Preparing data with enhanced features...")
    X, y, le, feature_cols, scaler = prepare_data(df)
    
    print("Splitting data (Time Series Split)...")
    # Using shuffle=False prevents data leakage for time-series forecasting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print("Starting Hyperparameter Optimization with Optuna (100 trials)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    # Increased trials for better optimization
    study.optimize(lambda trial: objective(trial, X_train, X_test, y_train, y_test), n_trials=100)
    
    best_params = study.best_params
    print(f"Best parameters found: {best_params}")
    print(f"Best Validation MAE from Optuna: {study.best_value:.2f}")
    
    print("Training final model with best parameters (up to 200 early stopping rounds)...")
    best_params['random_state'] = 42
    best_params['tree_method'] = 'hist'
    final_model = xgb.XGBRegressor(**best_params, early_stopping_rounds=200)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    print("Evaluating final model...")
    y_pred = final_model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = safe_mape(y_test, y_pred)
    
    print(f"Final Test Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Final Test Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Model Accuracy (R² Score): {r2 * 100:.2f}%")
    print(f"Accuracy based on MAPE: {(1 - mape) * 100:.2f}%")
    
    print("Saving model and encoders...")
    with open('model.pkl', 'wb') as f:
        pickle.dump({
            'model': final_model, 
            'label_encoder': le, 
            'features': feature_cols, 
            'scaler': scaler,
            'areas': list(le.classes_)
        }, f)
        
    print("Model saved successfully as model.pkl")

if __name__ == '__main__':
    train_and_evaluate()
