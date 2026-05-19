import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Define official holidays (YYYY-MM-DD)
HOLIDAYS = {
    datetime.date(2024, 1, 1),   # New Year's Day
    datetime.date(2024, 1, 7),   # Coptic Christmas
    datetime.date(2024, 4, 25),  # Sinai Liberation Day
    datetime.date(2024, 5, 1),   # Labour Day
    datetime.date(2024, 6, 18),  # Revolution Day
    datetime.date(2024, 7, 23),  # Armed Forces Day
    datetime.date(2024, 10, 6),  # Armed Forces Day (alternative)
}

def get_base_features(df):
    """
    Extracts basic temporal and holiday features.
    """
    # Ensure Date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Basic temporal
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x in [4, 5] else 0)
    df['Quarter'] = df['Date'].dt.quarter
    df['Month'] = df['Date'].dt.month
    df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)
    df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)
    
    # Holiday features
    df['IsHoliday'] = df['Date'].dt.date.apply(lambda x: 1 if x in HOLIDAYS else 0)
    
    # Day before/after holiday
    df['IsDayBeforeHoliday'] = (df['Date'] + datetime.timedelta(days=1)).dt.date.apply(lambda x: 1 if x in HOLIDAYS else 0)
    df['IsDayAfterHoliday'] = (df['Date'] - datetime.timedelta(days=1)).dt.date.apply(lambda x: 1 if x in HOLIDAYS else 0)
    
    # Exam Season
    df['IsExamSeason'] = df['Month'].isin([1, 5, 6, 12]).astype(int)
    
    # Time features
    if 'Time' in df.columns:
        df['Hour'] = df['Time'].apply(lambda x: int(x.split(':')[0]) if isinstance(x, str) else x)
    elif 'Hour' not in df.columns:
        df['Hour'] = df['Date'].dt.hour
        
    # Cyclical encoding for Hour
    df['Hour_sin'] = np.sin(df['Hour'] * (2. * np.pi / 24))
    df['Hour_cos'] = np.cos(df['Hour'] * (2. * np.pi / 24))
    
    # Rush Hour flag (7-9 AM and 4-6 PM are typical in Egypt)
    df['IsRushHour'] = df['Hour'].apply(lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 18) else 0)
    
    # Interaction: Hour * Weekend
    df['Hour_Weekend_Interaction'] = df['Hour'] * (df['IsWeekend'] + 1)
    
    return df

def get_historical_features(df):
    """
    Calculates lag and rolling statistics. 
    Assumes df is sorted by Area and DateTime.
    """
    if 'DateTime' not in df.columns:
        if 'Time' in df.columns:
             df['DateTime'] = pd.to_datetime(df['Date'].dt.strftime('%Y-%m-%d') + ' ' + df['Time'])
        else:
             df['DateTime'] = df['Date']

    df = df.sort_values(by=['Area', 'DateTime'])
    
    # Lags
    df['Lag_1H'] = df.groupby('Area')['Passengers'].shift(1)
    df['Lag_2H'] = df.groupby('Area')['Passengers'].shift(2)
    df['Lag_3H'] = df.groupby('Area')['Passengers'].shift(3)
    df['Lag_24H'] = df.groupby('Area')['Passengers'].shift(24)
    df['Lag_168H'] = df.groupby('Area')['Passengers'].shift(168)
    
    # Rolling Means
    df['Rolling_Mean_3H'] = df.groupby('Area')['Passengers'].transform(lambda x: x.shift(1).rolling(window=3).mean())
    df['Rolling_Mean_24H'] = df.groupby('Area')['Passengers'].transform(lambda x: x.shift(1).rolling(window=24).mean())
    df['Rolling_Mean_168H'] = df.groupby('Area')['Passengers'].transform(lambda x: x.shift(1).rolling(window=168).mean())
    
    # Rolling Stds (New)
    df['Rolling_Std_3H'] = df.groupby('Area')['Passengers'].transform(lambda x: x.shift(1).rolling(window=3).std())
    df['Rolling_Std_24H'] = df.groupby('Area')['Passengers'].transform(lambda x: x.shift(1).rolling(window=24).std())
    
    # Fill NaNs from rolling std with 0 (initial points)
    df['Rolling_Std_3H'] = df['Rolling_Std_3H'].fillna(0)
    df['Rolling_Std_24H'] = df['Rolling_Std_24H'].fillna(0)
    
    return df

def get_feature_cols(area_names):
    """
    Returns the consistent list of feature columns.
    """
    numeric_cols = [
        'DayOfWeek', 'IsWeekend', 'IsHoliday', 'IsDayBeforeHoliday', 'IsDayAfterHoliday',
        'Quarter', 'IsMonthEnd', 'IsMonthStart', 'IsExamSeason', 'IsRushHour',
        'Hour', 'Hour_sin', 'Hour_cos', 'Hour_Weekend_Interaction',
        'Lag_1H', 'Lag_2H', 'Lag_3H', 'Lag_24H', 'Lag_168H',
        'Rolling_Mean_3H', 'Rolling_Mean_24H', 'Rolling_Mean_168H',
        'Rolling_Std_3H', 'Rolling_Std_24H'
    ]
    area_cols = [f'Area_{area}' for area in area_names]
    return numeric_cols + area_cols, numeric_cols
