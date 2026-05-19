from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import features_utils
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model and encoder
try:
    with open('model.pkl', 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        le = data['label_encoder']
        feature_cols = data.get('features', None)
        scaler = data.get('scaler', None)
        areas_list = data.get('areas', list(le.classes_) if le else [])
except FileNotFoundError:
    print("Warning: model.pkl not found. Predictions will not work until the model is trained.")
    model = None
    le = None
    feature_cols = None
    areas_list = []

# Load historical data for lag features
try:
    historical_df = pd.read_csv('transport_data.csv')
    historical_df['DateTime'] = pd.to_datetime(historical_df['Date'] + ' ' + historical_df['Time'])
    
    # Calculate average passengers per area/hour for fallbacks
    historical_df['Hour'] = historical_df['DateTime'].dt.hour
    avg_demand = historical_df.groupby(['Area', 'Hour'])['Passengers'].mean().to_dict()
    
    # Set index to Area and DateTime for fast lookups
    historical_df.set_index(['Area', 'DateTime'], inplace=True)
    historical_df.sort_index(inplace=True)
    
except FileNotFoundError:
    print("Warning: transport_data.csv not found. Lag features fallback will be used.")
    historical_df = None
    avg_demand = {}

@app.route('/')
def index():
    return render_template('index.html', areas=areas_list)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or le is None:
        return jsonify({'error': 'Model not trained yet.'}), 500
        
    try:
        data = request.json
        date_str = data['date']
        time_str = data['time']
        area = data['area']
        
        # 1. Process inputs into a DataFrame
        dt = datetime.strptime(date_str + ' ' + time_str, '%Y-%m-%d %H:%M')
        input_df = pd.DataFrame([{
            'Date': dt.date(),
            'Time': time_str,
            'Area': area
        }])
        
        # 2. Extract base features using shared utility
        input_df = features_utils.get_base_features(input_df)
        
        # 3. Calculate lag and rolling features manually for single prediction
        # (This mirrors the logic in features_utils.get_historical_features but for one row)
        lag_1h, lag_2h, lag_3h, lag_24h, lag_168h = 0, 0, 0, 0, 0
        rolling_3h, rolling_24h, rolling_168h = 0, 0, 0
        rolling_std_3h, rolling_std_24h = 0, 0
        
        if historical_df is not None:
            def get_lag_value(target_dt):
                try:
                    val = historical_df.loc[(area, target_dt), 'Passengers']
                    return val.iloc[0] if isinstance(val, pd.Series) else val
                except KeyError:
                    return avg_demand.get((area, target_dt.hour), 0)
            
            lag_1h = get_lag_value(dt - timedelta(hours=1))
            lag_2h = get_lag_value(dt - timedelta(hours=2))
            lag_3h = get_lag_value(dt - timedelta(hours=3))
            lag_24h = get_lag_value(dt - timedelta(hours=24))
            lag_168h = get_lag_value(dt - timedelta(hours=168))
            
            def get_rolling_stats(target_dt, window_size):
                try:
                    area_data = historical_df.xs(area, level='Area')
                    prev_data = area_data[:target_dt].tail(window_size)
                    if len(prev_data) > 0:
                        return prev_data['Passengers'].mean(), prev_data['Passengers'].std()
                    return avg_demand.get((area, target_dt.hour), 0), 0
                except:
                    return avg_demand.get((area, target_dt.hour), 0), 0
            
            rolling_3h, rolling_std_3h = get_rolling_stats(dt, 3)
            rolling_24h, rolling_std_24h = get_rolling_stats(dt, 24)
            rolling_168h, _ = get_rolling_stats(dt, 168)
            
        # Add to dataframe
        input_df['Lag_1H'] = lag_1h
        input_df['Lag_2H'] = lag_2h
        input_df['Lag_3H'] = lag_3h
        input_df['Lag_24H'] = lag_24h
        input_df['Lag_168H'] = lag_168h
        input_df['Rolling_Mean_3H'] = rolling_3h
        input_df['Rolling_Mean_24H'] = rolling_24h
        input_df['Rolling_Mean_168H'] = rolling_168h
        input_df['Rolling_Std_3H'] = np.nan_to_num(rolling_std_3h)
        input_df['Rolling_Std_24H'] = np.nan_to_num(rolling_std_24h)
        
        # 4. Add Area dummy variables
        for cls in areas_list:
            input_df[f'Area_{cls}'] = 1 if area == cls else 0
            
        # 5. Get consistent feature column list
        all_feature_cols, numeric_cols = features_utils.get_feature_cols(areas_list)
        
        # 6. Scale numeric features
        if scaler is not None:
            input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
        
        # 7. Reorder columns to match training exactly
        input_data = input_df[all_feature_cols]
        
        # 8. Predict
        prediction = model.predict(input_data)[0]
        prediction = max(0, int(prediction))
        
        return jsonify({
            'success': True,
            'passengers': prediction,
            'area': area,
            'datetime': f"{date_str} {time_str}"
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 400

@app.route('/api/dashboard_data')
def dashboard_data():
    if historical_df is None:
        return jsonify({'error': 'No historical data available'}), 404
        
    hourly_trend = {}
    for (area, hour), mean_passengers in avg_demand.items():
        if area not in hourly_trend:
            hourly_trend[area] = [0] * 24
        hourly_trend[area][hour] = round(float(mean_passengers), 1)
        
    total_per_area = historical_df.groupby(level='Area')['Passengers'].sum().to_dict()
    # Convert int64 to int for JSON serialization
    total_per_area = {k: int(v) for k, v in total_per_area.items()}
    
    return jsonify({
        'hourly_trend': hourly_trend,
        'total_per_area': total_per_area
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
