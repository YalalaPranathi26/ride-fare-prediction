"""
Ride Fare Prediction — Flask Web Application
=============================================
Loads trained XGBoost model from pickle. User fills trip details;
derived/engineered features are computed automatically on the backend.
Returns predicted fare in INR.
"""

import os, pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(PARENT_DIR, 'Dataset', 'ride_fare_model.pkl')

# ── Load model bundle ─────────────────────────────────────────────────────────
with open(MODEL_PATH, 'rb') as f:
    bundle = pickle.load(f)

model        = bundle['model']
encoders     = bundle['encoders']
feature_cols = bundle['feature_cols']
model_name   = bundle['model_name']
model_r2     = bundle['r2']
model_mae    = bundle['mae']
model_mape   = bundle['mape']

print(f"[OK] Model    : {model_name}")
print(f"[OK] R2       : {model_r2}")
print(f"[OK] MAE      : Rs.{model_mae}")
print(f"[OK] Features : {len(feature_cols)}")

# ── City → Zone mapping ───────────────────────────────────────────────────────
CITY_ZONES = {
    'Bangalore': ['Koramangala','Indiranagar','Whitefield','Electronic City',
                  'Hebbal','JP Nagar','HSR Layout','Marathahalli','BTM Layout','Jayanagar'],
    'Mumbai'   : ['Andheri','Bandra','Dadar','Kurla','Thane','Borivali','Colaba','Worli','Malad','Powai'],
    'Delhi'    : ['Connaught Place','Lajpat Nagar','Dwarka','Rohini','Noida',
                  'Gurugram','Saket','Karol Bagh','Pitampura','Greater Kailash'],
    'Hyderabad': ['Hitech City','Banjara Hills','Kukatpally','Uppal','Secunderabad',
                  'Gachibowli','Madhapur','LB Nagar','Dilsukhnagar','Ameerpet'],
    'Chennai'  : ['T Nagar','Anna Nagar','Velachery','Tambaram','Porur',
                  'Adyar','Guindy','Sholinganallur','Chromepet','Mylapore'],
    'Pune'     : ['Koregaon Park','Wakad','Hadapsar','Kothrud','Hinjawadi',
                  'Viman Nagar','Baner','Kharadi','Katraj','Shivajinagar'],
}

VEHICLE_CATEGORY = {
    'Bike': '2-Wheeler', 'Auto': '3-Wheeler',
    'Mini': '4-Wheeler', 'Sedan': '4-Wheeler', 'SUV': '4-Wheeler', 'Prime': '4-Wheeler'
}

SEASONS = {
    1:'Winter',2:'Winter',3:'Summer',4:'Summer',5:'Summer',
    6:'Monsoon',7:'Monsoon',8:'Monsoon',9:'Monsoon',
    10:'Post-Monsoon',11:'Post-Monsoon',12:'Winter'
}

DEFAULTS = {
    'hour_of_day': 14, 'day_of_week': 2, 'month': 6,
    'is_weekend': 0, 'is_peak_hour': 0, 'is_night': 0,
    'city': 'Bangalore', 'pickup_zone': 'Koramangala', 'dropoff_zone': 'Indiranagar',
    'vehicle_type': 'Mini',
    'distance_km': 8.0, 'estimated_distance_km': 7.5,
    'trip_duration_min': 30.0, 'estimated_duration_min': 28.0,
    'waiting_time_min': 4.0,
    'weather_condition': 'Clear', 'temperature_celsius': 28.0,
    'rainfall_mm': 0.0, 'visibility_km': 14.0,
    'traffic_level': 'Medium', 'traffic_index': 5.0,
    'demand_level': 'Normal', 'demand_index': 4.5, 'supply_ratio': 1.0,
    'surge_multiplier': 1.0,
    'payment_method': 'UPI',
    'driver_rating': 4.5, 'passenger_rating': 4.8, 'driver_experience_yr': 3.0,
}

app = Flask(__name__)


def compute_derived(d):
    """Compute the 8 engineered features from raw inputs."""
    dist     = float(d.get('distance_km', 8))
    est_dist = float(d.get('estimated_distance_km', dist))
    dur      = float(d.get('trip_duration_min', 30))
    est_dur  = float(d.get('estimated_duration_min', dur))
    demand   = float(d.get('demand_index', 4.5))
    supply   = float(d.get('supply_ratio', 1.0))
    peak     = int(float(d.get('is_peak_hour', 0)))
    night    = int(float(d.get('is_night', 0)))
    weather  = d.get('weather_condition', 'Clear')
    hour     = int(float(d.get('hour_of_day', 14)))

    dist_diff_pct     = round((dist - est_dist) / (est_dist + 0.01), 4)
    duration_diff_pct = round((dur - est_dur) / (est_dur + 0.01), 4)
    effective_speed   = round(dist / (dur / 60 + 0.001), 2)
    demand_supply_gap = round(demand / (supply + 0.01), 3)
    peak_rain_flag    = 1 if (peak == 1 and weather == 'Rainy') else 0
    night_demand_flag = 1 if (night == 1 and demand > 6) else 0
    km_per_min        = round(dist / (dur + 0.01), 4)

    if   hour <= 5:  hour_bucket = 0
    elif hour <= 11: hour_bucket = 1
    elif hour <= 16: hour_bucket = 2
    elif hour <= 21: hour_bucket = 3
    else:            hour_bucket = 4

    return {
        'dist_diff_pct'     : dist_diff_pct,
        'duration_diff_pct' : duration_diff_pct,
        'effective_speed_kmh': effective_speed,
        'demand_supply_gap' : demand_supply_gap,
        'peak_rain_flag'    : peak_rain_flag,
        'night_demand_flag' : night_demand_flag,
        'km_per_min'        : km_per_min,
        'hour_bucket'       : hour_bucket,
    }


def build_feature_vector(d):
    """Build complete 39-feature DataFrame for model inference."""
    # Derive season from month
    month  = int(float(d.get('month', 6)))
    season = SEASONS.get(month, 'Summer')

    # Vehicle category from vehicle type
    vtype    = d.get('vehicle_type', 'Mini')
    vcat     = VEHICLE_CATEGORY.get(vtype, '4-Wheeler')

    # Derived features
    derived = compute_derived(d)

    raw = {
        'hour_of_day'          : float(d.get('hour_of_day', 14)),
        'day_of_week'          : float(d.get('day_of_week', 2)),
        'month'                : float(month),
        'season'               : season,
        'is_weekend'           : float(d.get('is_weekend', 0)),
        'is_peak_hour'         : float(d.get('is_peak_hour', 0)),
        'is_night'             : float(d.get('is_night', 0)),
        'city'                 : d.get('city', 'Bangalore'),
        'pickup_zone'          : d.get('pickup_zone', 'Koramangala'),
        'dropoff_zone'         : d.get('dropoff_zone', 'Indiranagar'),
        'vehicle_type'         : vtype,
        'vehicle_category'     : vcat,
        'distance_km'          : float(d.get('distance_km', 8)),
        'estimated_distance_km': float(d.get('estimated_distance_km', 7.5)),
        'trip_duration_min'    : float(d.get('trip_duration_min', 30)),
        'estimated_duration_min': float(d.get('estimated_duration_min', 28)),
        'waiting_time_min'     : float(d.get('waiting_time_min', 4)),
        'weather_condition'    : d.get('weather_condition', 'Clear'),
        'temperature_celsius'  : float(d.get('temperature_celsius', 28)),
        'rainfall_mm'          : float(d.get('rainfall_mm', 0)),
        'visibility_km'        : float(d.get('visibility_km', 14)),
        'traffic_level'        : d.get('traffic_level', 'Medium'),
        'traffic_index'        : float(d.get('traffic_index', 5.0)),
        'demand_level'         : d.get('demand_level', 'Normal'),
        'demand_index'         : float(d.get('demand_index', 4.5)),
        'supply_ratio'         : float(d.get('supply_ratio', 1.0)),
        'surge_multiplier'     : float(d.get('surge_multiplier', 1.0)),
        'payment_method'       : d.get('payment_method', 'UPI'),
        'driver_rating'        : float(d.get('driver_rating', 4.5)),
        'passenger_rating'     : float(d.get('passenger_rating', 4.8)),
        'driver_experience_yr' : float(d.get('driver_experience_yr', 3.0)),
        **derived,
    }

    # Encode categorical columns
    row = {}
    for col in feature_cols:
        val = raw.get(col, 0)
        if col in encoders:
            try:    row[col] = float(encoders[col].transform([str(val)])[0])
            except: row[col] = 0.0
        else:
            try:    row[col] = float(val)
            except: row[col] = 0.0

    return pd.DataFrame([row])[feature_cols]


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html',
                           defaults=DEFAULTS,
                           city_zones=CITY_ZONES,
                           vehicle_categories=list(VEHICLE_CATEGORY.keys()),
                           weather_options=['Clear','Cloudy','Rainy','Foggy'],
                           traffic_options=['Low','Medium','High','Very High'],
                           demand_options=['Low','Normal','High','Very High'],
                           payment_options=['UPI','Cash','Wallet','Card'],
                           cities=list(CITY_ZONES.keys()),
                           model_name=model_name,
                           model_r2=model_r2,
                           model_mae=model_mae,
                           model_mape=model_mape)


@app.route('/predict', methods=['POST'])
def predict():
    X_input = build_feature_vector(request.form)
    fare    = round(float(model.predict(X_input)[0]), 2)
    # Round to nearest 5
    fare_rounded = round(fare / 5) * 5

    # Fare breakdown estimate
    dist     = float(request.form.get('distance_km', 8))
    duration = float(request.form.get('trip_duration_min', 30))
    surge    = float(request.form.get('surge_multiplier', 1.0))
    vtype    = request.form.get('vehicle_type', 'Mini')

    return render_template('result.html',
                           fare=fare_rounded,
                           fare_exact=fare,
                           vehicle_type=vtype,
                           distance=dist,
                           duration=duration,
                           surge=surge,
                           model_name=model_name,
                           model_r2=model_r2,
                           model_mae=model_mae,
                           input_data=dict(request.form))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API endpoint."""
    data   = request.get_json(force=True) or {}
    X_input = build_feature_vector(data)
    fare   = round(float(model.predict(X_input)[0]), 2)
    return jsonify({
        'predicted_fare_inr' : round(fare / 5) * 5,
        'predicted_fare_exact': fare,
        'vehicle_type'       : data.get('vehicle_type', 'Mini'),
        'distance_km'        : data.get('distance_km', 8),
        'surge_multiplier'   : data.get('surge_multiplier', 1.0),
        'model'              : model_name,
        'model_r2'           : model_r2,
    })


@app.route('/zones/<city>')
def get_zones(city):
    """Return zones for a city (AJAX)."""
    return jsonify(CITY_ZONES.get(city, []))


if __name__ == '__main__':
    print("\n" + "="*55)
    print("  Ride Fare Prediction - Flask App")
    print(f"  Model : {model_name}  |  R2={model_r2}  MAE=Rs.{model_mae}")
    print("  Open  : http://127.0.0.1:5000")
    print("="*55 + "\n")
    app.run(debug=True, port=5000)
