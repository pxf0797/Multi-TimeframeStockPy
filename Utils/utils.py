import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def calculate_volatility(df, window=20):
    return df['Close'].pct_change().rolling(window=window).std() * np.sqrt(252)

def calculate_trend_strength(df):
    return (df['MA_5'] - df['MA_20']) / df['MA_20']

def calculate_accuracy(predictions, actual):
    return np.mean((np.sign(predictions) == np.sign(actual)).astype(int))

def identify_market_regime(df, max_clusters=5):
    features = ['Volatility', 'Trend_Strength', 'Volume']
    
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    
    df_copy = df.copy()
    df_clean = df_copy.dropna(subset=features)
    
    if len(df_clean) < max_clusters:
        print(f"Warning: Not enough data points ({len(df_clean)}) for clustering after removing NaN values.")
        df_copy['Regime'] = 0
        return df_copy
    
    X = df_clean[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    inertias = []
    for n_clusters in range(1, min(max_clusters, len(X_scaled)) + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    optimal_clusters = 1
    for i in range(1, len(inertias)):
        if inertias[i-1] == 0:  # Avoid division by zero
            continue
        if (inertias[i-1] - inertias[i]) / inertias[i-1] < 0.2:
            optimal_clusters = i
            break
    
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
    df_clean['Regime'] = kmeans.fit_predict(X_scaled)
    
    df_copy['Regime'] = df_clean['Regime']
    return df_copy

def adaptive_stop_loss(entry_price, atr, risk_total, k=2):
    return entry_price * (1 - k * atr * risk_total)

def position_sizing(account_balance, risk_per_trade, entry_price, stop_loss):
    return (account_balance * risk_per_trade) / abs(entry_price - stop_loss)

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate/252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calculate_maximum_drawdown(equity_curve):
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    return drawdown.min()

def log_trade(trade_info, log_file):
    with open(log_file, 'a') as f:
        f.write(f"{pd.Timestamp.now()}: {trade_info}\n")

def implement_circuit_breaker(price_change, threshold=0.1):
    return abs(price_change) > threshold

def visualize_attention_weights(attention_weights, timeframes):
    plt.figure(figsize=(10, 6))
    plt.imshow(attention_weights, cmap='hot', aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(timeframes)), timeframes)
    plt.yticks(range(len(timeframes)), timeframes)
    plt.title("Attention Weights Heatmap")
    plt.show()

def calculate_trend_consistency(featured_data):
    trend_signs = []
    for tf, df in featured_data.items():
        if 'TSI' not in df.columns:
            print(f"Warning: TSI not found in {tf} timeframe. Skipping.")
            continue
        tsi = df['TSI'].iloc[-1]
        trend_signs.append(np.sign(tsi))
    
    if not trend_signs:
        print("Warning: No valid TSI data found across timeframes.")
        return 0
    
    trend_cons = np.prod(trend_signs)
    return trend_cons

def async_data_processing(data_queue, processed_data_queue):
    while True:
        raw_data = data_queue.get()
        if raw_data is None:
            break
        processed_data = process_data(raw_data)  # Implement this function
        processed_data_queue.put(processed_data)

def shap_feature_importance(model, X):
    import shap
    explainer = shap.DeepExplainer(model, X)
    shap_values = explainer.shap_values(X)
    return pd.DataFrame(shap_values, columns=X.columns)

def plot_equity_curve(equity):
    plt.figure(figsize=(12, 6))
    plt.plot(equity.index, equity.values)
    plt.title('Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.grid(True)
    plt.show()

def calculate_calmar_ratio(returns, window=36):
    cagr = (1 + returns.mean()) ** 12 - 1
    max_drawdown = calculate_maximum_drawdown(returns.cumsum())
    return cagr / abs(max_drawdown)

def handle_nan_inf(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.interpolate(method='time', inplace=True, limit_direction='both')
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df
