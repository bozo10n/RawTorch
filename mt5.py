import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import pytz

def connect_to_mt5():
    if not mt5.initialize():
        print("MT5 initialization failed")
        mt5.shutdown()
        return False
    return True

def categorize_price_change(change):
    if change >= 0:
        if change <= 0.5:
            return "0-0.5% increase"
        elif change <= 1:
            return "0.51-1% increase"
        elif change <= 2:
            return "1-2% increase"
        else:
            return "2.1%+ increase"
    else:
        change = abs(change)
        if change <= 0.5:
            return "0-0.5% decrease"
        elif change <= 1:
            return "0.51-1% decrease"
        elif change <= 2:
            return "1-2% decrease"
        else:
            return "2.1%+ decrease"

def calculate_rsi(df, periods=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_session_data(symbol, start_time, end_time):
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M15, start_time, end_time)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['RSI'] = calculate_rsi(df)
    return df

def fill_blank_rsi(df):
    df['RSI'] = df['RSI'].fillna(50)  # Fill NaN values with 50
    return df

def analyze_sessions(symbol, start_date, end_date):
    sessions = {
        'Tokyo': ('21:00', '06:00'),
        'London': ('08:00', '16:00'),
        'NYSE': ('13:30', '20:00')
    }
    
    results = []
    current_date = start_date
    
    while current_date <= end_date:
        for session, (start, end) in sessions.items():
            session_start = datetime.combine(current_date, datetime.strptime(start, '%H:%M').time())
            session_end = datetime.combine(current_date, datetime.strptime(end, '%H:%M').time())
            
            if session_end < session_start:
                session_end += timedelta(days=1)
            
            df = get_session_data(symbol, session_start, session_end)
            df = fill_blank_rsi(df)
            
            if df.empty:
                continue
            
            session_open = df.iloc[0]['open']
            session_close = df.iloc[-1]['close']
            
            price_change = ((session_close - session_open) / session_open) * 100
            category = categorize_price_change(price_change)
            
            next_session = list(sessions.keys())[(list(sessions.keys()).index(session) + 1) % len(sessions)]
            next_start = datetime.combine(session_end.date(), datetime.strptime(sessions[next_session][0], '%H:%M').time())
            next_end = datetime.combine(session_end.date(), datetime.strptime(sessions[next_session][1], '%H:%M').time())
            
            if next_end < next_start:
                next_end += timedelta(days=1)
            
            next_df = get_session_data(symbol, next_start, next_end)
            next_df = fill_blank_rsi(next_df)
            
            if not next_df.empty:
                next_open = next_df.iloc[0]['open']
                next_close = next_df.iloc[-1]['close']
                next_high = next_df['high'].max()
                next_low = next_df['low'].min()
                
                next_change = ((next_close - next_open) / next_open) * 100
                next_high_change = ((next_high - next_open) / next_open) * 100
                next_low_change = ((next_low - next_open) / next_open) * 100
            else:
                next_change = price_change
                next_high_change = None
                next_low_change = None
            
            results.append({
                'date': session_start.date(),
                'symbol': symbol,
                'session': session,
                'session_change': price_change,
                'session_change_category': category,
                'RSI_first': df['RSI'].iloc[0],
                'RSI_last': df['RSI'].iloc[-1],
                'next_session': next_session,
                'next_session_change': next_change,
                'next_session_high_change': next_high_change,
                'next_session_low_change': next_low_change
            })
        
        current_date += timedelta(days=1)
    
    return pd.DataFrame(results)

def main():
    if not connect_to_mt5():
        return

    symbols = ["EURUSD", "GBPUSD", "USDJPY"]
    end_date = datetime.now(pytz.utc).date() - timedelta(days=1)
    start_date = end_date - timedelta(days=365*2)

    all_results = []
    for symbol in symbols:
        print(f"Analyzing {symbol}...")
        results = analyze_sessions(symbol, start_date, end_date)
        all_results.append(results)

    combined_results = pd.concat(all_results)
    
    percentage_columns = ['session_change', 'next_session_change', 'next_session_high_change', 'next_session_low_change']
    combined_results[percentage_columns] = combined_results[percentage_columns].round(2)
    
    combined_results.to_csv("market_session_analysis_results_2years.csv", index=False)
    print("Analysis complete. Results saved to 'market_session_analysis_results_2years.csv'")

    print("\nData Statistics:")
    print(combined_results[percentage_columns + ['RSI_first', 'RSI_last']].describe())

    mt5.shutdown()

if __name__ == "__main__":
    main()