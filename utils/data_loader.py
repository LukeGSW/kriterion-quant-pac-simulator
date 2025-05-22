# simulatore_pac/utils/data_loader.py

import pandas as pd
import yfinance as yf
from datetime import datetime

def load_historical_data_yf(ticker: str, start_date: str, end_date: str = None) -> pd.DataFrame:
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    print(f"INFO (data_loader): Tentativo di scaricare dati per {ticker} da {start_date} a {end_date}")
    
    history_df = pd.DataFrame() 
    try:
        stock_ticker = yf.Ticker(ticker)
        print(f"DEBUG (data_loader): Oggetto Ticker creato per {ticker}. Chiamata a .history(start='{start_date}', end='{end_date}')...")
        
        try:
            # RIMOSSO progress=False
            history_df = stock_ticker.history(start=start_date, end=end_date, auto_adjust=False) 
        except Exception as e_history:
            print(f"ERRORE (data_loader): Eccezione DURANTE stock_ticker.history() per {ticker}: {e_history}")
            import traceback
            traceback.print_exc() 
            return pd.DataFrame()

        if history_df.empty:
            print(f"ATTENZIONE (data_loader): yfinance HA RESTITUITO DataFrame VUOTO per {ticker} ({start_date} to {end_date}).")
            try:
                info = stock_ticker.info
                if not info or 'regularMarketPrice' not in info : 
                    print(f"DEBUG (data_loader): stock_ticker.info per {ticker} è vuota o incompleta.")
                else:
                    print(f"DEBUG (data_loader): stock_ticker.info per {ticker} recuperata, ma history è vuota. Info: { {k: info[k] for k in ['symbol', 'shortName', 'exchange', 'marketState'] if k in info} }")
            except Exception as e_info:
                print(f"DEBUG (data_loader): Eccezione durante il recupero di stock_ticker.info per {ticker}: {e_info}")
            return pd.DataFrame()
            
        print(f"INFO (data_loader): Dati per {ticker} scaricati. Righe: {len(history_df)}. Date da {history_df.index.min().date()} a {history_df.index.max().date()}")
        
        if history_df.index.tz is not None:
            history_df.index = history_df.index.tz_localize(None)
        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividends']
        final_df = pd.DataFrame(index=history_df.index)
        for col in required_cols:
            if col in history_df.columns:
                final_df[col] = history_df[col]
            elif col == 'Dividends':
                final_df[col] = 0.0
        
        if 'Adj Close' not in final_df.columns or final_df['Adj Close'].isnull().all():
            if 'Close' in final_df.columns and not final_df['Close'].isnull().all():
                final_df['Adj Close'] = final_df['Close']
            else:
                print(f"ERRORE (data_loader): 'Adj Close' e 'Close' mancanti o tutti NaN per {ticker} dopo download.")
                return pd.DataFrame()
        
        if 'Dividends' in final_df.columns:
            final_df.rename(columns={'Dividends': 'Dividend'}, inplace=True)
        elif 'Dividend' not in final_df.columns: 
            final_df['Dividend'] = 0.0
            
        if 'Adj Close' in final_df.columns:
            final_df.dropna(subset=['Adj Close'], inplace=True)
            final_df = final_df[final_df['Adj Close'] > 1e-6] 

        if final_df.empty:
            print(f"ATTENZIONE (data_loader): DataFrame vuoto per {ticker} dopo pulizia.")
            return pd.DataFrame()
            
        return final_df.sort_index()

    except Exception as e:
        print(f"ERRORE CRITICO in load_historical_data_yf per {ticker} ({start_date}-{end_date}): {e}")
        import traceback
        print("--- TRACEBACK ECCEZIONE DATA_LOADER (generale) ---")
        traceback.print_exc()
        print("--- FINE TRACEBACK ECCEZIONE DATA_LOADER (generale) ---")
        return pd.DataFrame()
