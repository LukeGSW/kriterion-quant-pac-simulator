# utils/data_loader.py
import pandas as pd
import yfinance as yf
from datetime import datetime
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "data_cache")

def load_historical_data_yf(
    ticker: str, 
    start_date: str, 
    end_date: str = None, 
    use_cache_for_preset: bool = False,  # <-- PARAMETRO NECESSARIO
    preset_ticker_file: str = None       # <-- PARAMETRO NECESSARIO
) -> pd.DataFrame:
    """
    Carica dati storici, con opzione per usare un file CSV dalla cache per i preset.
    """
    if use_cache_for_preset and preset_ticker_file:
        cache_file_path = os.path.join(CACHE_DIR, preset_ticker_file)
        # ... resto della logica per caricare da cache ...
        # (come fornito nel messaggio precedente dove abbiamo introdotto la cache)
        print(f"INFO (data_loader): Tentativo di caricare dati dalla cache: {cache_file_path}")
        if os.path.exists(cache_file_path):
            try:
                df_cache = pd.read_csv(cache_file_path)
                if 'Date' in df_cache.columns:
                    df_cache['Date'] = pd.to_datetime(df_cache['Date'])
                    df_cache.set_index('Date', inplace=True)
                # Assicurati che le colonne CSV siano corrette (es. 'Dividend')
                if 'Dividends' in df_cache.columns and 'Dividend' not in df_cache.columns:
                     df_cache.rename(columns={'Dividends':'Dividend'}, inplace=True)
                if 'Adj Close' not in df_cache.columns and 'Close' in df_cache.columns:
                    df_cache['Adj Close'] = df_cache['Close']


                print(f"INFO (data_loader): Dati per {ticker} caricati con successo dalla cache.")
                # Filtra per le date richieste, anche se il file è per un preset
                # Questo è importante se start_date/end_date della simulazione sono DIVERSE dal range del file cache
                # Tuttavia, per il preset, le date di simulazione dovrebbero corrispondere al contenuto del file cache.
                # Per sicurezza, filtriamo.
                df_cache_filtered = df_cache[(df_cache.index >= pd.to_datetime(start_date)) & (df_cache.index <= pd.to_datetime(end_date))]
                if df_cache_filtered.empty:
                    print(f"ATTENZIONE (data_loader): Cache per {ticker} non contiene dati per range {start_date}-{end_date}")
                    # Fallback a yfinance se il range richiesto non è nella cache
                    return load_historical_data_yf(ticker, start_date, end_date, use_cache_for_preset=False)

                return df_cache_filtered.sort_index()
            except Exception as e_cache:
                print(f"ERRORE (data_loader): Impossibile leggere/processare file cache {cache_file_path}: {e_cache}. Fallback a yfinance.")
                # Fallback a yfinance se il caricamento da cache fallisce
                return load_historical_data_yf(ticker, start_date, end_date, use_cache_for_preset=False)
        else:
            print(f"ATTENZIONE (data_loader): File cache {cache_file_path} non trovato. Tentativo con yfinance.")
            # Fallback a yfinance se il file cache non esiste
            return load_historical_data_yf(ticker, start_date, end_date, use_cache_for_preset=False)

    # Logica yfinance esistente (se non si usa la cache o se il fallback è attivato)
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    
    print(f"INFO (data_loader): Tentativo yfinance per {ticker} da {start_date} a {end_date}")
    history_df = pd.DataFrame() 
    try:
        stock_ticker = yf.Ticker(ticker)
        history_df = stock_ticker.history(start=start_date, end=end_date, auto_adjust=False) # Rimosso progress=False
        
        if history_df.empty:
            print(f"ATTENZIONE (data_loader): yfinance HA RESTITUITO DataFrame VUOTO per {ticker} ({start_date} to {end_date}).")
            return pd.DataFrame()
            
        if history_df.index.tz is not None:
            history_df.index = history_df.index.tz_localize(None)
        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividends']
        final_df = pd.DataFrame(index=history_df.index)
        for col in required_cols:
            if col in history_df.columns: final_df[col] = history_df[col]
            elif col == 'Dividends': final_df[col] = 0.0
        
        if 'Adj Close' not in final_df.columns or final_df['Adj Close'].isnull().all():
            if 'Close' in final_df.columns and not final_df['Close'].isnull().all():
                final_df['Adj Close'] = final_df['Close']
            else:
                return pd.DataFrame()
        
        if 'Dividends' in final_df.columns:
            final_df.rename(columns={'Dividends': 'Dividend'}, inplace=True)
        elif 'Dividend' not in final_df.columns: 
            final_df['Dividend'] = 0.0
            
        if 'Adj Close' in final_df.columns:
            final_df.dropna(subset=['Adj Close'], inplace=True)
            final_df = final_df[final_df['Adj Close'] > 1e-6] 

        if final_df.empty:
            return pd.DataFrame()
            
        return final_df.sort_index()

    except Exception as e:
        print(f"ERRORE CRITICO in load_historical_data_yf (yfinance) per {ticker}: {e}")
        return pd.DataFrame()
