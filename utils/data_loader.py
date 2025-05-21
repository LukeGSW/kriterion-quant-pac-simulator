# simulatore_pac/utils/data_loader.py

import pandas as pd
import yfinance as yf
from datetime import datetime

def load_historical_data_yf(ticker: str, start_date: str, end_date: str = None) -> pd.DataFrame:
    """
    Carica i dati storici (prezzi EOD e dividendi) per un ticker da Yahoo Finance.
    L'indice Datetime restituito è reso timezone-naive (UTC rimosso).

    Args:
        ticker (str): Il simbolo del ticker (es. 'AAPL', 'CSPX.AS', 'VWCE.DE').
        start_date (str): Data di inizio in formato 'YYYY-MM-DD'.
        end_date (str, optional): Data di fine in formato 'YYYY-MM-DD'.
                                   Se None, scarica fino alla data più recente. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame con 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividends'.
                      Indicizzato per data (timezone-naive).
                      Restituisce un DataFrame vuoto in caso di errore o assenza di dati.
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    print(f"Richiesta dati da Yahoo Finance per {ticker} da {start_date} a {end_date}...")
    try:
        stock_ticker = yf.Ticker(ticker)
        history_df = stock_ticker.history(start=start_date, end=end_date, auto_adjust=False)

        if history_df.empty:
            print(f"Nessun dato storico trovato per {ticker} da Yahoo Finance nel periodo specificato.")
            return pd.DataFrame()
        print(f"Dati per {ticker} scaricati con successo da Yahoo Finance.")

        # ***** MODIFICA CHIAVE: Rendi l'indice timezone-naive *****
        if history_df.index.tz is not None:
            print(f"Conversione dell'indice da timezone-aware ({history_df.index.tz}) a timezone-naive.")
            history_df.index = history_df.index.tz_localize(None)
        # ***********************************************************

        required_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividends']
        final_df = pd.DataFrame(index=history_df.index) # Ora l'indice è tz-naive

        for col in required_cols:
            if col in history_df.columns:
                final_df[col] = history_df[col]
            elif col == 'Dividends':
                final_df[col] = 0.0
        
        if 'Adj Close' not in final_df.columns or final_df['Adj Close'].isnull().all():
            if 'Close' in final_df.columns:
                print("Colonna 'Adj Close' mancante o vuota, utilizzando 'Close' come fallback per 'Adj Close'.")
                final_df['Adj Close'] = final_df['Close']
            else:
                print(f"Colonne 'Adj Close' e 'Close' mancanti o vuote per {ticker}. Impossibile procedere.")
                return pd.DataFrame()

        if 'Dividends' in final_df.columns:
            final_df.rename(columns={'Dividends': 'Dividend'}, inplace=True)
        else:
             final_df['Dividend'] = 0.0
            
        if final_df.empty:
            print(f"Nessun dato valido rimasto per {ticker} dopo la pulizia iniziale.")
            return pd.DataFrame()
        return final_df
    except Exception as e:
        print(f"Errore durante il recupero dei dati da Yahoo Finance per {ticker}: {e}")
        return pd.DataFrame()
