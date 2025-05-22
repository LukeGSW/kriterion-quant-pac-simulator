# simulatore_pac/utils/data_loader.py

import pandas as pd
import yfinance as yf
from datetime import datetime 

# Non serve timedelta qui, era in una versione precedente di main.py per calcoli di date fetch
# from datetime import timedelta 

def load_historical_data_yf(ticker: str, start_date: str, end_date: str = None) -> pd.DataFrame:
    """
    Carica i dati storici (prezzi EOD e dividendi) per un ticker da Yahoo Finance.
    L'indice Datetime restituito è reso timezone-naive.
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    # Stampa di debug per le date effettive usate per yfinance
    # print(f"DEBUG (data_loader): Richiesta yf per {ticker} da {start_date} a {end_date}")

    try:
        stock_ticker = yf.Ticker(ticker)
        # Scarica dati. auto_adjust=False per avere 'Adj Close' solo per split e colonna 'Dividends' separata.
        history_df = stock_ticker.history(start=start_date, end=end_date, auto_adjust=False, progress=False)

        if history_df.empty:
            # print(f"ATTENZIONE (data_loader): Nessun dato storico per {ticker} da YF tra {start_date} e {end_date}.")
            return pd.DataFrame()
        # print(f"INFO (data_loader): Dati per {ticker} scaricati. Righe: {len(history_df)}")

        # Rendi l'indice timezone-naive
        if history_df.index.tz is not None:
            # print(f"INFO (data_loader): Indice per {ticker} è tz-aware ({history_df.index.tz}), conversione a naive.")
            history_df.index = history_df.index.tz_localize(None)
        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividends']
        final_df = pd.DataFrame(index=history_df.index)

        for col in required_cols:
            if col in history_df.columns:
                final_df[col] = history_df[col]
            elif col == 'Dividends': # Se 'Dividends' non c'è (nessun dividendo), crea colonna di zeri
                final_df[col] = 0.0
            # Non ci preoccupiamo di altre colonne mancanti, yfinance di solito le fornisce

        if 'Adj Close' not in final_df.columns or final_df['Adj Close'].isnull().all():
            if 'Close' in final_df.columns and not final_df['Close'].isnull().all():
                # print(f"ATTENZIONE (data_loader): 'Adj Close' mancante per {ticker}, uso 'Close'.")
                final_df['Adj Close'] = final_df['Close']
            else:
                # print(f"ERRORE (data_loader): 'Adj Close' e 'Close' mancanti per {ticker}.")
                return pd.DataFrame()
        
        # Rinomina 'Dividends' in 'Dividend' per coerenza interna
        if 'Dividends' in final_df.columns:
            final_df.rename(columns={'Dividends': 'Dividend'}, inplace=True)
        elif 'Dividend' not in final_df.columns: # Assicura che esista anche se vuota
            final_df['Dividend'] = 0.0
            
        # Rimuovi righe dove il prezzo di chiusura (o Adj Close) è NaN o zero, 
        # potrebbero invalidare calcoli successivi.
        # È importante farlo dopo aver gestito i dividendi, perché potremmo avere un dividendo
        # in un giorno in cui il prezzo è mancante per qualche motivo (raro).
        final_df.dropna(subset=['Adj Close'], inplace=True)
        final_df = final_df[final_df['Adj Close'] > 1e-6] # Rimuovi prezzi zero o trascurabili

        if final_df.empty:
            # print(f"ATTENZIONE (data_loader): DataFrame vuoto per {ticker} dopo pulizia prezzi NaN/zero.")
            return pd.DataFrame()
            
        return final_df.sort_index() # Assicura ordinamento per data

    except Exception as e:
        print(f"ERRORE CRITICO in load_historical_data_yf per {ticker} tra {start_date}-{end_date}: {e}")
        import traceback
        # traceback.print_exc() # Utile per debug locale, ma potrebbe essere troppo verboso per Streamlit Cloud
        return pd.DataFrame()
