# simulatore_pac/main.py (Versione di Test per data_loader)
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

# Importazioni dai tuoi moduli utils
try:
    from utils.data_loader import load_historical_data_yf
    # Commenta le altre importazioni se causano problemi durante questo test specifico
    # from utils.pac_engine import run_pac_simulation
    # from utils.benchmark_engine import run_lump_sum_simulation
    # from utils.performance import ( ... )
    IMPORT_SUCCESS = True
except ImportError as import_err:
    IMPORT_SUCCESS = False
    IMPORT_ERROR_MESSAGE = str(import_err)


st.set_page_config(page_title="Test Data Loader", layout="wide")
st.title("🧪 Test Caricamento Dati (data_loader)")

if not IMPORT_SUCCESS:
    st.error(f"Errore critico durante l'importazione dei moduli utils: {IMPORT_ERROR_MESSAGE}")
    st.stop()

st.sidebar.header("Test Data Loader")
ticker_to_test = st.sidebar.text_input("Ticker da Testare", "AAPL")
start_date_test_str = st.sidebar.text_input("Data Inizio (YYYY-MM-DD)", "2019-01-01")
end_date_test_str = st.sidebar.text_input("Data Fine (YYYY-MM-DD)", datetime.today().strftime('%Y-%m-%d'))

if st.sidebar.button("Esegui Test Caricamento Dati"):
    st.write(f"--- Tentativo di caricare dati per {ticker_to_test} da {start_date_test_str} a {end_date_test_str} ---")
    
    # Aggiungi un blocco try-except attorno alla chiamata a load_historical_data_yf
    # per catturare eccezioni direttamente qui.
    data_df = pd.DataFrame() # Inizializza
    try:
        data_df = load_historical_data_yf(ticker_to_test, start_date_test_str, end_date_test_str)
    except Exception as e_load:
        st.error(f"Eccezione DENTRO load_historical_data_yf: {e_load}")
        import traceback
        st.text(traceback.format_exc())


    if not data_df.empty:
        st.success(f"Dati caricati per {ticker_to_test}!")
        st.write(f"Prime 5 righe:")
        st.dataframe(data_df.head())
        st.write(f"Ultime 5 righe:")
        st.dataframe(data_df.tail())
        st.write(f"Info sul DataFrame:")
        # Per visualizzare info() in Streamlit, dobbiamo catturare l'output
        from io import StringIO
        buffer = StringIO()
        data_df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
    else:
        st.error(f"Nessun dato restituito per {ticker_to_test} per l'intervallo specificato.")
        st.write("Controlla i log di Streamlit Cloud per eventuali messaggi da `utils/data_loader.py` (se hai lasciato i `print` di debug lì).")
