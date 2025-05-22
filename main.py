# simulatore_pac/main.py (Versione di Test per data_loader)
import streamlit as st
import pandas as pd
from datetime import datetime 
# Rimuovi le altre importazioni utils se non servono per questo test specifico
try:
    from utils.data_loader import load_historical_data_yf
    IMPORT_SUCCESS = True
except ImportError as import_err:
    IMPORT_SUCCESS = False
    IMPORT_ERROR_MESSAGE = str(import_err)


st.set_page_config(page_title="Test Data Loader V2", layout="wide")
st.title("🧪 Test Caricamento Dati (data_loader) V2")

if not IMPORT_SUCCESS:
    st.error(f"Errore critico durante l'importazione di data_loader: {IMPORT_ERROR_MESSAGE}")
    st.stop()

st.sidebar.header("Test Data Loader")
ticker_to_test = st.sidebar.text_input("Ticker da Testare", "AAPL")
start_date_test_str = st.sidebar.text_input("Data Inizio (YYYY-MM-DD)", "2019-01-01")
end_date_test_str = st.sidebar.text_input("Data Fine (YYYY-MM-DD)", datetime.today().strftime('%Y-%m-%d'))

if st.sidebar.button("Esegui Test Caricamento Dati"):
    st.write(f"--- Tentativo di caricare dati per '{ticker_to_test}' da {start_date_test_str} a {end_date_test_str} ---")
    
    data_df = pd.DataFrame()
    try:
        data_df = load_historical_data_yf(str(ticker_to_test).strip().upper(), start_date_test_str, end_date_test_str)
    except Exception as e_load:
        st.error(f"Eccezione DURANTE la chiamata a load_historical_data_yf da main.py: {e_load}")
        import traceback
        st.text(traceback.format_exc())

    if not data_df.empty:
        st.success(f"Dati caricati per {ticker_to_test}!")
        st.write(f"Recuperate {len(data_df)} righe.")
        st.write(f"Prime 5 righe:")
        st.dataframe(data_df.head())
        st.write(f"Ultime 5 righe:")
        st.dataframe(data_df.tail())
        from io import StringIO
        buffer = StringIO()
        data_df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
    else:
        st.error(f"Nessun dato restituito (DataFrame vuoto) per '{ticker_to_test}' per l'intervallo specificato.")
        st.info("Controlla i log di Streamlit Cloud per messaggi `INFO (data_loader):`, `ATTENZIONE (data_loader):` o `ERRORE (data_loader):` dal file `utils/data_loader.py`.")
