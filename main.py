# main.py (versione di debug super semplificata)
import streamlit as st
import pandas as pd # Importa anche se non lo usi subito, per testare l'import
import yfinance as yf # Importa anche se non lo usi subito
from datetime import datetime, date # Aggiunto date

st.set_page_config(page_title="Test Debug PAC", layout="wide") # Deve essere la prima chiamata st

st.write("--- Script Iniziato ---")

try:
    st.write("--- Importazioni Utilità Tentativo ---")
    # Prova a importare le tue utilità
    from utils.data_loader import load_historical_data_yf
    from utils.pac_engine import run_basic_pac_simulation
    from utils.performance import get_total_capital_invested 
    st.write("--- Importazioni Utilità Riuscite ---")

    st.title("📘 Simulatore PAC - Test di Debug")
    st.write("--- Titolo Scritto ---")

    # Test Sidebar Semplice
    st.sidebar.header("Test Input")
    test_input = st.sidebar.text_input("Input di Test", "Ciao")
    st.write(f"--- Valore Input di Test: {test_input} ---")
    
    if st.sidebar.button("Test Bottone"):
        st.write("--- Bottone Premuto ---")
        st.balloons()
    
    st.write("--- Script Terminato Normalmente (parte visibile) ---")

except Exception as e:
    st.error(f"ERRORE CRITICO NELLO SCRIPT PRINCIPALE: {e}")
    import traceback
    st.text(traceback.format_exc())
