# simulatore_pac/main.py (Parte 1 della Ristrutturazione)
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

# --- Importazioni Utils (le manterremo qui per ora) ---
try:
    from utils.data_loader import load_historical_data_yf
    from utils.pac_engine import run_pac_simulation
    from utils.benchmark_engine import run_lump_sum_simulation
    from utils.performance import (
        get_total_capital_invested, get_final_portfolio_value,
        calculate_total_return_percentage, calculate_cagr, get_duration_years,
        calculate_portfolio_returns, calculate_annualized_volatility,
        calculate_sharpe_ratio, calculate_max_drawdown, calculate_drawdown_series,
        generate_cash_flows_for_xirr, calculate_xirr_metric, calculate_annual_returns,
        calculate_rolling_volatility, calculate_rolling_sharpe_ratio, calculate_rolling_cagr
    )
    IMPORT_SUCCESS = True
except ImportError as import_err:
    IMPORT_SUCCESS = False
    IMPORT_ERROR_MESSAGE = str(import_err)

st.set_page_config(page_title="Simulatore PAC Ristrutturato", layout="wide")
st.title("📘 Simulatore PAC (Logica Date Ristrutturata)")
st.caption("Progetto Kriterion Quant")

if not IMPORT_SUCCESS:
    st.error(f"Errore critico import moduli utils: {IMPORT_ERROR_MESSAGE}")
    st.stop()

# --- Sidebar (come prima) ---
st.sidebar.header("Parametri Simulazione")
st.sidebar.subheader("Asset e Allocazioni")
tickers_input_str = st.sidebar.text_input("Tickers (virgola sep.)", "AAPL,GOOG,MSFT")
allocations_input_str = st.sidebar.text_input("Allocazioni % (virgola sep.)", "60,20,20")
st.sidebar.subheader("Parametri PAC")
monthly_investment_input = st.sidebar.number_input("Importo Mensile (€/$)", 10.0, value=200.0, step=10.0)
default_start_date_pac_sidebar = date(2020, 1, 1)
pac_start_date_input_ui = st.sidebar.date_input("Data Inizio Contributi PAC", default_start_date_pac_sidebar) # Rinominato per chiarezza
end_date_for_default_duration = default_start_date_pac_sidebar.replace(year=default_start_date_pac_sidebar.year + 3)
if end_date_for_default_duration > date.today(): end_date_for_default_duration = date.today()
default_duration_months = max(6, (end_date_for_default_duration.year - pac_start_date_input_ui.year) * 12 + \
                          (end_date_for_default_duration.month - pac_start_date_input_ui.month))
duration_months_contributions_input = st.sidebar.number_input("Durata Contributi PAC (mesi)", 6, value=default_duration_months, step=1) # Rinominato
reinvest_dividends_input = st.sidebar.checkbox("Reinvesti Dividendi?", True)
st.sidebar.subheader("Ribilanciamento PAC")
rebalance_active_input = st.sidebar.checkbox("Attiva Ribilanciamento?", False)
rebalance_frequency_input_str = None
if rebalance_active_input:
    rebalance_frequency_input_str = st.sidebar.selectbox("Frequenza Ribilanciamento", ["Annuale", "Semestrale", "Trimestrale"], 0)
st.sidebar.subheader("Metriche Avanzate")
risk_free_rate_input = st.sidebar.number_input("Tasso Risk-Free Ann. (%)", 0.0, value=1.0, step=0.1, format="%.2f")
rolling_window_months_input = st.sidebar.number_input("Finestra Rolling Metrics (mesi)", 6, value=36, step=1) # Ridotto step
run_simulation_button = st.sidebar.button("🚀 Avvia Simulazioni")

if run_simulation_button:
    st.write("--- DEBUG: Pulsante Avvia PREMUTO ---")
    # --- VALIDAZIONE INPUT (come prima) ---
    tickers_list = [t.strip().upper() for t in tickers_input_str.split(',') if t.strip()]
    error_in_input = False; allocations_list_norm = []; allocations_float_list_raw = []
    if not tickers_list: st.error("Errore: Minimo un ticker."); error_in_input = True
    if not error_in_input: # Continua la validazione solo se non ci sono stati errori precedenti
        try:
            allocations_float_list_raw = [float(a.strip()) for a in allocations_input_str.split(',') if a.strip()]
            if len(tickers_list)!=len(allocations_float_list_raw): st.error("Errore: N. ticker diverso da N. allocazioni."); error_in_input=True
            elif not np.isclose(sum(allocations_float_list_raw),100.0): st.error(f"Errore: Somma allocazioni ({sum(allocations_float_list_raw)}%) != 100%."); error_in_input=True
            else: allocations_list_norm = [a/100.0 for a in allocations_float_list_raw]
        except ValueError: st.error("Errore: Allocazioni non numeriche."); error_in_input=True
    if error_in_input: st.write("--- DEBUG: Errore negli input ---"); st.stop()
    st.write(f"--- DEBUG: Input OK. Tickers: {tickers_list} ---")

    # --- FASE A: DEFINIZIONE DATE CHIAVE E CARICAMENTO DATI ---
    pac_contribution_start_dt = pd.to_datetime(pac_start_date_input_ui)
    pac_contribution_start_str = pac_contribution_start_dt.strftime('%Y-%m-%d')
    pac_contribution_end_dt = pac_contribution_start_dt + pd.DateOffset(months=duration_months_contributions_input)
    
    data_download_start_str = (pac_contribution_start_dt - pd.Timedelta(days=365*1)).strftime('%Y-%m-%d') # 1 anno prima
    data_download_end_str = datetime.today().strftime('%Y-%m-%d') # Scarica dati fino ad oggi

    st.write(f"--- DEBUG: `pac_contribution_start_dt`: {pac_contribution_start_dt.date()} ---")
    st.write(f"--- DEBUG: `pac_contribution_end_dt`: {pac_contribution_end_dt.date()} ---")
    st.write(f"--- DEBUG: Download dati da: {data_download_start_str} a: {data_download_end_str} ---")

    historical_data_map = {}
    all_data_loaded_successfully = True
    # latest_data_date_for_all_tickers: ultima data per cui TUTTI i ticker hanno dati.
    # Inizializza con una data molto futura, poi prendi il minimo.
    latest_data_date_for_all_tickers = pd.Timestamp(datetime.today())

    for tkr in tickers_list:
        with st.spinner(f"Dati per {tkr}..."): 
            data = load_historical_data_yf(tkr, data_download_start_str, data_download_end_str)
        
        # Controllo se i dati coprono almeno il periodo di contribuzione del PAC
        if data.empty or data.index.min() > pac_contribution_start_dt or data.index.max() < (pac_contribution_end_dt - pd.Timedelta(days=1)):
            st.error(f"Dati storici insufficienti per {tkr} per coprire almeno il periodo di contribuzione PAC (fino a {pac_contribution_end_dt.date()}).")
            all_data_loaded_successfully = False; break
        historical_data_map[tkr] = data
        # Aggiorna latest_data_date_for_all_tickers con la data più RECENTE tra quelle minime
        if data.index.max() < latest_data_date_for_all_tickers:
             latest_data_date_for_all_tickers = data.index.max()
    
    if not all_data_loaded_successfully: st.write("--- DEBUG: Caricamento dati fallito ---"); st.stop()
    st.success("Dati storici caricati.")

    # Data finale per l'intera simulazione e per l'asse X dei grafici
    # Deve essere l'ultima data per cui abbiamo dati per TUTTI gli asset, ma non oltre oggi.
    simulation_actual_end_dt = min(latest_data_date_for_all_tickers, pd.Timestamp(datetime.today()))
    
    st.write(f"--- DEBUG: `latest_data_date_for_all_tickers` (dopo loop): {latest_data_date_for_all_tickers.date()} ---")
    st.write(f"--- DEBUG: `simulation_actual_end_dt` (fine simulazioni e grafici): {simulation_actual_end_dt.date()} ---")

    # --- PREPARAZIONE INDICE DATE BASE PER GRAFICI ---
    base_chart_date_index = pd.DatetimeIndex([])
    if pac_contribution_start_dt <= simulation_actual_end_dt:
        base_chart_date_index = pd.date_range(start=pac_contribution_start_dt, end=simulation_actual_end_dt, freq='B')
        base_chart_date_index.name = 'Date'
        if not base_chart_date_index.empty: 
             st.write(f"--- DEBUG: `base_chart_date_index` creato: da {base_chart_date_index.min().date()} a {base_chart_date_index.max().date()} ({len(base_chart_date_index)} punti) ---")
        else: st.write(f"--- DEBUG: `base_chart_date_index` è VUOTO. Controllare start/end dates. ---")
    else:
        st.write(f"--- DEBUG: Data inizio PAC ({pac_contribution_start_dt.date()}) > fine simulazione ({simulation_actual_end_dt.date()}). Indice base vuoto. ---")


    # --- DA QUI IN POI, PASSEREMO simulation_actual_end_dt AI MOTORI E USEREMO base_chart_date_index PER I GRAFICI ---
    # Per ora, fermiamoci qui per verificare questa prima parte della ristrutturazione.
    # Commenta il resto del codice (simulazioni, metriche, grafici) per ora.

    st.subheader("Prossimo Passo: Modificare pac_engine e benchmark_engine")
    st.info("Fase A della ristrutturazione completata. Controlla i messaggi di DEBUG qui sopra per la validità delle date.")
    st.info(f"Il simulatore tenterà di eseguire analisi fino al: {simulation_actual_end_dt.strftime('%Y-%m-%d')}")
    
    # Per ora, commentiamo il resto per testare solo la parte delle date e del caricamento
    # if error_in_input: st.stop() # Questo è già sopra
    # ... (tutta la logica di esecuzione simulazioni, metriche, grafici sarà qui sotto) ...

else: 
    st.info("Inserisci parametri e avvia simulazione.")
    st.write("--- DEBUG: Pagina iniziale ---")

st.sidebar.markdown("---"); st.sidebar.markdown("Kriterion Quant")
