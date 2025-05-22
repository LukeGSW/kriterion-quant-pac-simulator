# simulatore_pac/main.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

try:
    from utils.data_loader import load_historical_data_yf
    from utils.pac_engine import run_pac_simulation # Aspetta simulation_actual_end_date_dt
    from utils.benchmark_engine import run_lump_sum_simulation # Aspetta simulation_end_date estesa
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

st.set_page_config(page_title="Simulatore PAC Debug V2", layout="wide")
st.title("📘 Simulatore PAC con Debug Avanzato V2")
st.caption("Progetto Kriterion Quant")

if not IMPORT_SUCCESS:
    st.error(f"Errore critico import moduli utils: {IMPORT_ERROR_MESSAGE}")
    st.stop()

# --- Sidebar (come prima) ---
st.sidebar.header("Parametri Simulazione")
# ... (Tutti gli input della sidebar rimangono invariati) ...
tickers_input_str = st.sidebar.text_input("Tickers (virgola sep.)", "AAPL,GOOG,MSFT")
allocations_input_str = st.sidebar.text_input("Allocazioni % (virgola sep.)", "60,20,20")
monthly_investment_input = st.sidebar.number_input("Importo Mensile (€/$)", 10.0, value=200.0, step=10.0)
default_start_date_pac_sidebar = date(2020, 1, 1)
pac_start_date_input = st.sidebar.date_input("Data Inizio PAC", default_start_date_pac_sidebar)
end_date_for_default_duration = default_start_date_pac_sidebar.replace(year=default_start_date_pac_sidebar.year + 3)
if end_date_for_default_duration > date.today(): end_date_for_default_duration = date.today()
default_duration_months = max(6, (end_date_for_default_duration.year - pac_start_date_input.year) * 12 + \
                          (end_date_for_default_duration.month - pac_start_date_input.month))
duration_months_input = st.sidebar.number_input("Durata PAC (mesi)", 6, value=default_duration_months, step=1)
reinvest_dividends_input = st.sidebar.checkbox("Reinvesti Dividendi?", True)
rebalance_active_input = st.sidebar.checkbox("Attiva Ribilanciamento?", False)
rebalance_frequency_input_str = None
if rebalance_active_input: rebalance_frequency_input_str = st.sidebar.selectbox("Frequenza", ["Annuale", "Semestrale", "Trimestrale"], 0)
risk_free_rate_input = st.sidebar.number_input("Tasso Risk-Free Ann. (%)", 0.0, value=1.0, step=0.1, format="%.2f")
rolling_window_months_input = st.sidebar.number_input("Finestra Rolling Metrics (mesi)", 6, value=36, step=6)
run_simulation_button = st.sidebar.button("🚀 Avvia Simulazioni")


if run_simulation_button:
    st.write("--- DEBUG: Pulsante Avvia PREMUTO ---")
    # --- VALIDAZIONE INPUT (come prima) ---
    tickers_list = [t.strip().upper() for t in tickers_input_str.split(',') if t.strip()] # ... (resto validazione) ...
    error_in_input = False; allocations_list_norm = []; allocations_float_list_raw = []
    if not tickers_list: st.error("Errore: Minimo un ticker."); error_in_input = True
    if not error_in_input:
        try:
            allocations_float_list_raw = [float(a.strip()) for a in allocations_input_str.split(',') if a.strip()]
            if len(tickers_list)!=len(allocations_float_list_raw): st.error("Errore: N. ticker diverso da N. allocazioni."); error_in_input=True
            elif not np.isclose(sum(allocations_float_list_raw),100.0): st.error(f"Errore: Somma allocazioni ({sum(allocations_float_list_raw)}%) != 100%."); error_in_input=True
            else: allocations_list_norm = [a/100.0 for a in allocations_float_list_raw]
        except ValueError: st.error("Errore: Allocazioni non numeriche."); error_in_input=True
    if error_in_input: st.write("--- DEBUG: Errore negli input ---"); st.stop()
    st.write(f"--- DEBUG: Input OK. Tickers: {tickers_list} ---")

    # --- PREPARAZIONE DATE ---
    pac_start_date_dt = pd.to_datetime(pac_start_date_input)
    pac_start_date_str = pac_start_date_dt.strftime('%Y-%m-%d')
    actual_pac_contribution_end_date_dt = pac_start_date_dt + pd.DateOffset(months=duration_months_input) # Fine dei versamenti PAC
    
    data_fetch_start_date_str = (pac_start_date_dt - pd.Timedelta(days=365*1)).strftime('%Y-%m-%d')
    data_fetch_end_date_str = datetime.today().strftime('%Y-%m-%d') # Scarica dati fino ad oggi

    historical_data_map = {}; all_data_loaded_successfully = True; 
    latest_overall_data_date_ts = pd.Timestamp.min 
    for tkr in tickers_list:
        with st.spinner(f"Dati per {tkr}..."): data = load_historical_data_yf(tkr, data_fetch_start_date_str, data_fetch_end_date_str)
        # Il controllo sulla data finale dei dati deve considerare la fine della contribuzione PAC
        if data.empty or data.index.min()>pac_start_date_dt or data.index.max()<(actual_pac_contribution_end_date_dt-pd.Timedelta(days=1)):
            st.error(f"Dati storici insufficienti per {tkr} per coprire almeno il periodo di contribuzione PAC (fino a {actual_pac_contribution_end_date_dt.date()})."); all_data_loaded_successfully=False; break
        historical_data_map[tkr] = data
        if data.index.max() > latest_overall_data_date_ts:
             latest_overall_data_date_ts = data.index.max()
    if not all_data_loaded_successfully: st.write("--- DEBUG: Caricamento dati fallito ---"); st.stop()
    st.success("Dati storici OK.")
    
    # Data finale per l'intera simulazione e per l'asse X dei grafici
    simulation_and_chart_end_date = min(latest_overall_data_date_ts, pd.Timestamp(datetime.today()))
    st.write(f"--- DEBUG: `latest_overall_data_date_ts`: {latest_overall_data_date_ts.date()} ---")
    st.write(f"--- DEBUG: `actual_pac_contribution_end_date_dt`: {actual_pac_contribution_end_date_dt.date()} ---")
    st.write(f"--- DEBUG: `simulation_and_chart_end_date` (fine simulazioni e grafici): {simulation_and_chart_end_date.date()} ---")

    base_chart_date_index = pd.DatetimeIndex([])
    if pac_start_date_dt <= simulation_and_chart_end_date:
        base_chart_date_index = pd.date_range(start=pac_start_date_dt, end=simulation_and_chart_end_date, freq='B')
        base_chart_date_index.name = 'Date'
        if not base_chart_date_index.empty: st.write(f"--- DEBUG: `base_chart_date_index` creato: da {base_chart_date_index.min().date()} a {base_chart_date_index.max().date()} ---")
    else: st.write(f"--- DEBUG: Data inizio PAC ({pac_start_date_dt.date()}) > fine simulazione ({simulation_and_chart_end_date.date()}). Indice base vuoto. ---")
    
    # --- ESECUZIONE SIMULAZIONI ---
    pac_total_df, asset_details_history_df = pd.DataFrame(), pd.DataFrame()
    try:
        with st.spinner("Simulazione PAC..."): 
            pac_total_df, asset_details_history_df = run_pac_simulation(
                historical_data_map, tickers_list, allocations_list_norm, 
                monthly_investment_input, pac_start_date_str, 
                duration_months_input, # Questo è per la durata dei versamenti
                reinvest_dividends_input, rebalance_active_input, rebalance_frequency_input_str
                # `run_pac_simulation` userà le date in historical_data_map per determinare la sua fine effettiva
            )
        st.write("--- DEBUG: `run_pac_simulation` COMPLETATA ---")
    except Exception as e: st.error(f"Errore CRITICO PAC: {e}"); import traceback; st.text(traceback.format_exc()); st.stop()
    # ... (Stampe DEBUG per pac_total_df e asset_details_history_df come prima)

    lump_sum_df = pd.DataFrame()
    if not pac_total_df.empty and 'PortfolioValue' in pac_total_df.columns and len(pac_total_df)>=2:
        total_invested_by_pac = get_total_capital_invested(pac_total_df) # Questo è il capitale versato nel PAC
        if total_invested_by_pac > 0:
            with st.spinner("Simulazione LS..."): 
                # LS deve simulare per lo stesso periodo esteso del PAC
                lump_sum_df = run_lump_sum_simulation(
                    historical_data_map, tickers_list, allocations_list_norm, 
                    total_invested_by_pac, # Capitale per LS
                    pac_start_date_dt,  # Data investimento LS
                    pac_start_date_dt, # Inizio tracciamento LS (allineato a PAC)
                    simulation_and_chart_end_date, # Fine tracciamento LS (allineato a estensione)
                    reinvest_dividends_input
                )
            if not lump_sum_df.empty: st.success("Simulazione LS OK.")

    if pac_total_df.empty or 'PortfolioValue' not in pac_total_df.columns or len(pac_total_df)<2: st.error("Simulazione PAC fallita."); st.stop()
    st.success("Simulazioni OK. Output:")

    # --- TABELLA METRICHE (invariata) ---
    # ...
    # --- GRAFICO EQUITY LINE (usa base_chart_date_index) ---
    st.subheader("Andamento Comparativo del Portafoglio")
    if not base_chart_date_index.empty:
        combined_equity_plot_df = pd.DataFrame(index=base_chart_date_index)
        # ... (logica di join e ffill per combined_equity_plot_df come nell'ULTIMA versione completa) ...
        # Assicurati che il ffill del PAC Capitale Investito sia corretto
        if not pac_total_df.empty: # PAC Data
            pac_plot = pac_total_df.set_index(pd.to_datetime(pac_total_df['Date']))
            combined_equity_plot_df = combined_equity_plot_df.join(pac_plot[['PortfolioValue', 'InvestedCapital']])
            combined_equity_plot_df.rename(columns={'PortfolioValue': 'PAC Valore Portafoglio', 'InvestedCapital': 'PAC Capitale Investito'}, inplace=True)
        if not lump_sum_df.empty: # LS Data
            ls_plot = lump_sum_df.set_index(pd.to_datetime(lump_sum_df['Date']))
            combined_equity_plot_df = combined_equity_plot_df.join(ls_plot[['PortfolioValue']])
            combined_equity_plot_df.rename(columns={'PortfolioValue': 'Lump Sum Valore Portafoglio'}, inplace=True)
        cash_bm_val = get_total_capital_invested(pac_total_df) if not pac_total_df.empty else 0 # Basato sul PAC
        if cash_bm_val > 0: combined_equity_plot_df['Cash (Valore Fisso 0%)'] = cash_bm_val
        
        # Ffill per estendere
        cols_to_ffill_equity = ['PAC Valore Portafoglio', 'Lump Sum Valore Portafoglio', 'Cash (Valore Fisso 0%)']
        for col in cols_to_ffill_equity:
            if col in combined_equity_plot_df.columns: combined_equity_plot_df[col] = combined_equity_plot_df[col].ffill()
        
        # Gestione PAC Capitale Investito (ffill fino a fine PAC, poi costante)
        if 'PAC Capitale Investito' in combined_equity_plot_df.columns and not pac_total_df.empty:
            combined_equity_plot_df['PAC Capitale Investito'] = combined_equity_plot_df['PAC Capitale Investito'].ffill() 
            last_pac_contribution_date_in_index = base_chart_date_index[base_chart_date_index.get_indexer([actual_pac_contribution_end_date_dt], method='ffill')[0]]
            last_known_invested_capital = combined_equity_plot_df.loc[last_pac_contribution_date_in_index, 'PAC Capitale Investito']
            if pd.notna(last_known_invested_capital): combined_equity_plot_df.loc[combined_equity_plot_df.index > last_pac_contribution_date_in_index, 'PAC Capitale Investito'] = last_known_invested_capital
        
        actual_cols_to_plot_equity = [c for c in ['PAC Valore Portafoglio', 'PAC Capitale Investito', 'Lump Sum Valore Portafoglio', 'Cash (Valore Fisso 0%)'] if c in combined_equity_plot_df.columns and not combined_equity_plot_df[c].isnull().all()]
        if actual_cols_to_plot_equity: st.line_chart(combined_equity_plot_df[actual_cols_to_plot_equity]); st.write("--- DEBUG: Grafico Equity VISUALIZZATO ---")
    else: st.warning("Indice base per grafici non creato.")


    # --- GRAFICO DRAWDOWN (usa base_chart_date_index) ---
    st.subheader("Andamento del Drawdown nel Tempo")
    # ... (codice come prima, ma usa base_chart_date_index per reindex e ffill dd_plot_df)
    if not base_chart_date_index.empty:
        drawdown_data_to_plot = {} # ... (popola come prima) ...
        if not pac_total_df.empty:
            pac_pv_dd = pac_total_df.set_index(pd.to_datetime(pac_total_df['Date']))['PortfolioValue']
            pac_dd_series = calculate_drawdown_series(pac_pv_dd)
            if not pac_dd_series.empty: drawdown_data_to_plot['PAC Drawdown (%)'] = pac_dd_series
        if not lump_sum_df.empty:
            ls_pv_dd = lump_sum_df.set_index(pd.to_datetime(lump_sum_df['Date']))['PortfolioValue']
            ls_dd_series = calculate_drawdown_series(ls_pv_dd)
            if not ls_dd_series.empty: drawdown_data_to_plot['Lump Sum Drawdown (%)'] = ls_dd_series
        if drawdown_data_to_plot:
            dd_plot_df_temp = pd.DataFrame(drawdown_data_to_plot)
            dd_plot_df = pd.DataFrame(index=base_chart_date_index) # Inizia con l'indice completo
            for col in dd_plot_df_temp.columns: dd_plot_df = dd_plot_df.join(dd_plot_df_temp[[col]], how='left'); dd_plot_df[col] = dd_plot_df[col].ffill()
            if not dd_plot_df.empty and not dd_plot_df.isnull().all().all(): st.line_chart(dd_plot_df); st.write("--- DEBUG: Grafico Drawdown VISUALIZZATO ---")
    else: st.warning("Indice base non creato, grafico drawdown saltato.")


    # --- ISTOGRAMMA RENDIMENTI ANNUALI (invariato) ---
    # ...
    
    # --- ROLLING METRICS (invariato, ma i dati di input ora sono estesi) ---
    # ... (assicurati che usi pac_total_df, che ora è esteso)
    # ... e che i grafici rolling usino .reindex(base_chart_date_index).ffill() se vuoi estenderli
    
    # --- STACKED AREA CHART (usa base_chart_date_index) ---
    st.subheader("Allocazione Dinamica Portafoglio PAC (Valore per Asset)")
    # ... (codice come prima, ma usa base_chart_date_index per reindex e ffill stack_df_data_reindexed) ...
    if asset_details_history_df is not None and not asset_details_history_df.empty and not base_chart_date_index.empty:
        # ... (logica per val_cols_stack e stack_df_data_temp) ...
        val_cols_stack = [f'{t}_value' for t in tickers_list if f'{t}_value' in asset_details_history_df.columns]
        if val_cols_stack:
            stack_df_data_temp = asset_details_history_df.set_index(pd.to_datetime(asset_details_history_df['Date']))[val_cols_stack]
            stack_df_data_reindexed = pd.DataFrame(index=base_chart_date_index)
            for col in stack_df_data_temp.columns: stack_df_data_reindexed=stack_df_data_reindexed.join(stack_df_data_temp[[col]],how='left'); stack_df_data_reindexed[col]=stack_df_data_reindexed[col].ffill().fillna(0)
            if not stack_df_data_reindexed.empty and not stack_df_data_reindexed.isnull().all().all(): st.area_chart(stack_df_data_reindexed); st.write("--- DEBUG: Stacked Area VISUALIZZATO ---")


    # --- TABELLE QUOTE/WAP (invariata) ---
    # ...
else: 
    st.info("Inserisci parametri e avvia simulazione.")
st.sidebar.markdown("---"); st.sidebar.markdown("Kriterion Quant")
