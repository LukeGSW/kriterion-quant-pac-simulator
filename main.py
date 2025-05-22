# simulatore_pac/main.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

# Importazioni dai moduli utils
try:
    from utils.data_loader import load_historical_data_yf
    from utils.pac_engine import run_pac_simulation
    from utils.benchmark_engine import run_lump_sum_simulation
    from utils.performance import (
        get_total_capital_invested, get_final_portfolio_value,
        calculate_total_return_percentage, calculate_cagr, get_duration_years,
        calculate_portfolio_returns, calculate_annualized_volatility,
        calculate_sharpe_ratio, calculate_max_drawdown, calculate_drawdown_series,
        generate_cash_flows_for_xirr, calculate_xirr_metric,
        # calculate_annual_returns, # RIMOSSO
        get_final_asset_details, calculate_wap_for_assets
        calculate_tracking_error # NUOVA IMPORTAZIONE
    )
    IMPORT_SUCCESS = True
except ImportError as import_err:
    IMPORT_SUCCESS = False
    IMPORT_ERROR_MESSAGE = str(import_err)

st.set_page_config(page_title="Simulatore PAC Core V4", layout="wide")
st.title("📘 Simulatore PAC Core Funzionalità V4")
st.caption("Progetto Kriterion Quant - Versione Semplificata")

if not IMPORT_SUCCESS:
    st.error(f"Errore critico import moduli utils: {IMPORT_ERROR_MESSAGE}")
    st.stop()

# --- Sidebar per Input Utente ---
st.sidebar.header("Parametri Simulazione")
st.sidebar.subheader("Asset e Allocazioni")
tickers_input_str = st.sidebar.text_input("Tickers (virgola sep.)", "AAPL,GOOG,MSFT", key="ui_tickers_v4")
allocations_input_str = st.sidebar.text_input("Allocazioni % (virgola sep.)", "60,20,20", key="ui_allocations_v4")
st.sidebar.subheader("Parametri PAC")
monthly_investment_input = st.sidebar.number_input("Importo Mensile (€/$)", 10.0, value=200.0, step=10.0, key="ui_monthly_inv_v4")
default_start_date_pac_sidebar = date(2020, 1, 1)
pac_start_date_contributions_ui = st.sidebar.date_input("Data Inizio Contributi PAC", default_start_date_pac_sidebar, key="ui_pac_start_date_v4")
default_duration_months = 36
duration_months_contributions_input = st.sidebar.number_input("Durata Contributi PAC (mesi)", 6, value=default_duration_months, step=1, key="ui_duration_months_v4")
reinvest_dividends_input = st.sidebar.checkbox("Reinvesti Dividendi (per PAC e LS)?", True, key="ui_reinvest_div_v4")
st.sidebar.subheader("Ribilanciamento Periodico (Solo per PAC)")
rebalance_active_input = st.sidebar.checkbox("Attiva Ribilanciamento (per PAC)?", False, key="ui_rebalance_active_v4")
rebalance_frequency_input_str = None
if rebalance_active_input:
    rebalance_frequency_input_str = st.sidebar.selectbox("Frequenza", ["Annuale", "Semestrale", "Trimestrale"], 0, key="ui_rebalance_freq_v4")
st.sidebar.subheader("Parametri Metriche")
risk_free_rate_input = st.sidebar.number_input("Tasso Risk-Free Annuale (%) per Sharpe", 0.0, value=1.0, step=0.1, format="%.2f", key="ui_rf_rate_v4")
# Rimosso input per rolling metrics window
run_simulation_button = st.sidebar.button("🚀 Avvia Simulazioni", key="ui_run_button_v4")

if run_simulation_button:
    # --- VALIDAZIONE INPUT (come prima) ---
    tickers_list = [t.strip().upper() for t in tickers_input_str.split(',') if t.strip()] # ... (resto validazione)
    error_in_input = False; allocations_list_norm = []; allocations_float_list_raw = []
    if not tickers_list: st.error("Errore: Minimo un ticker."); error_in_input = True
    if not error_in_input:
        try:
            allocations_float_list_raw = [float(a.strip()) for a in allocations_input_str.split(',') if a.strip()]
            if len(tickers_list)!=len(allocations_float_list_raw): st.error("Errore: N. ticker diverso da N. allocazioni."); error_in_input=True
            elif not np.isclose(sum(allocations_float_list_raw),100.0): st.error(f"Errore: Somma allocazioni ({sum(allocations_float_list_raw)}%) != 100%."); error_in_input=True
            else: allocations_list_norm = [a/100.0 for a in allocations_float_list_raw]
        except ValueError: st.error("Errore: Allocazioni non numeriche."); error_in_input=True
    if error_in_input: st.stop()

    st.header(f"Risultati Simulazione per: {', '.join(tickers_list)}")
    # ... (Info allocazioni e ribilanciamento)

    # --- PREPARAZIONE DATE E CARICAMENTO DATI (come ultima versione stabile) ---
    pac_contribution_start_dt = pd.to_datetime(pac_start_date_contributions_ui) # ... (resto come prima)
    pac_contribution_start_str = pac_contribution_start_dt.strftime('%Y-%m-%d')
    pac_contribution_end_dt = pac_contribution_start_dt + pd.DateOffset(months=duration_months_contributions_input)
    data_download_start_str = (pac_contribution_start_dt - pd.Timedelta(days=365*1)).strftime('%Y-%m-%d')
    data_download_end_str = datetime.today().strftime('%Y-%m-%d')
    historical_data_map = {}; all_data_loaded_successfully = True; latest_overall_data_date_ts = pd.Timestamp.min
    for tkr in tickers_list:
        with st.spinner(f"Dati per {tkr}..."): data = load_historical_data_yf(tkr, data_download_start_str, data_download_end_str)
        if data.empty or data.index.min() > pac_contribution_start_dt or data.index.max() < (pac_contribution_end_dt - pd.Timedelta(days=1)):
            st.error(f"Dati storici insufficienti per {tkr} per periodo PAC. Ultima data per {tkr}: {data.index.max().date() if not data.empty else 'N/A'}.")
            all_data_loaded_successfully = False; break
        historical_data_map[tkr] = data
        if data.index.max() > latest_overall_data_date_ts: latest_overall_data_date_ts = data.index.max()
    if not all_data_loaded_successfully: st.stop()
    st.success("Dati storici caricati.")
    simulation_actual_end_dt = min(latest_overall_data_date_ts, pd.Timestamp(datetime.today()))
    base_chart_date_index = pd.DatetimeIndex([])
    if pac_contribution_start_dt <= simulation_actual_end_dt:
        base_chart_date_index = pd.date_range(start=pac_contribution_start_dt, end=simulation_actual_end_dt, freq='B')
        base_chart_date_index.name = 'Date'
    if base_chart_date_index.empty: st.warning("Indice date per grafici non generato.")

    # --- ESECUZIONE SIMULAZIONI (come ultima versione stabile) ---
    pac_total_df, asset_details_history_df = pd.DataFrame(), pd.DataFrame() # ... (come prima)
    try:
        with st.spinner("Simulazione PAC..."): pac_total_df, asset_details_history_df = run_pac_simulation(historical_data_map, tickers_list, allocations_list_norm, monthly_investment_input, pac_contribution_start_str, duration_months_contributions_input, reinvest_dividends_input, rebalance_active_input, rebalance_frequency_input_str)
    except Exception as e: st.error(f"Errore CRITICO PAC: {e}"); import traceback; st.text(traceback.format_exc()); st.stop()
    lump_sum_df = pd.DataFrame()
    if not pac_total_df.empty and 'PortfolioValue' in pac_total_df.columns and len(pac_total_df)>=2:
        total_invested_by_pac = get_total_capital_invested(pac_total_df)
        if total_invested_by_pac > 0:
            with st.spinner("Simulazione LS..."): lump_sum_df = run_lump_sum_simulation(historical_data_map, tickers_list, allocations_list_norm, total_invested_by_pac, pac_contribution_start_dt, reinvest_dividends_input)
            if not lump_sum_df.empty: st.success("Simulazione LS completata.")
    if pac_total_df.empty or 'PortfolioValue' not in pac_total_df.columns or len(pac_total_df)<2: st.error("Simulazione PAC fallita."); st.stop()
    st.success("Simulazioni OK. Elaborazione output.")

    # --- FUNZIONE HELPER PER METRICHE (invariata dalla tua ultima versione funzionante che nasconde Vol PAC) ---
    # --- FUNZIONE HELPER PER METRICHE (MODIFICATA) ---
    def calculate_and_format_metrics(sim_df, strategy_name, 
                                     total_invested_override=None, 
                                     is_pac=False, 
                                     benchmark_returns_for_te=None): # Nuovo parametro per TE
        metrics_display = {}
        # Aggiungiamo "Tracking Error" alle chiavi se è un PAC e ha un benchmark
        keys_ordered_pac = ["Capitale Investito", "Valore Finale", "Rend. Totale", "CAGR", "XIRR", "Sharpe", "Max Drawdown"]
        if is_pac and benchmark_returns_for_te is not None:
            keys_ordered_pac.append("Tracking Error (vs LS)")
            
        keys_ordered_ls = ["Capitale Investito", "Valore Finale", "Rend. Totale", "CAGR", "XIRR", "Volatilità Ann.", "Sharpe", "Max Drawdown"]
        
        current_keys = keys_ordered_pac if is_pac else keys_ordered_ls
        for k in current_keys: metrics_display[k] = "N/A"

        if sim_df.empty or 'PortfolioValue' not in sim_df.columns or len(sim_df) < 2: 
            return metrics_display
        
        # ... (calcolo di actual_total_invested, final_portfolio_val, rend_tot, cagr_strat come prima) ...
        actual_total_invested = total_invested_override if total_invested_override is not None else get_total_capital_invested(sim_df)
        metrics_display["Capitale Investito"] = f"{actual_total_invested:,.2f}"; final_portfolio_val = get_final_portfolio_value(sim_df)
        metrics_display["Valore Finale"] = f"{final_portfolio_val:,.2f}"; rend_tot = calculate_total_return_percentage(final_portfolio_val, actual_total_invested)
        metrics_display["Rend. Totale"] = f"{rend_tot:.2f}%" if pd.notna(rend_tot) else "N/A"
        duration_yrs_strat = get_duration_years(sim_df.copy()); cagr_strat = calculate_cagr(final_portfolio_val, actual_total_invested, duration_yrs_strat)
        metrics_display["CAGR"] = f"{cagr_strat:.2f}%" if pd.notna(cagr_strat) else "N/A"

        returns_strat = calculate_portfolio_returns(sim_df.copy())
        
        vol_strat_numeric = calculate_annualized_volatility(returns_strat)
        if not is_pac: 
            metrics_display["Volatilità Ann."] = f"{vol_strat_numeric:.2f}%" if pd.notna(vol_strat_numeric) else "N/A"
        
        sharpe_strat = calculate_sharpe_ratio(returns_strat, risk_free_rate_annual=(risk_free_rate_input/100.0))
        metrics_display["Sharpe"] = f"{sharpe_strat:.2f}" if pd.notna(sharpe_strat) else "N/A"
        
        mdd_strat = calculate_max_drawdown(sim_df.copy())
        metrics_display["Max Drawdown"] = f"{mdd_strat:.2f}%" if pd.notna(mdd_strat) else "N/A"

        if is_pac:
            xirr_dates, xirr_values = generate_cash_flows_for_xirr(sim_df, pac_contribution_start_str, duration_months_contributions_input, monthly_investment_input, final_portfolio_val)
            xirr_val = calculate_xirr_metric(xirr_dates, xirr_values)
            metrics_display["XIRR"] = f"{xirr_val:.2f}%" if pd.notna(xirr_val) else "N/A"
            
            # Calcola Tracking Error per PAC se benchmark_returns sono forniti
            if benchmark_returns_for_te is not None and not benchmark_returns_for_te.empty:
                te_strat = calculate_tracking_error(returns_strat, benchmark_returns_for_te)
                metrics_display["Tracking Error (vs LS)"] = f"{te_strat:.2f}%" if pd.notna(te_strat) else "N/A"
            elif "Tracking Error (vs LS)" in metrics_display: # Assicura che sia N/A se non calcolabile
                metrics_display["Tracking Error (vs LS)"] = "N/A"

        else: # Per Lump Sum
            metrics_display["XIRR"] = metrics_display["CAGR"] 
            # Il Tracking Error per LS rispetto a se stesso non ha senso, quindi non lo calcoliamo qui.
            # Se volessimo TE di LS vs un altro benchmark, la logica cambierebbe.
        return metrics_display

    # Calcola i rendimenti giornalieri del Lump Sum per il Tracking Error del PAC
    ls_daily_returns_for_te = pd.Series(dtype=float)
    if not lump_sum_df.empty:
        ls_daily_returns_for_te = calculate_portfolio_returns(lump_sum_df.copy())

    metrics_pac_display = calculate_and_format_metrics(pac_total_df, "PAC", is_pac=True, benchmark_returns_for_te=ls_daily_returns_for_te)
    
    # Aggiorna l'ordine desiderato delle metriche per includere Tracking Error
    desired_ordered_metrics = ["Capitale Investito", "Valore Finale", "Rend. Totale", "CAGR", "XIRR", 
                               "Volatilità Ann.", "Sharpe", "Max Drawdown", "Tracking Error (vs LS)"]
    
    # Costruisci la tabella
    df_for_table = pd.DataFrame(index=desired_ordered_metrics)
    df_for_table["PAC"] = df_for_table.index.map(metrics_pac_display).fillna("N/A")

    if not lump_sum_df.empty:
        total_invested_val_pac = get_total_capital_invested(pac_total_df)
        metrics_ls_display = calculate_and_format_metrics(lump_sum_df, "Lump Sum", total_invested_override=total_invested_val_pac, is_pac=False)
        df_for_table["Lump Sum"] = df_for_table.index.map(metrics_ls_display).fillna("N/A")
    
    # Rimuovi la riga "Tracking Error (vs LS)" se la colonna "Lump Sum" non esiste (non dovrebbe accadere se PAC esiste)
    # o se il valore per PAC è N/A e non c'è Lump Sum
    if "Lump Sum" not in df_for_table.columns and "Tracking Error (vs LS)" in df_for_table.index:
         df_for_table.drop("Tracking Error (vs LS)", inplace=True)
    elif "Tracking Error (vs LS)" in df_for_table.index and df_for_table.loc["Tracking Error (vs LS)", "PAC"] == "N/A":
         # Se non c'è LS, non ha senso mostrare TE del PAC vs LS
         if "Lump Sum" not in df_for_table.columns:
             df_for_table.drop("Tracking Error (vs LS)", inplace=True)


    # Assicurati che la riga "Volatilità Ann." esista e sia N/A per PAC se non già gestito
    if "Volatilità Ann." in df_for_table.index:
        if "PAC" in df_for_table.columns and df_for_table.loc["Volatilità Ann.", "PAC"] != "N/A": # Evita di sovrascrivere se già N/A
            if metrics_pac_display.get("Volatilità Ann.", "N/A") == "N/A": # Controlla se era inteso come N/A
                 df_for_table.loc["Volatilità Ann.", "PAC"] = "N/A"
    elif "Lump Sum" in df_for_table.columns and metrics_ls_display.get("Volatilità Ann.") != "N/A": # Aggiungi riga Volatilità se LS ce l'ha e la riga manca
        df_for_table.loc["Volatilità Ann.", "Lump Sum"] = metrics_ls_display.get("Volatilità Ann.")
        df_for_table.loc["Volatilità Ann.", "PAC"] = "N/A"


    # Filtra per le righe effettivamente presenti in df_for_table per evitare errori se una metrica manca per tutti
    final_ordered_index = [label for label in desired_ordered_metrics if label in df_for_table.index]
    df_for_table = df_for_table.reindex(final_ordered_index).fillna("N/A")


    st.subheader("Metriche di Performance Riepilogative")
    if not df_for_table.empty: 
        st.table(df_for_table)


    # --- GRAFICO EQUITY LINE ESTESO (Logica invariata) ---
    st.subheader("Andamento Comparativo del Portafoglio")
    # ... (Logica grafico Equity come ultima versione stabile) ...
    if not base_chart_date_index.empty:
        equity_plot_df = pd.DataFrame(index=base_chart_date_index)
        if not pac_total_df.empty: pac_plot = pac_total_df.set_index(pd.to_datetime(pac_total_df['Date'])); equity_plot_df = equity_plot_df.join(pac_plot[['PortfolioValue', 'InvestedCapital']]); equity_plot_df.rename(columns={'PortfolioValue': 'PAC Valore Portafoglio', 'InvestedCapital': 'PAC Capitale Investito'}, inplace=True)
        if not lump_sum_df.empty: ls_plot = lump_sum_df.set_index(pd.to_datetime(lump_sum_df['Date'])); equity_plot_df = equity_plot_df.join(ls_plot[['PortfolioValue']]); equity_plot_df.rename(columns={'PortfolioValue': 'Lump Sum Valore Portafoglio'}, inplace=True)
        cash_bm_val = get_total_capital_invested(pac_total_df) if not pac_total_df.empty else 0
        if cash_bm_val > 0: equity_plot_df['Cash (Valore Fisso 0%)'] = cash_bm_val
        cols_to_ffill_equity = ['PAC Valore Portafoglio', 'Lump Sum Valore Portafoglio', 'Cash (Valore Fisso 0%)', 'PAC Capitale Investito']
        for col in cols_to_ffill_equity:
            if col in equity_plot_df.columns: equity_plot_df[col] = equity_plot_df[col].ffill()
        if 'PAC Capitale Investito' in equity_plot_df.columns and not pac_total_df.empty:
            equity_plot_df['PAC Capitale Investito'] = equity_plot_df['PAC Capitale Investito'].ffill() 
            last_pac_contrib_dt = pac_contribution_start_dt + pd.DateOffset(months=duration_months_contributions_input -1)
            if not base_chart_date_index.empty:
                 idx_loc_series = base_chart_date_index.get_indexer([last_pac_contrib_dt], method='ffill')
                 if idx_loc_series.size > 0 and idx_loc_series[0] != -1:
                    actual_last_contrib_idx_date = base_chart_date_index[idx_loc_series[0]]
                    last_known_cap_val = equity_plot_df.loc[actual_last_contrib_idx_date, 'PAC Capitale Investito']
                    if pd.notna(last_known_cap_val): equity_plot_df.loc[equity_plot_df.index > actual_last_contrib_idx_date, 'PAC Capitale Investito'] = last_known_cap_val
        cols_to_plot = [c for c in equity_plot_df.columns if not equity_plot_df[c].isnull().all()]
        if cols_to_plot: st.line_chart(equity_plot_df[cols_to_plot])

    # --- GRAFICO DRAWDOWN ESTESO (Logica invariata) ---
    st.subheader("Andamento del Drawdown nel Tempo")
    # ... (Logica grafico Drawdown come ultima versione stabile) ...
    if not base_chart_date_index.empty:
        drawdown_data_to_plot = {}
        if not pac_total_df.empty: pac_pv_dd = pac_total_df.set_index(pd.to_datetime(pac_total_df['Date']))['PortfolioValue']; pac_dd_series = calculate_drawdown_series(pac_pv_dd);
        if not pac_dd_series.empty: drawdown_data_to_plot['PAC Drawdown (%)'] = pac_dd_series
        if not lump_sum_df.empty: ls_pv_dd = lump_sum_df.set_index(pd.to_datetime(lump_sum_df['Date']))['PortfolioValue']; ls_dd_series = calculate_drawdown_series(ls_pv_dd);
        if not ls_dd_series.empty: drawdown_data_to_plot['Lump Sum Drawdown (%)'] = ls_dd_series
        if drawdown_data_to_plot:
            dd_plot_df_temp = pd.DataFrame(drawdown_data_to_plot); dd_plot_df = pd.DataFrame(index=base_chart_date_index)
            for col in dd_plot_df_temp.columns: dd_plot_df = dd_plot_df.join(dd_plot_df_temp[[col]], how='left'); dd_plot_df[col] = dd_plot_df[col].ffill()
            if not dd_plot_df.empty and not dd_plot_df.isnull().all().all(): st.line_chart(dd_plot_df)

    # --- STACKED AREA CHART ESTESO (Logica invariata) ---
    if asset_details_history_df is not None and not asset_details_history_df.empty and not base_chart_date_index.empty:
        st.subheader("Allocazione Dinamica Portafoglio PAC (Valore per Asset)")
        # ... (Logica stacked area chart come ultima versione stabile) ...
        value_cols_stacked = [f'{t}_value' for t in tickers_list if f'{t}_value' in asset_details_history_df.columns]
        if value_cols_stacked:
            stacked_df_temp = asset_details_history_df.set_index(pd.to_datetime(asset_details_history_df['Date']))[value_cols_stacked]
            stacked_df_plot = pd.DataFrame(index=base_chart_date_index)
            for col in stacked_df_temp.columns: stacked_df_plot=stacked_df_plot.join(stacked_df_temp[[col]],how='left'); stacked_df_plot[col]=stacked_df_plot[col].ffill().fillna(0)
            if not stacked_df_plot.empty and not stacked_df_plot.isnull().all().all(): st.area_chart(stacked_df_plot)

    # --- TABELLA QUOTE/WAP (Logica invariata) ---
    if asset_details_history_df is not None and not asset_details_history_df.empty:
        st.subheader("Dettagli Finali per Asset nel PAC")
        # ... (Logica tabella WAP come prima) ...
        final_asset_details_map = get_final_asset_details(asset_details_history_df, tickers_list)
        wap_map = calculate_wap_for_assets(final_asset_details_map)
        table_data_wap = []
        for ticker_wap in tickers_list:
            asset_info = final_asset_details_map.get(ticker_wap, {'shares': 0.0, 'capital_invested': 0.0})
            table_data_wap.append({"Ticker": ticker_wap, "Quote Finali": f"{asset_info['shares']:.4f}", "Capitale Investito (Asset)": f"{asset_info['capital_invested']:,.2f}", "Prezzo Medio Carico (WAP)": f"{wap_map.get(ticker_wap, np.nan):,.2f}" if pd.notna(wap_map.get(ticker_wap, np.nan)) else "N/A"})
        if table_data_wap: st.table(pd.DataFrame(table_data_wap).set_index("Ticker"))
else: 
    st.info("Inserisci parametri e avvia simulazione.")
# In main.py, all'interno della definizione della sidebar



# --- MINI GUIDA ---
st.sidebar.markdown("---") # Separatore
st.sidebar.subheader("Guida Rapida")
st.sidebar.caption(
    "1. **Inserisci i Ticker**: Specifica uno o più simboli di strumenti finanziari (es. AAPL, GOOG per azioni USA; "
    "ISP.MI per Borsa Italiana; VWCE.DE per ETF su Xetra) separati da virgola.\n"
    "2. **Definisci le Allocazioni**: Inserisci le percentuali per ciascun ticker, separate da virgola. La somma deve fare 100%.\n"
    "3. **Imposta i Parametri PAC**: Definisci l'importo del versamento mensile, la data di inizio dei contributi e la loro durata in mesi.\n"
    "4. **Opzioni Avanzate**: Scegli se reinvestire i dividendi e se attivare il ribilanciamento periodico per la strategia PAC.\n"
    "5. **Metriche**: Imposta il tasso risk-free per il calcolo dello Sharpe Ratio.\n"
    "6. **Avvia**: Clicca 'Avvia Simulazioni' per visualizzare i risultati e i grafici comparativi."
)
# --- FINE MINI GUIDA ---

st.sidebar.markdown("---")
st.sidebar.markdown("Progetto Kriterion Quant")
st.sidebar.markdown("---")
