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
        # Funzioni per WAP e quote finali che avevamo in performance.py
        get_final_asset_details, calculate_wap_for_assets
        # Assicurati che calculate_annual_returns e le funzioni di rolling NON siano importate
        # se le abbiamo rimosse da performance.py per questo test di debug sulla volatilità
    )
    IMPORT_SUCCESS = True
except ImportError as import_err:
    IMPORT_SUCCESS = False
    IMPORT_ERROR_MESSAGE = str(import_err)

st.set_page_config(page_title="Simulatore PAC - Debug Volatilità", layout="wide")
st.title("📘 Simulatore PAC - Debug Volatilità")
st.caption("Progetto Kriterion Quant")

if not IMPORT_SUCCESS:
    st.error(f"Errore critico durante l'importazione dei moduli utils: {IMPORT_ERROR_MESSAGE}")
    st.stop()

# --- Sidebar per Input Utente (invariata rispetto all'ultima versione stabile) ---
st.sidebar.header("Parametri Simulazione")
st.sidebar.subheader("Asset e Allocazioni")
tickers_input_str = st.sidebar.text_input("Tickers (virgola sep.)", "AAPL,GOOG,MSFT", key="main_tickers")
allocations_input_str = st.sidebar.text_input("Allocazioni % (virgola sep.)", "60,20,20", key="main_allocations")
st.sidebar.subheader("Parametri PAC")
monthly_investment_input = st.sidebar.number_input("Importo Mensile (€/$)", 10.0, value=200.0, step=10.0, key="main_monthly_inv")
default_start_date_pac_sidebar = date(2020, 1, 1)
pac_start_date_contributions_ui = st.sidebar.date_input("Data Inizio Contributi PAC", default_start_date_pac_sidebar, key="main_pac_start_date")
end_date_for_default_duration_calc = default_start_date_pac_sidebar.replace(year=default_start_date_pac_sidebar.year + 3)
if end_date_for_default_duration_calc > date.today():
    end_date_for_default_duration_calc = date.today()
default_duration_months = max(6, (end_date_for_default_duration_calc.year - pac_start_date_contributions_ui.year) * 12 + \
                                 (end_date_for_default_duration_calc.month - pac_start_date_contributions_ui.month))
if default_duration_months < 6: default_duration_months = 36
duration_months_contributions_input = st.sidebar.number_input("Durata Contributi PAC (mesi)", 6, value=default_duration_months, step=1, key="main_duration_months")
reinvest_dividends_input = st.sidebar.checkbox("Reinvesti Dividendi (per PAC e LS)?", True, key="main_reinvest_div")
st.sidebar.subheader("Ribilanciamento Periodico (Solo per PAC)")
rebalance_active_input = st.sidebar.checkbox("Attiva Ribilanciamento (per PAC)?", False, key="main_rebalance_active")
rebalance_frequency_input_str = None
if rebalance_active_input:
    rebalance_frequency_input_str = st.sidebar.selectbox(
        "Frequenza Ribilanciamento", ["Annuale", "Semestrale", "Trimestrale"], index=0, key="main_rebalance_freq"
    )
st.sidebar.subheader("Parametri Metriche")
risk_free_rate_input = st.sidebar.number_input("Tasso Risk-Free Annuale (%) per Sharpe", 0.0, value=1.0, step=0.1, format="%.2f", key="main_rf_rate")
# Rimosso input per rolling metrics per ora
run_simulation_button = st.sidebar.button("🚀 Avvia Simulazioni", key="main_run_button")

if run_simulation_button:
    # --- VALIDAZIONE INPUT (come prima) ---
    tickers_list = [t.strip().upper() for t in tickers_input_str.split(',') if t.strip()]
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
    pac_contribution_start_dt = pd.to_datetime(pac_start_date_contributions_ui)
    pac_contribution_start_str = pac_contribution_start_dt.strftime('%Y-%m-%d')
    pac_contribution_end_dt = pac_contribution_start_dt + pd.DateOffset(months=duration_months_contributions_input)
    data_download_start_str = (pac_contribution_start_dt - pd.Timedelta(days=365*1)).strftime('%Y-%m-%d')
    data_download_end_str = datetime.today().strftime('%Y-%m-%d')
    historical_data_map = {}; all_data_loaded_successfully = True; latest_overall_data_date_ts = pd.Timestamp.min
    for tkr in tickers_list:
        with st.spinner(f"Dati per {tkr}..."): data = load_historical_data_yf(tkr, data_download_start_str, data_download_end_str)
        if data.empty or data.index.min() > pac_contribution_start_dt or data.index.max() < (pac_contribution_end_dt - pd.Timedelta(days=1)):
            st.error(f"Dati storici insufficienti per {tkr} per coprire periodo PAC (fino a {pac_contribution_end_dt.date()}). Ultima data per {tkr}: {data.index.max().date() if not data.empty else 'N/A'}.")
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
    if base_chart_date_index.empty: st.warning("Indice date per grafici non generato."); # Non fare st.stop() qui

    # --- ESECUZIONE SIMULAZIONI (come ultima versione stabile) ---
    pac_total_df, asset_details_history_df = pd.DataFrame(), pd.DataFrame()
    try:
        with st.spinner("Simulazione PAC..."): pac_total_df, asset_details_history_df = run_pac_simulation(historical_data_map, tickers_list, allocations_list_norm, monthly_investment_input, pac_contribution_start_str, duration_months_contributions_input, reinvest_dividends_input, rebalance_active_input, rebalance_frequency_input_str)
    except Exception as e: st.error(f"Errore CRITICO PAC: {e}"); import traceback; st.text(traceback.format_exc()); st.stop()

    lump_sum_df = pd.DataFrame()
    if not pac_total_df.empty and 'PortfolioValue' in pac_total_df.columns and len(pac_total_df)>=2:
        total_invested_by_pac = get_total_capital_invested(pac_total_df)
        if total_invested_by_pac > 0:
            with st.spinner("Simulazione LS..."): lump_sum_df = run_lump_sum_simulation(historical_data_map, tickers_list, allocations_list_norm, total_invested_by_pac, pac_contribution_start_dt, reinvest_dividends_input) # Rimossi start/end espliciti per LS engine
            if not lump_sum_df.empty: st.success("Simulazione LS completata.")
    
    if pac_total_df.empty or 'PortfolioValue' not in pac_total_df.columns or len(pac_total_df)<2: st.error("Simulazione PAC fallita."); st.stop()
    st.success("Simulazioni OK. Elaborazione output.")

    # --- FUNZIONE HELPER PER METRICHE (con DEBUG per rendimenti PAC) ---
    def calculate_and_format_metrics(sim_df, strategy_name, total_invested_override=None, is_pac=False):
        metrics_display = {}
        keys_ordered = ["Capitale Investito", "Valore Finale", "Rend. Totale", "CAGR", "XIRR", "Volatilità Ann.", "Sharpe", "Max Drawdown"]
        for k in keys_ordered: metrics_display[k] = "N/A"

        if sim_df.empty or 'PortfolioValue' not in sim_df.columns or len(sim_df) < 2: 
            return metrics_display
        
        actual_total_invested = total_invested_override if total_invested_override is not None else get_total_capital_invested(sim_df)
        metrics_display["Capitale Investito"] = f"{actual_total_invested:,.2f}"
        final_portfolio_val = get_final_portfolio_value(sim_df)
        metrics_display["Valore Finale"] = f"{final_portfolio_val:,.2f}"
        rend_tot = calculate_total_return_percentage(final_portfolio_val, actual_total_invested)
        metrics_display["Rend. Totale"] = f"{rend_tot:.2f}%" if pd.notna(rend_tot) else "N/A"
        
        duration_yrs_strat = get_duration_years(sim_df.copy())
        cagr_strat = calculate_cagr(final_portfolio_val, actual_total_invested, duration_yrs_strat)
        metrics_display["CAGR"] = f"{cagr_strat:.2f}%" if pd.notna(cagr_strat) else "N/A"
        
        returns_strat = calculate_portfolio_returns(sim_df.copy())

        # <<< --- BLOCCO DI DEBUG PER RENDIMENTI PAC --- >>>
        if is_pac and not returns_strat.empty:
            st.subheader(f"--- DEBUG: Dettagli Rendimenti Giornalieri {strategy_name} ---")
            st.write(f"Primi 30 rendimenti giornalieri {strategy_name}:")
            st.dataframe(returns_strat.head(30))
            st.write(f"Ultimi 30 rendimenti giornalieri {strategy_name}:")
            st.dataframe(returns_strat.tail(30))
            st.write(f"Statistiche descrittive rendimenti giornalieri {strategy_name}:")
            st.dataframe(returns_strat.describe())
        # <<< --- FINE BLOCCO DI DEBUG --- >>>
            
        vol_strat = calculate_annualized_volatility(returns_strat)
        sharpe_strat = calculate_sharpe_ratio(returns_strat, risk_free_rate_annual=(risk_free_rate_input/100.0))
        mdd_strat = calculate_max_drawdown(sim_df.copy()) # Passa sim_df, non returns_strat
        
        metrics_display["Volatilità Ann."] = f"{vol_strat:.2f}%" if pd.notna(vol_strat) else "N/A"
        metrics_display["Sharpe"] = f"{sharpe_strat:.2f}" if pd.notna(sharpe_strat) else "N/A"
        metrics_display["Max Drawdown"] = f"{mdd_strat:.2f}%" if pd.notna(mdd_strat) else "N/A"

        if is_pac:
            xirr_dates, xirr_values = generate_cash_flows_for_xirr(sim_df, pac_contribution_start_str, duration_months_contributions_input, monthly_investment_input, final_portfolio_val)
            xirr_val = calculate_xirr_metric(xirr_dates, xirr_values)
            metrics_display["XIRR"] = f"{xirr_val:.2f}%" if pd.notna(xirr_val) else "N/A"
        else: 
            metrics_display["XIRR"] = metrics_display["CAGR"]
        return metrics_display

    metrics_pac_display = calculate_and_format_metrics(pac_total_df, "PAC", is_pac=True)
    df_for_table = pd.DataFrame.from_dict(metrics_pac_display, orient='index', columns=['PAC'])
    df_for_table.index.name = "Metrica"

    if not lump_sum_df.empty:
        total_invested_val_pac = get_total_capital_invested(pac_total_df)
        metrics_ls_display = calculate_and_format_metrics(lump_sum_df, "Lump Sum", total_invested_override=total_invested_val_pac, is_pac=False)
        df_for_table["Lump Sum"] = df_for_table.index.map(metrics_ls_display)
    
    st.subheader("Metriche di Performance Riepilogative")
    st.table(df_for_table)

    # --- GRAFICO EQUITY LINE ESTESO ---
    st.subheader("Andamento Comparativo del Portafoglio")
    if not base_chart_date_index.empty:
        equity_plot_df = pd.DataFrame(index=base_chart_date_index)
        # ... (Logica di join e ffill per equity_plot_df come nella versione completa precedente) ...
        if not pac_total_df.empty: pac_plot = pac_total_df.set_index(pd.to_datetime(pac_total_df['Date'])); equity_plot_df = equity_plot_df.join(pac_plot[['PortfolioValue', 'InvestedCapital']]); equity_plot_df.rename(columns={'PortfolioValue': 'PAC Valore Portafoglio', 'InvestedCapital': 'PAC Capitale Investito'}, inplace=True)
        if not lump_sum_df.empty: ls_plot = lump_sum_df.set_index(pd.to_datetime(lump_sum_df['Date'])); equity_plot_df = equity_plot_df.join(ls_plot[['PortfolioValue']]); equity_plot_df.rename(columns={'PortfolioValue': 'Lump Sum Valore Portafoglio'}, inplace=True)
        cash_bm_val = get_total_capital_invested(pac_total_df) if not pac_total_df.empty else 0
        if cash_bm_val > 0: equity_plot_df['Cash (Valore Fisso 0%)'] = cash_bm_val
        cols_to_ffill_equity = ['PAC Valore Portafoglio', 'Lump Sum Valore Portafoglio', 'Cash (Valore Fisso 0%)', 'PAC Capitale Investito']
        for col in cols_to_ffill_equity:
            if col in equity_plot_df.columns: equity_plot_df[col] = equity_plot_df[col].ffill()
        if 'PAC Capitale Investito' in equity_plot_df.columns and not pac_total_df.empty:
            equity_plot_df['PAC Capitale Investito'] = equity_plot_df['PAC Capitale Investito'].ffill() 
            last_pac_contrib_date_dt_for_fill = pac_contribution_start_dt + pd.DateOffset(months=duration_months_contributions_input -1)
            if not base_chart_date_index.empty and last_pac_contrib_date_dt_for_fill in base_chart_date_index: # Verifica se la data è nell'indice
                 actual_last_contrib_idx_date_fill = last_pac_contrib_date_dt_for_fill
            elif not base_chart_date_index.empty: # Altrimenti prendi il più vicino precedente
                 actual_last_contrib_idx_date_fill = base_chart_date_index[base_chart_date_index.get_indexer([last_pac_contrib_date_dt_for_fill], method='ffill')[0]]
            else: actual_last_contrib_idx_date_fill = None # Non fare nulla se l'indice è vuoto
            if actual_last_contrib_idx_date_fill is not None:
                 last_known_cap_val = equity_plot_df.loc[actual_last_contrib_idx_date_fill, 'PAC Capitale Investito']
                 if pd.notna(last_known_cap_val): equity_plot_df.loc[equity_plot_df.index > actual_last_contrib_idx_date_fill, 'PAC Capitale Investito'] = last_known_cap_val
        cols_to_plot = [c for c in equity_plot_df.columns if not equity_plot_df[c].isnull().all()]
        if cols_to_plot: st.line_chart(equity_plot_df[cols_to_plot])

    # --- GRAFICO DRAWDOWN ESTESO ---
    st.subheader("Andamento del Drawdown nel Tempo")
    if not base_chart_date_index.empty:
        # ... (Logica per drawdown_plot_df come nella versione completa precedente) ...
        drawdown_data_to_plot = {}
        if not pac_total_df.empty: pac_pv_dd = pac_total_df.set_index(pd.to_datetime(pac_total_df['Date']))['PortfolioValue']; pac_dd_series = calculate_drawdown_series(pac_pv_dd);
        if not pac_dd_series.empty: drawdown_data_to_plot['PAC Drawdown (%)'] = pac_dd_series
        if not lump_sum_df.empty: ls_pv_dd = lump_sum_df.set_index(pd.to_datetime(lump_sum_df['Date']))['PortfolioValue']; ls_dd_series = calculate_drawdown_series(ls_pv_dd);
        if not ls_dd_series.empty: drawdown_data_to_plot['Lump Sum Drawdown (%)'] = ls_dd_series
        if drawdown_data_to_plot:
            dd_plot_df_temp = pd.DataFrame(drawdown_data_to_plot); dd_plot_df = pd.DataFrame(index=base_chart_date_index)
            for col in dd_plot_df_temp.columns: dd_plot_df = dd_plot_df.join(dd_plot_df_temp[[col]], how='left'); dd_plot_df[col] = dd_plot_df[col].ffill()
            if not dd_plot_df.empty and not dd_plot_df.isnull().all().all(): st.line_chart(dd_plot_df)

    # --- STACKED AREA CHART ESTESO ---
    if asset_details_history_df is not None and not asset_details_history_df.empty and not base_chart_date_index.empty:
        st.subheader("Allocazione Dinamica Portafoglio PAC (Valore per Asset)")
        # ... (Logica per stacked_df_plot come nella versione completa precedente) ...
        value_cols_stacked = [f'{t}_value' for t in tickers_list if f'{t}_value' in asset_details_history_df.columns]
        if value_cols_stacked:
            stacked_df_temp = asset_details_history_df.set_index(pd.to_datetime(asset_details_history_df['Date']))[value_cols_stacked]
            stacked_df_plot = pd.DataFrame(index=base_chart_date_index)
            for col in stacked_df_temp.columns: stacked_df_plot=stacked_df_plot.join(stacked_df_temp[[col]],how='left'); stacked_df_plot[col]=stacked_df_plot[col].ffill().fillna(0)
            if not stacked_df_plot.empty and not stacked_df_plot.isnull().all().all(): st.area_chart(stacked_df_plot)

    # --- TABELLA QUOTE/WAP (come prima) ---
    if asset_details_history_df is not None and not asset_details_history_df.empty:
        st.subheader("Dettagli Finali per Asset nel PAC")
        # ... (Logica tabella WAP come prima, usando get_final_asset_details e calculate_wap_for_assets) ...
        final_asset_details_map = get_final_asset_details(asset_details_history_df, tickers_list)
        wap_map = calculate_wap_for_assets(final_asset_details_map)
        table_data_wap = []
        for ticker_wap in tickers_list:
            asset_info = final_asset_details_map.get(ticker_wap, {'shares': 0.0, 'capital_invested': 0.0})
            table_data_wap.append({"Ticker": ticker_wap, "Quote Finali": f"{asset_info['shares']:.4f}", "Capitale Investito (Asset)": f"{asset_info['capital_invested']:,.2f}", "Prezzo Medio Carico (WAP)": f"{wap_map.get(ticker_wap, np.nan):,.2f}" if pd.notna(wap_map.get(ticker_wap, np.nan)) else "N/A"})
        if table_data_wap: st.table(pd.DataFrame(table_data_wap).set_index("Ticker"))

else: 
    st.info("Inserisci parametri e avvia simulazione.")

st.sidebar.markdown("---"); st.sidebar.markdown("Kriterion Quant")
