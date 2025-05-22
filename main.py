# simulatore_pac/main.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

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

st.set_page_config(page_title="Simulatore PAC Debug V3", layout="wide") # Changed title for clarity
st.title("📘 Simulatore PAC con Debug Avanzato V3")
st.caption("Progetto Kriterion Quant")

if not IMPORT_SUCCESS:
    st.error(f"Errore critico import moduli utils: {IMPORT_ERROR_MESSAGE}")
    st.stop()

# --- Sidebar ---
st.sidebar.header("Parametri Simulazione")
# ... (Sidebar inputs remain exactly the same as the last full version I gave you)
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
    if error_in_input: st.write("--- DEBUG: Errore negli input ---"); st.stop()
    st.write(f"--- DEBUG: Input OK. Tickers: {tickers_list} ---")

    # --- PREPARAZIONE DATE ---
    # Moved date definitions to be available before the loop
    pac_start_date_dt = pd.to_datetime(pac_start_date_input)
    pac_start_date_str = pac_start_date_dt.strftime('%Y-%m-%d') # String for pac_engine cashflow gen
    actual_pac_contribution_end_date_dt = pac_start_date_dt + pd.DateOffset(months=duration_months_input)
    
    data_download_start_str = (pac_start_date_dt - pd.Timedelta(days=365*1)).strftime('%Y-%m-%d')
    data_download_end_str = datetime.today().strftime('%Y-%m-%d')

    st.write(f"--- DEBUG: `pac_contribution_start_dt`: {pac_start_date_dt.date()} ---")
    st.write(f"--- DEBUG: `actual_pac_contribution_end_date_dt`: {actual_pac_contribution_end_date_dt.date()} ---")
    st.write(f"--- DEBUG: Download dati da: {data_download_start_str} a: {data_download_end_str} ---")

    # --- CARICAMENTO DATI ---
    historical_data_map = {}
    all_data_loaded_successfully = True
    latest_overall_data_date_ts = pd.Timestamp.min 
    
    for tkr in tickers_list:
        with st.spinner(f"Dati per {tkr}..."): 
            # Ensure data_download_start_str and data_download_end_str are used
            data = load_historical_data_yf(tkr, data_download_start_str, data_download_end_str)
        
        if not data.empty:
            st.write(f"--- DEBUG (Data Loader Output for {tkr}): Dati da {data.index.min().date()} a {data.index.max().date()} ---")
        else:
            st.write(f"--- DEBUG (Data Loader Output for {tkr}): DataFrame VUOTO restituito. ---")

        if data.empty or data.index.min() > pac_start_date_dt or \
           data.index.max() < (actual_pac_contribution_end_date_dt - pd.Timedelta(days=1)):
            st.error(f"Dati storici insufficienti per {tkr} per coprire almeno il periodo di contribuzione PAC (fino a {actual_pac_contribution_end_date_dt.date()}). Ultima data per {tkr}: {data.index.max().date() if not data.empty else 'N/A'}.")
            all_data_loaded_successfully = False; break
        historical_data_map[tkr] = data
        if data.index.max() > latest_overall_data_date_ts:
             latest_overall_data_date_ts = data.index.max()
    
    if not all_data_loaded_successfully: st.write("--- DEBUG: Caricamento dati fallito ---"); st.stop()
    st.success("Dati storici OK.")
    
    simulation_actual_end_dt = min(latest_overall_data_date_ts, pd.Timestamp(datetime.today()))
    st.write(f"--- DEBUG: `latest_overall_data_date_ts`: {latest_overall_data_date_ts.date()} ---")
    st.write(f"--- DEBUG: `simulation_actual_end_dt` (fine simulazioni e grafici): {simulation_actual_end_dt.date()} ---")

    base_chart_date_index = pd.DatetimeIndex([])
    chart_axis_start_dt = pac_start_date_dt # Corrected variable name here
    if chart_axis_start_dt <= simulation_actual_end_dt: # Use simulation_actual_end_dt
        base_chart_date_index = pd.date_range(start=chart_axis_start_dt, end=simulation_actual_end_dt, freq='B')
        base_chart_date_index.name = 'Date'
        if not base_chart_date_index.empty: st.write(f"--- DEBUG: `base_chart_date_index` creato: da {base_chart_date_index.min().date()} a {base_chart_date_index.max().date()} ---")
    else: st.write(f"--- DEBUG: Data inizio grafici ({chart_axis_start_dt.date()}) > fine simulazione ({simulation_actual_end_dt.date()}). ---")
    
    st.write("--- DEBUG: Dati storici caricati, prima di `run_pac_simulation` ---")
    pac_total_df, asset_details_history_df = pd.DataFrame(), pd.DataFrame()
    try:
        with st.spinner("Simulazione PAC..."): 
            pac_total_df, asset_details_history_df = run_pac_simulation(
                historical_data_map, tickers_list, allocations_list_norm, 
                monthly_investment_input, pac_start_date_str, 
                duration_months_input, # Renamed this for clarity, was duration_months_contributions_input
                reinvest_dividends_input, rebalance_active_input, rebalance_frequency_input_str
            )
        st.write("--- DEBUG: `run_pac_simulation` COMPLETATA ---")
    except Exception as e: st.error(f"Errore CRITICO PAC: {e}"); import traceback; st.text(traceback.format_exc()); st.stop()
    
    pac_total_ok_debug = not pac_total_df.empty
    asset_details_ok_debug = not asset_details_history_df.empty
    st.write(f"--- DEBUG: `pac_total_df` OK: {pac_total_ok_debug}, `asset_details_history_df` OK: {asset_details_ok_debug} ---")
    # ... (The rest of your main.py for simulations, metrics, and charts remains the same as the last full version)
    # Make sure to use 'duration_months_input' where 'duration_months_contributions_input' was used for XIRR cashflow generation
    # and 'pac_start_date_str' for the XIRR cashflow start.

    # ... (Rest of the file from the previous complete version, including:
    #      Lump Sum simulation logic
    #      calculate_metrics_for_strategy function
    #      Metrics table display
    #      Equity line chart display (using base_chart_date_index)
    #      Drawdown chart display (using base_chart_date_index)
    #      Annual returns histogram display
    #      Rolling metrics display
    #      Stacked area chart display (using base_chart_date_index)
    #      WAP/Final Shares table display
    #      Checkboxes for detailed data
    #      Final else block and sidebar footers)
    # For brevity, I'm not repeating the ~200 lines of chart and table display code here,
    # but they should follow directly from the last complete version I provided.
    # The key changes are at the top of this 'if run_simulation_button:' block.

    # --- (Ensure this part is copied from the previous full version) ---
    if pac_total_df.empty or 'PortfolioValue' not in pac_total_df.columns or len(pac_total_df)<2: st.error("Simulazione PAC fallita."); st.stop()
    st.success("Simulazioni OK. Output:")
    
    def calculate_metrics_for_strategy(sim_df, strategy_name, total_invested_override=None, is_pac=False):
        # ... (FULL HELPER FUNCTION AS BEFORE) ...
        st.write(f"--- DEBUG: Metriche per {strategy_name} ---")
        metrics = {}; keys = ["Capitale Investito", "Valore Finale", "Rend. Totale", "CAGR", "XIRR", "Volatilità Ann.", "Sharpe", "Max Drawdown"]
        if sim_df.empty or 'PortfolioValue' not in sim_df.columns or len(sim_df) < 2: return {k: "N/A" for k in keys}
        actual_total_invested = total_invested_override if total_invested_override is not None else get_total_capital_invested(sim_df)
        metrics["Capitale Investito"] = f"{actual_total_invested:,.2f}"; final_portfolio_val = get_final_portfolio_value(sim_df)
        metrics["Valore Finale"] = f"{final_portfolio_val:,.2f}"; metrics["Rend. Totale"] = f"{calculate_total_return_percentage(final_portfolio_val, actual_total_invested):.2f}%"
        duration_yrs_strat = get_duration_years(sim_df.copy()); cagr_strat = calculate_cagr(final_portfolio_val, actual_total_invested, duration_yrs_strat)
        metrics["CAGR"] = f"{cagr_strat:.2f}%" if pd.notna(cagr_strat) else "N/A"; returns_strat = calculate_portfolio_returns(sim_df.copy())
        vol_strat = calculate_annualized_volatility(returns_strat); sharpe_strat = calculate_sharpe_ratio(returns_strat, risk_free_rate_annual=(risk_free_rate_input/100.0))
        mdd_strat = calculate_max_drawdown(sim_df.copy()); metrics["Volatilità Ann."] = f"{vol_strat:.2f}%" if pd.notna(vol_strat) else "N/A"
        metrics["Sharpe"] = f"{sharpe_strat:.2f}" if pd.notna(sharpe_strat) else "N/A"; metrics["Max Drawdown"] = f"{mdd_strat:.2f}%" if pd.notna(mdd_strat) else "N/A"
        if is_pac: # Use pac_start_date_str and duration_months_input for XIRR cash flows
            xirr_dates, xirr_values = generate_cash_flows_for_xirr(sim_df, pac_start_date_str, duration_months_input, monthly_investment_input, final_portfolio_val)
            xirr_val = calculate_xirr_metric(xirr_dates, xirr_values); metrics["XIRR"] = f"{xirr_val:.2f}%" if pd.notna(xirr_val) else "N/A"
        else: metrics["XIRR"] = metrics["CAGR"] 
        return metrics

    metrics_pac = calculate_metrics_for_strategy(pac_total_df, "PAC", is_pac=True)
    display_data_metrics = {"Metrica": list(metrics_pac.keys()), "PAC": list(metrics_pac.values())}
    if not lump_sum_df.empty:
        metrics_ls = calculate_metrics_for_strategy(lump_sum_df, "Lump Sum", total_invested_override=get_total_capital_invested(pac_total_df), is_pac=False)
        if metrics_ls and metrics_pac: ls_values_aligned = [metrics_ls.get(key, "N/A") for key in metrics_pac.keys()]; display_data_metrics["Lump Sum"] = ls_values_aligned
    st.subheader("Metriche di Performance Riepilogative")
    if display_data_metrics.get("Metrica"):
        df_metrics_display = pd.DataFrame(display_data_metrics)
        if not df_metrics_display.empty: st.table(df_metrics_display.set_index("Metrica")); st.write("--- DEBUG: Tabella Metriche VISUALIZZATA ---")
    
    st.subheader("Andamento Comparativo del Portafoglio")
    if not base_chart_date_index.empty:
        # ... (The rest of the equity chart, drawdown, annual returns, rolling metrics, stacked area, WAP table, and checkboxes as in the previous full version) ...
        # Ensure all these sections correctly use base_chart_date_index for reindexing and ffill where appropriate.
        # For brevity, I will assume these sections are correctly copied from the last known good state of these individual parts.
        # Equity Line
        combined_equity_plot_df = pd.DataFrame(index=base_chart_date_index)
        if not pac_total_df.empty: pac_plot = pac_total_df.set_index(pd.to_datetime(pac_total_df['Date'])); combined_equity_plot_df = combined_equity_plot_df.join(pac_plot[['PortfolioValue', 'InvestedCapital']]); combined_equity_plot_df.rename(columns={'PortfolioValue': 'PAC Valore Portafoglio', 'InvestedCapital': 'PAC Capitale Investito'}, inplace=True)
        if not lump_sum_df.empty: ls_plot = lump_sum_df.set_index(pd.to_datetime(lump_sum_df['Date'])); combined_equity_plot_df = combined_equity_plot_df.join(ls_plot[['PortfolioValue']]); combined_equity_plot_df.rename(columns={'PortfolioValue': 'Lump Sum Valore Portafoglio'}, inplace=True)
        cash_bm_val = get_total_capital_invested(pac_total_df) if not pac_total_df.empty else 0
        if cash_bm_val > 0: combined_equity_plot_df['Cash (Valore Fisso 0%)'] = cash_bm_val
        cols_ffill_equity = ['PAC Valore Portafoglio', 'Lump Sum Valore Portafoglio', 'Cash (Valore Fisso 0%)', 'PAC Capitale Investito']
        for col in cols_ffill_equity:
            if col in combined_equity_plot_df.columns: combined_equity_plot_df[col] = combined_equity_plot_df[col].ffill()
        if 'PAC Capitale Investito' in combined_equity_plot_df.columns and not pac_total_df.empty:
            combined_equity_plot_df['PAC Capitale Investito'] = combined_equity_plot_df['PAC Capitale Investito'].ffill() 
            last_pac_actual_date_contrib_end = pac_start_date_dt + pd.DateOffset(months=duration_months_input-1) # Approximate end of contribution
            # Find the closest date in index
            if not base_chart_date_index.empty:
                 actual_last_contrib_date_in_index = base_chart_date_index[base_chart_date_index.get_indexer([last_pac_actual_date_contrib_end], method='ffill')[0]]
                 last_known_invested_capital = combined_equity_plot_df.loc[actual_last_contrib_date_in_index, 'PAC Capitale Investito']
                 if pd.notna(last_known_invested_capital): combined_equity_plot_df.loc[combined_equity_plot_df.index > actual_last_contrib_date_in_index, 'PAC Capitale Investito'] = last_known_invested_capital
        actual_cols_to_plot_equity = [c for c in ['PAC Valore Portafoglio', 'PAC Capitale Investito', 'Lump Sum Valore Portafoglio', 'Cash (Valore Fisso 0%)'] if c in combined_equity_plot_df.columns and not combined_equity_plot_df[c].isnull().all()]
        if actual_cols_to_plot_equity: st.line_chart(combined_equity_plot_df[actual_cols_to_plot_equity]); st.write("--- DEBUG: Grafico Equity VISUALIZZATO ---")
        else: st.warning("Dati insuff. per grafico equity.")
    else: st.warning("Indice base per grafici non creato, grafico equity saltato.")

    # Drawdown Chart
    st.subheader("Andamento del Drawdown nel Tempo")
    if not base_chart_date_index.empty:
        drawdown_data_to_plot = {}
        if not pac_total_df.empty: pac_pv_dd = pac_total_df.set_index(pd.to_datetime(pac_total_df['Date']))['PortfolioValue']; pac_dd_series = calculate_drawdown_series(pac_pv_dd);
        if not pac_dd_series.empty: drawdown_data_to_plot['PAC Drawdown (%)'] = pac_dd_series
        if not lump_sum_df.empty: ls_pv_dd = lump_sum_df.set_index(pd.to_datetime(lump_sum_df['Date']))['PortfolioValue']; ls_dd_series = calculate_drawdown_series(ls_pv_dd);
        if not ls_dd_series.empty: drawdown_data_to_plot['Lump Sum Drawdown (%)'] = ls_dd_series
        if drawdown_data_to_plot:
            dd_plot_df_temp = pd.DataFrame(drawdown_data_to_plot); dd_plot_df = pd.DataFrame(index=base_chart_date_index)
            for col in dd_plot_df_temp.columns: dd_plot_df = dd_plot_df.join(dd_plot_df_temp[[col]], how='left'); dd_plot_df[col] = dd_plot_df[col].ffill()
            if not dd_plot_df.empty and not dd_plot_df.isnull().all().all(): st.line_chart(dd_plot_df); st.write("--- DEBUG: Grafico Drawdown VISUALIZZATO ---")
    else: st.warning("Indice base non creato, grafico drawdown saltato.")
    
    # Annual Returns Histogram
    st.subheader("Istogramma Rendimenti Annuali (%)")
    data_for_annual_hist = {}
    if not pac_total_df.empty: pac_pv_ah = pac_total_df.set_index(pd.to_datetime(pac_total_df['Date']))['PortfolioValue']; ar_pac = calculate_annual_returns(pac_pv_ah);
    if not ar_pac.empty: data_for_annual_hist["PAC"] = ar_pac # Index is already year
    if not lump_sum_df.empty: ls_pv_ah = lump_sum_df.set_index(pd.to_datetime(lump_sum_df['Date']))['PortfolioValue']; ar_ls = calculate_annual_returns(ls_pv_ah);
    if not ar_ls.empty: data_for_annual_hist["Lump Sum"] = ar_ls # Index is already year
    if data_for_annual_hist: ah_df = pd.DataFrame(data_for_annual_hist).dropna(how='all');
    if not ah_df.empty: st.bar_chart(ah_df); st.write("--- DEBUG: Istogramma Rend. Ann. VISUALIZZATO ---")

    # Rolling Metrics
    st.subheader(f"Analisi Rolling Metrics per PAC (Finestra: {rolling_window_months_input} mesi)")
    if not pac_total_df.empty:
        pac_pv_rm = pac_total_df.set_index(pd.to_datetime(pac_total_df['Date']))['PortfolioValue']; pac_dr_rm = calculate_portfolio_returns(pac_total_df.copy())
        roll_win_days = rolling_window_months_input * 21; st.write(f"--- DEBUG Rolling: DailyReturns len: {len(pac_dr_rm)}, WindowDays: {roll_win_days} ---")
        if len(pac_dr_rm) >= roll_win_days and len(pac_pv_rm) >= roll_win_days:
            roll_vol = calculate_rolling_volatility(pac_dr_rm, roll_win_days);
            if not roll_vol.empty: st.markdown("##### Volatilità Ann. Mobile (%)"); st.line_chart(roll_vol.reindex(base_chart_date_index).ffill())
            roll_sharpe = calculate_rolling_sharpe_ratio(pac_dr_rm, roll_win_days, (risk_free_rate_input/100.0));
            if not roll_sharpe.empty: st.markdown("##### Sharpe Ratio Ann. Mobile"); st.line_chart(roll_sharpe.reindex(base_chart_date_index).ffill())
            roll_cagr = calculate_rolling_cagr(pac_pv_rm, roll_win_days);
            if not roll_cagr.empty: st.markdown("##### CAGR Mobile (%)"); st.line_chart(roll_cagr.reindex(base_chart_date_index).ffill())
            st.write("--- DEBUG: Grafici Rolling VISUALIZZATI ---")
        else: st.warning(f"Dati insuff. per rolling metrics ({len(pac_dr_rm)} gg rend.) con finestra {rolling_window_months_input} mesi.")

    # Stacked Area & WAP/Shares Table
    st.write("--- DEBUG: Verifica dati per Stacked Area e Tabelle Quote/WAP ---")
    if asset_details_history_df is not None and not asset_details_history_df.empty and not base_chart_date_index.empty:
        st.subheader("Allocazione Dinamica Portafoglio PAC (Valore per Asset)")
        val_cols_stack = [f'{t}_value' for t in tickers_list if f'{t}_value' in asset_details_history_df.columns]
        if val_cols_stack:
            stack_df_data_temp = asset_details_history_df.set_index(pd.to_datetime(asset_details_history_df['Date']))[val_cols_stack]
            stack_df_data_reindexed = pd.DataFrame(index=base_chart_date_index)
            for col in stack_df_data_temp.columns: stack_df_data_reindexed=stack_df_data_reindexed.join(stack_df_data_temp[[col]],how='left'); stack_df_data_reindexed[col]=stack_df_data_reindexed[col].ffill().fillna(0)
            if not stack_df_data_reindexed.empty and not stack_df_data_reindexed.isnull().all().all(): st.area_chart(stack_df_data_reindexed); st.write("--- DEBUG: Stacked Area VISUALIZZATO ---")
        st.subheader("Dettagli Finali per Asset nel PAC")
        final_shares_map = get_final_shares_per_asset(asset_details_history_df, tickers_list)
        wap_map = calculate_wap_per_asset(asset_details_history_df, tickers_list)
        table_data_wap = []
        for ticker_wap in tickers_list:
            capital_invested_specific_asset = asset_details_history_df.iloc[-1].get(f'{ticker_wap}_capital_invested', 0.0)
            table_data_wap.append({"Ticker": ticker_wap, "Quote Finali": f"{final_shares_map.get(ticker_wap, 0.0):.4f}", "Capitale Investito (Asset)": f"{capital_invested_specific_asset:,.2f}", "Prezzo Medio Carico (WAP)": f"{wap_map.get(ticker_wap, np.nan):,.2f}" if pd.notna(wap_map.get(ticker_wap, np.nan)) else "N/A"})
        if table_data_wap: st.table(pd.DataFrame(table_data_wap).set_index("Ticker")); st.write("--- DEBUG: Tabella Quote/WAP VISUALIZZATA ---")
    else: st.warning("Dati dettagliati per asset o indice base non disponibili.")

else: 
    st.info("Inserisci parametri e avvia simulazione.")
    st.write("--- DEBUG: Pagina iniziale ---")
st.sidebar.markdown("---"); st.sidebar.markdown("Kriterion Quant")
