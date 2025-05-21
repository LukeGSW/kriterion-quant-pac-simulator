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

st.set_page_config(page_title="Simulatore PAC Debug", layout="wide")
st.title("📘 Simulatore PAC con Debug Avanzato")
st.caption("Progetto Kriterion Quant")

if not IMPORT_SUCCESS:
    st.error(f"Errore critico durante l'importazione dei moduli utils: {IMPORT_ERROR_MESSAGE}")
    st.error("Assicurati che tutti i file .py dei moduli utils siano presenti e corretti nel repository GitHub e che tutte le dipendenze in requirements.txt siano installate correttamente su Streamlit Cloud.")
    st.stop()

# --- Sidebar ---
st.sidebar.header("Parametri Simulazione")
st.sidebar.subheader("Asset e Allocazioni")
tickers_input_str = st.sidebar.text_input("Tickers (virgola sep.)", "AAPL,GOOG,MSFT")
allocations_input_str = st.sidebar.text_input("Allocazioni % (virgola sep.)", "60,20,20")
st.sidebar.subheader("Parametri PAC")
monthly_investment_input = st.sidebar.number_input("Importo Mensile (€/$)", 10.0, value=200.0, step=10.0)
default_start_date_pac_sidebar = date(2020, 1, 1)
pac_start_date_input = st.sidebar.date_input("Data Inizio PAC", default_start_date_pac_sidebar)
end_date_for_default_duration = default_start_date_pac_sidebar.replace(year=default_start_date_pac_sidebar.year + 3)
if end_date_for_default_duration > date.today(): end_date_for_default_duration = date.today()
default_duration_months = max(6, (end_date_for_default_duration.year - pac_start_date_input.year) * 12 + \
                          (end_date_for_default_duration.month - pac_start_date_input.month))
duration_months_input = st.sidebar.number_input("Durata PAC (mesi)", 6, value=default_duration_months, step=1)
reinvest_dividends_input = st.sidebar.checkbox("Reinvesti Dividendi?", True)
st.sidebar.subheader("Ribilanciamento PAC")
rebalance_active_input = st.sidebar.checkbox("Attiva Ribilanciamento?", False)
rebalance_frequency_input_str = None
if rebalance_active_input:
    rebalance_frequency_input_str = st.sidebar.selectbox("Frequenza", ["Annuale", "Semestrale", "Trimestrale"], 0)
st.sidebar.subheader("Metriche Avanzate")
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

    pac_start_date_dt = pd.to_datetime(pac_start_date_input)
    pac_start_date_str = pac_start_date_dt.strftime('%Y-%m-%d')
    actual_pac_end_date_dt = pac_start_date_dt + pd.DateOffset(months=duration_months_input)
    data_fetch_start_date_str = (pac_start_date_dt - pd.Timedelta(days=365*1)).strftime('%Y-%m-%d')
    data_fetch_end_date_str = datetime.today().strftime('%Y-%m-%d')
    historical_data_map = {}; all_data_loaded_successfully = True; min_data_available_until_ts = pd.Timestamp.max
    for tkr in tickers_list:
        with st.spinner(f"Dati per {tkr}..."): data = load_historical_data_yf(tkr, data_fetch_start_date_str, data_fetch_end_date_str)
        if data.empty or data.index.min()>pac_start_date_dt or data.index.max()<(actual_pac_end_date_dt-pd.Timedelta(days=1)):
            st.error(f"Dati insufficienti per {tkr} per periodo PAC."); all_data_loaded_successfully=False; break
        historical_data_map[tkr] = data
        if data.index.max()<min_data_available_until_ts: min_data_available_until_ts=data.index.max()
    if not all_data_loaded_successfully: st.write("--- DEBUG: Caricamento dati fallito ---"); st.stop()
    st.success("Dati storici OK.")
    chart_display_end_date_dt = min(actual_pac_end_date_dt+pd.Timedelta(days=1), min_data_available_until_ts, pd.Timestamp(datetime.today()))
    st.write(f"--- DEBUG: `min_data_available_until_ts`: {min_data_available_until_ts.date()}, `actual_pac_end_date_dt`: {actual_pac_end_date_dt.date()}, `chart_display_end_date_dt`: {chart_display_end_date_dt.date()} ---")
    st.write("--- DEBUG: Dati storici caricati, prima di `run_pac_simulation` ---")

    pac_total_df, asset_details_history_df = pd.DataFrame(), pd.DataFrame()
    try:
        with st.spinner("Simulazione PAC..."): pac_total_df, asset_details_history_df = run_pac_simulation(historical_data_map, tickers_list, allocations_list_norm, monthly_investment_input, pac_start_date_str, duration_months_input, reinvest_dividends_input, rebalance_active_input, rebalance_frequency_input_str)
        st.write("--- DEBUG: `run_pac_simulation` COMPLETATA ---")
    except Exception as e: st.error(f"Errore CRITICO PAC: {e}"); import traceback; st.text(traceback.format_exc()); st.stop()
    
    pac_total_ok_debug = not pac_total_df.empty
    asset_details_ok_debug = not asset_details_history_df.empty
    st.write(f"--- DEBUG: `pac_total_df` OK: {pac_total_ok_debug}, `asset_details_history_df` OK: {asset_details_ok_debug} ---")
    if not pac_total_df.empty:
        st.write(f"--- DEBUG: `pac_total_df` Colonne: {pac_total_df.columns.tolist()} ---")
    if not asset_details_history_df.empty:
        st.write(f"--- DEBUG: `asset_details_history_df` Colonne: {asset_details_history_df.columns.tolist()} ---")
        st.write("--- DEBUG `asset_details_history_df` HEAD ---")
        st.dataframe(asset_details_history_df.head(2))

    lump_sum_df = pd.DataFrame()
    if not pac_total_df.empty and 'PortfolioValue' in pac_total_df.columns and len(pac_total_df)>=2:
        total_invested_pac = get_total_capital_invested(pac_total_df)
        if total_invested_pac > 0:
            with st.spinner("Simulazione LS..."): lump_sum_df = run_lump_sum_simulation(historical_data_map, tickers_list, allocations_list_norm, total_invested_pac, pac_start_date_dt, pd.to_datetime(pac_total_df['Date'].iloc[0]), pd.to_datetime(pac_total_df['Date'].iloc[-1]), reinvest_dividends_input)
            if not lump_sum_df.empty: st.success("Simulazione LS OK.")
            else: st.warning("Simulazione LS vuota.")
    
    if pac_total_df.empty or 'PortfolioValue' not in pac_total_df.columns or len(pac_total_df)<2: st.error("Simulazione PAC fallita o dati insufficienti."); st.stop()
    st.success("Simulazioni OK. Elaborazione output.")

    def calculate_metrics_for_strategy(sim_df, strategy_name, total_invested_override=None, is_pac=False):
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
        if is_pac:
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
    st.write(f"--- DEBUG Equity: Inizio. `chart_display_end_date_dt`: {chart_display_end_date_dt.date()} ---")
    full_equity_date_range = pd.date_range(start=pac_start_date_dt, end=chart_display_end_date_dt, freq='B')
    combined_equity_plot_df = pd.DataFrame(index=full_equity_date_range); combined_equity_plot_df.index.name = 'Date'
    if not pac_total_df.empty:
        pac_plot = pac_total_df.set_index(pd.to_datetime(pac_total_df['Date']))
        combined_equity_plot_df['PAC Valore Portafoglio'] = pac_plot['PortfolioValue']
        combined_equity_plot_df['PAC Capitale Investito'] = pac_plot['InvestedCapital']
    if not lump_sum_df.empty:
        ls_plot = lump_sum_df.set_index(pd.to_datetime(lump_sum_df['Date']))
        combined_equity_plot_df['Lump Sum Valore Portafoglio'] = ls_plot['PortfolioValue']
    cash_bm_val = get_total_capital_invested(pac_total_df) if not pac_total_df.empty else 0
    if cash_bm_val > 0: combined_equity_plot_df['Cash (Valore Fisso 0%)'] = cash_bm_val
    cols_ffill_equity = ['PAC Valore Portafoglio', 'PAC Capitale Investito', 'Lump Sum Valore Portafoglio', 'Cash (Valore Fisso 0%)']
    for col in cols_ffill_equity:
        if col in combined_equity_plot_df.columns: combined_equity_plot_df[col] = combined_equity_plot_df[col].ffill()
    cols_plot_equity = [c for c in cols_ffill_equity if c in combined_equity_plot_df.columns and not combined_equity_plot_df[c].isnull().all()]
    if cols_plot_equity: st.line_chart(combined_equity_plot_df[cols_plot_equity]); st.write("--- DEBUG: Grafico Equity VISUALIZZATO ---")
    else: st.warning("Dati insufficienti per grafico equity.")

    st.subheader("Andamento del Drawdown nel Tempo")
    drawdown_data_to_plot = {}
    if not pac_total_df.empty:
        pac_pv_dd = pac_total_df.set_index(pd.to_datetime(pac_total_df['Date']))['PortfolioValue']
        pac_dd_series = calculate_drawdown_series(pac_pv_dd)
        if not pac_dd_series.empty: drawdown_data_to_plot['PAC Drawdown (%)'] = pac_dd_series
    if not lump_sum_df.empty:
        ls_pv_dd = lump_sum_df.set_index(pd.to_datetime(lump_sum_df['Date']))['PortfolioValue']
        ls_dd_series = calculate_drawdown_series(ls_pv_dd)
        if not ls_dd_series.empty: drawdown_data_to_plot['Lump Sum Drawdown (%)'] = ls_dd_series
    if drawdown_data_to_plot:
        dd_plot_df = pd.DataFrame(drawdown_data_to_plot)
        dd_plot_df = dd_plot_df.reindex(full_equity_date_range).ffill()
        if not dd_plot_df.empty: st.line_chart(dd_plot_df); st.write("--- DEBUG: Grafico Drawdown VISUALIZZATO ---")
        else: st.warning("DataFrame Drawdown vuoto dopo reindex/ffill.")
    else: st.info("Dati insuff. per drawdown.")

    st.subheader("Istogramma Rendimenti Annuali (%)")
    data_for_annual_hist = {}
    if not pac_total_df.empty:
        pac_pv_ah = pac_total_df.set_index(pd.to_datetime(pac_total_df['Date']))['PortfolioValue']
        ar_pac = calculate_annual_returns(pac_pv_ah)
        if not ar_pac.empty: ar_pac.index = ar_pac.index.year; data_for_annual_hist["PAC"] = ar_pac
    if not lump_sum_df.empty:
        ls_pv_ah = lump_sum_df.set_index(pd.to_datetime(lump_sum_df['Date']))['PortfolioValue']
        ar_ls = calculate_annual_returns(ls_pv_ah)
        if not ar_ls.empty: ar_ls.index = ar_ls.index.year; data_for_annual_hist["Lump Sum"] = ar_ls
    if data_for_annual_hist:
        ah_df = pd.DataFrame(data_for_annual_hist).dropna(how='all')
        if not ah_df.empty: st.bar_chart(ah_df); st.write("--- DEBUG: Istogramma Rend. Ann. VISUALIZZATO ---")
        else: st.warning("Dati per Istogramma Rend. Ann. vuoti dopo dropna.")
    else: st.warning("Dati insuff. per Istogramma Rend. Ann.")

    st.subheader(f"Analisi Rolling Metrics per PAC (Finestra: {rolling_window_months_input} mesi)")
    pac_pv_rm = pac_total_df.set_index(pd.to_datetime(pac_total_df['Date']))['PortfolioValue']
    pac_dr_rm = calculate_portfolio_returns(pac_total_df.copy())
    roll_win_days = rolling_window_months_input * 21
    st.write(f"--- DEBUG Rolling: DailyReturns len: {len(pac_dr_rm)}, WindowDays: {roll_win_days} ---")
    if len(pac_dr_rm) >= roll_win_days and len(pac_pv_rm) >= roll_win_days:
        st.write("--- DEBUG: Calcolo Rolling Metrics ---")
        roll_vol = calculate_rolling_volatility(pac_dr_rm, roll_win_days)
        if not roll_vol.empty: st.markdown("##### Volatilità Ann. Mobile (%)"); st.line_chart(roll_vol)
        roll_sharpe = calculate_rolling_sharpe_ratio(pac_dr_rm, roll_win_days, (risk_free_rate_input/100.0))
        if not roll_sharpe.empty: st.markdown("##### Sharpe Ratio Ann. Mobile"); st.line_chart(roll_sharpe)
        roll_cagr = calculate_rolling_cagr(pac_pv_rm, roll_win_days)
        if not roll_cagr.empty: st.markdown("##### CAGR Mobile (%)"); st.line_chart(roll_cagr)
        st.write("--- DEBUG: Grafici Rolling VISUALIZZATI (se dati non vuoti) ---")
    else: st.warning(f"Dati insuff. per rolling metrics ({len(pac_dr_rm)} gg rend.) con finestra {rolling_window_months_input} mesi.")

    st.write("--- DEBUG: Verifica dati per Stacked Area e Tabelle Quote/WAP ---")
    if asset_details_history_df is not None and not asset_details_history_df.empty:
        st.write(f"--- DEBUG: `asset_details_history_df` per Stacked Area ha {len(asset_details_history_df)} righe.")
        st.subheader("Allocazione Dinamica Portafoglio PAC (Valore per Asset)")
        val_cols_stack = [f'{t}_value' for t in tickers_list if f'{t}_value' in asset_details_history_df.columns]
        st.write(f"--- DEBUG: Colonne per Stacked Area: {val_cols_stack} ---")
        if val_cols_stack:
            stack_df_data = asset_details_history_df.copy()
            if 'Date' in stack_df_data.columns: stack_df_data['Date']=pd.to_datetime(stack_df_data['Date']); stack_df_data=stack_df_data.set_index('Date')
            actual_cols_stack = [c for c in val_cols_stack if c in stack_df_data.columns]
            if actual_cols_stack: 
                stack_df_data_reindexed = stack_df_data[actual_cols_stack].reindex(full_equity_date_range).ffill()
                st.area_chart(stack_df_data_reindexed)
                st.write("--- DEBUG: Stacked Area VISUALIZZATO ---")
            else: st.warning("Nessuna colonna di valore per Stacked Area.")
        else: st.warning("Dati per Stacked Area non sufficienti.")
        st.subheader("Dettagli Finali per Asset nel PAC")
        final_asset_details_list = []
        last_day_details = asset_details_history_df.iloc[-1]
        for tkr_name in tickers_list:
            s_col=f'{tkr_name}_shares'; c_col=f'{tkr_name}_capital_invested'
            s_final=last_day_details.get(s_col,0.0); c_total_asset=last_day_details.get(c_col,0.0)
            wap=np.nan
            if s_final > 1e-6 and c_total_asset > 1e-6: wap = c_total_asset / s_final
            elif s_final > 1e-6 and c_total_asset <= 1e-6: wap = 0.0
            final_asset_details_list.append({"Ticker":tkr_name, "Quote Finali":f"{s_final:.4f}", "Capitale Investito (Asset)":f"{c_total_asset:,.2f}", "Prezzo Medio Carico (WAP)":f"{wap:,.2f}" if pd.notna(wap) else "N/A"})
        if final_asset_details_list: st.table(pd.DataFrame(final_asset_details_list).set_index("Ticker")); st.write("--- DEBUG: Tabella Quote/WAP VISUALIZZATA ---")
    else: st.warning("Dati storici dettagliati per asset non disponibili.")
    
    if st.checkbox("Dati aggregati PAC", key="d1"): st.dataframe(pac_total_df)
    if asset_details_history_df is not None and not asset_details_history_df.empty and st.checkbox("Dati per asset PAC", key="d2"): st.dataframe(asset_details_history_df)
    if not lump_sum_df.empty and st.checkbox("Dati Lump Sum", key="d3"): st.dataframe(lump_sum_df)
else: 
    st.info("Inserisci parametri e avvia simulazione.")
    st.write("--- DEBUG: Pagina iniziale ---")
st.sidebar.markdown("---"); st.sidebar.markdown("Kriterion Quant")
