# simulatore_pac/main.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta # Aggiunto timedelta

# Importazioni dai tuoi moduli utils
try:
    from utils.data_loader import load_historical_data_yf
    from utils.pac_engine import run_pac_simulation
    from utils.benchmark_engine import run_lump_sum_simulation
    from utils.performance import (
        get_total_capital_invested,
        get_final_portfolio_value,
        calculate_total_return_percentage,
        calculate_cagr,
        get_duration_years,
        calculate_portfolio_returns,
        calculate_annualized_volatility,
        calculate_sharpe_ratio,
        calculate_max_drawdown,
        calculate_drawdown_series,
        generate_cash_flows_for_xirr,
        calculate_xirr_metric,
        calculate_annual_returns,
        calculate_rolling_volatility,
        calculate_rolling_sharpe_ratio,
        calculate_rolling_cagr
    )
    IMPORT_SUCCESS = True
except ImportError as import_err:
    IMPORT_SUCCESS = False
    IMPORT_ERROR_MESSAGE = str(import_err)


st.set_page_config(page_title="Simulatore PAC Completo", layout="wide")
st.title("📘 Simulatore PAC Completo con Analisi Avanzate")
st.caption("Progetto Kriterion Quant")

if not IMPORT_SUCCESS:
    st.error(f"Errore critico durante l'importazione dei moduli utils: {IMPORT_ERROR_MESSAGE}")
    st.error("Assicurati che tutti i file .py dei moduli utils siano presenti e corretti nel repository GitHub e che tutte le dipendenze in requirements.txt siano installate correttamente su Streamlit Cloud.")
    st.stop()

# --- Sidebar per Input Utente ---
st.sidebar.header("Parametri della Simulazione")

st.sidebar.subheader("Asset e Allocazioni")
tickers_input_str = st.sidebar.text_input("Tickers (separati da virgola, es. AAPL,MSFT,GOOG)", "AAPL,GOOG,MSFT")
allocations_input_str = st.sidebar.text_input("Allocazioni % (separate da virgola, es. 60,20,20)", "60,20,20")

st.sidebar.subheader("Parametri PAC")
monthly_investment_input = st.sidebar.number_input("Importo Versamento Mensile (€/$)", min_value=10.0, value=200.0, step=10.0)
default_start_date_pac_sidebar = date(2020, 1, 1)
end_date_for_default_duration = default_start_date_pac_sidebar.replace(year=default_start_date_pac_sidebar.year + 3)
if end_date_for_default_duration > date.today():
    end_date_for_default_duration = date.today()
pac_start_date_input = st.sidebar.date_input("Data Inizio PAC", default_start_date_pac_sidebar)
default_duration_months = (end_date_for_default_duration.year - pac_start_date_input.year) * 12 + \
                          (end_date_for_default_duration.month - pac_start_date_input.month)
if default_duration_months < 6 : default_duration_months = 36
duration_months_input = st.sidebar.number_input("Durata PAC (in mesi)", min_value=6, value=default_duration_months, step=1)
reinvest_dividends_input = st.sidebar.checkbox("Reinvesti Dividendi (per PAC e Lump Sum)?", value=True)

st.sidebar.subheader("Ribilanciamento Periodico (Solo per PAC)")
rebalance_active_input = st.sidebar.checkbox("Attiva Ribilanciamento (per PAC)?", value=False)
rebalance_frequency_input_str = None
if rebalance_active_input:
    rebalance_frequency_input_str = st.sidebar.selectbox(
        "Frequenza Ribilanciamento", ["Annuale", "Semestrale", "Trimestrale"], index=0 )

st.sidebar.subheader("Parametri Metriche Avanzate")
risk_free_rate_input = st.sidebar.number_input("Tasso Risk-Free Annuale (%) per Sharpe", min_value=0.0, value=1.0, step=0.1, format="%.2f")
rolling_window_months_input = st.sidebar.number_input("Finestra Mobile per Rolling Metrics (mesi)", min_value=6, value=36, step=6)

run_simulation_button = st.sidebar.button("🚀 Avvia Simulazioni")

# --- Area Principale per Output ---
if run_simulation_button:
    st.write("--- DEBUG: Pulsante 'Avvia Simulazioni' PREMUTO ---")

    # --- VALIDAZIONE INPUT ---
    tickers_list = [ticker.strip().upper() for ticker in tickers_input_str.split(',') if ticker.strip()]
    error_in_input = False
    allocations_list_norm = []
    allocations_float_list_raw = [] 
    if not tickers_list: st.error("Errore: Devi inserire almeno un ticker."); error_in_input = True
    if not error_in_input:
        try:
            allocations_float_list_raw = [float(alloc.strip()) for alloc in allocations_input_str.split(',') if alloc.strip()]
            if len(tickers_list) != len(allocations_float_list_raw): st.error("Errore: Numero di ticker e allocazioni non corrispondono."); error_in_input = True
            elif not np.isclose(sum(allocations_float_list_raw), 100.0): st.error(f"Errore: Somma allocazioni ({sum(allocations_float_list_raw)}%) deve essere 100%."); error_in_input = True
            else: allocations_list_norm = [alloc / 100.0 for alloc in allocations_float_list_raw]
        except ValueError: st.error("Errore: Allocazioni devono essere numeri validi."); error_in_input = True
    if error_in_input: 
        st.write("--- DEBUG: Errore negli input, esecuzione interrotta. ---")
        st.stop() 

    st.write(f"--- DEBUG: Input validati. Tickers: {tickers_list}, Allocazioni normalizzate: {allocations_list_norm} ---")
    st.header(f"Risultati Simulazione per: {', '.join(tickers_list)}")
    if allocations_float_list_raw:
        alloc_display_list = [f"{tickers_list[i]}: {allocations_float_list_raw[i]}%" for i in range(len(tickers_list))]
        st.write(f"Allocazioni Target: {', '.join(alloc_display_list)}")
    if rebalance_active_input: st.write(f"Ribilanciamento Attivo (PAC): Sì, Frequenza: {rebalance_frequency_input_str}")
    else: st.write("Ribilanciamento Attivo (PAC): No")

    # --- CARICAMENTO DATI ---
    pac_start_date_dt = pd.to_datetime(pac_start_date_input)
    pac_start_date_str = pac_start_date_dt.strftime('%Y-%m-%d')
    actual_pac_end_date_dt = pac_start_date_dt + pd.DateOffset(months=duration_months_input)
    data_fetch_start_date_str = (pac_start_date_dt - pd.Timedelta(days=365*1)).strftime('%Y-%m-%d') # 1 anno prima
    data_fetch_end_date_str = datetime.today().strftime('%Y-%m-%d') # Scarica dati fino ad oggi

    historical_data_map = {}
    all_data_loaded_successfully = True
    min_data_available_until_ts = pd.Timestamp.max 
    
    for ticker_to_load in tickers_list:
        with st.spinner(f"Caricamento dati per {ticker_to_load}..."):
            data = load_historical_data_yf(ticker_to_load, data_fetch_start_date_str, data_fetch_end_date_str)
            if data.empty or data.index.min() > pac_start_date_dt or data.index.max() < (actual_pac_end_date_dt - pd.Timedelta(days=1)):
                st.error(f"Dati storici insufficienti per {ticker_to_load} per coprire l'intero periodo PAC (da {pac_start_date_dt.date()} a {actual_pac_end_date_dt.date()}). Controlla la disponibilità dei dati per questo ticker.")
                all_data_loaded_successfully = False; break
            historical_data_map[ticker_to_load] = data
            if data.index.max() < min_data_available_until_ts:
                min_data_available_until_ts = data.index.max()
    
    if not all_data_loaded_successfully: 
        st.write("--- DEBUG: Caricamento dati fallito. ---")
        st.stop()
    st.success("Dati storici caricati.")
    
    chart_display_end_date_dt = min(actual_pac_end_date_dt + pd.Timedelta(days=1), min_data_available_until_ts, pd.Timestamp(datetime.today()))
    st.write(f"--- DEBUG: min_data_available_until_ts: {min_data_available_until_ts.date()} ---")
    st.write(f"--- DEBUG: actual_pac_end_date_dt (fine PAC): {actual_pac_end_date_dt.date()} ---")
    st.write(f"--- DEBUG: chart_display_end_date_dt (fine visualizzazione grafici): {chart_display_end_date_dt.date()} ---")
    st.write("--- DEBUG: Dati storici caricati, prima della chiamata a run_pac_simulation ---")
    
    # --- ESECUZIONE SIMULAZIONI ---
    pac_total_df, asset_details_history_df = pd.DataFrame(), pd.DataFrame()
    try:
        with st.spinner("Esecuzione simulazione PAC..."):
            pac_total_df, asset_details_history_df = run_pac_simulation(
                historical_data_map=historical_data_map, tickers=tickers_list, allocations=allocations_list_norm,
                monthly_investment=monthly_investment_input, start_date_pac=pac_start_date_str,
                duration_months=duration_months_input, reinvest_dividends=reinvest_dividends_input,
                rebalance_active=rebalance_active_input, rebalance_frequency=rebalance_frequency_input_str )
        st.write("--- DEBUG: Chiamata a run_pac_simulation COMPLETATA ---")
    except Exception as e_pac: st.error(f"Errore CRITICO run_pac_simulation: {e_pac}"); import traceback; st.text(traceback.format_exc()); st.stop()

    st.write(f"--- DEBUG: `pac_total_df` ricevuto. Vuoto: {pac_total_df.empty}. Colonne: {pac_total_df.columns if not pac_total_df.empty else 'N/A'} ---")
    st.write(f"--- DEBUG: `asset_details_history_df` ricevuto. Vuoto: {asset_details_history_df.empty}. Colonne: {asset_details_history_df.columns if not asset_details_history_df.empty else 'N/A'} ---")
    if not asset_details_history_df.empty:
        st.write("--- DEBUG: Prime righe di `asset_details_history_df`: ---")
        st.dataframe(asset_details_history_df.head())


    lump_sum_df = pd.DataFrame()
    if not pac_total_df.empty and 'PortfolioValue' in pac_total_df.columns and len(pac_total_df) >= 2:
        total_invested_pac_for_ls = get_total_capital_invested(pac_total_df)
        if total_invested_pac_for_ls > 0:
             with st.spinner("Esecuzione simulazione Lump Sum..."):
                ls_sim_start_date = pd.to_datetime(pac_total_df['Date'].iloc[0])
                ls_sim_end_date = pd.to_datetime(pac_total_df['Date'].iloc[-1])
                lump_sum_df = run_lump_sum_simulation(
                    historical_data_map=historical_data_map, tickers=tickers_list, allocations=allocations_list_norm,
                    total_investment_lump_sum=total_invested_pac_for_ls, lump_sum_investment_date=pac_start_date_dt,
                    simulation_start_date=ls_sim_start_date, simulation_end_date=ls_sim_end_date,
                    reinvest_dividends=reinvest_dividends_input)
                if not lump_sum_df.empty: st.success("Simulazione Lump Sum completata.")
                else: st.warning("Simulazione Lump Sum non ha prodotto risultati o è vuota.")
    
    # --- CONTROLLO RISULTATI E VISUALIZZAZIONE ---
    if pac_total_df.empty or 'PortfolioValue' not in pac_total_df.columns or len(pac_total_df) < 2:
        st.error("Simulazione PAC non ha prodotto risultati sufficienti per l'analisi dettagliata.")
    else:
        st.success("Simulazioni completate. Inizio elaborazione output.")
        
        def calculate_metrics_for_strategy(sim_df, strategy_name, total_invested_override=None, is_pac=False):
            # ... (codice funzione helper calculate_metrics_for_strategy come prima) ...
            st.write(f"--- DEBUG: calculate_metrics_for_strategy chiamata per {strategy_name} ---")
            metrics = {}
            if sim_df.empty or 'PortfolioValue' not in sim_df.columns or len(sim_df) < 2:
                keys = ["Capitale Investito", "Valore Finale", "Rend. Totale", "CAGR", "XIRR", "Volatilità Ann.", "Sharpe", "Max Drawdown"]
                return {k: "N/A" for k in keys}
            actual_total_invested = total_invested_override if total_invested_override is not None else get_total_capital_invested(sim_df)
            metrics["Capitale Investito"] = f"{actual_total_invested:,.2f}"
            final_portfolio_val = get_final_portfolio_value(sim_df)
            metrics["Valore Finale"] = f"{final_portfolio_val:,.2f}"
            metrics["Rend. Totale"] = f"{calculate_total_return_percentage(final_portfolio_val, actual_total_invested):.2f}%"
            duration_yrs_strat = get_duration_years(sim_df.copy()) 
            cagr_strat = calculate_cagr(final_portfolio_val, actual_total_invested, duration_yrs_strat)
            metrics["CAGR"] = f"{cagr_strat:.2f}%" if pd.notna(cagr_strat) else "N/A"
            returns_strat = calculate_portfolio_returns(sim_df.copy())
            vol_strat = calculate_annualized_volatility(returns_strat)
            sharpe_strat = calculate_sharpe_ratio(returns_strat, risk_free_rate_annual=(risk_free_rate_input/100.0))
            mdd_strat = calculate_max_drawdown(sim_df.copy())
            metrics["Volatilità Ann."] = f"{vol_strat:.2f}%" if pd.notna(vol_strat) else "N/A"
            metrics["Sharpe"] = f"{sharpe_strat:.2f}" if pd.notna(sharpe_strat) else "N/A"
            metrics["Max Drawdown"] = f"{mdd_strat:.2f}%" if pd.notna(mdd_strat) else "N/A"
            if is_pac:
                xirr_dates, xirr_values = generate_cash_flows_for_xirr(sim_df, pac_start_date_str, duration_months_input, monthly_investment_input, final_portfolio_val)
                xirr_val = calculate_xirr_metric(xirr_dates, xirr_values)
                metrics["XIRR"] = f"{xirr_val:.2f}%" if pd.notna(xirr_val) else "N/A"
            else: 
                metrics["XIRR"] = metrics["CAGR"] 
            return metrics

        metrics_pac = calculate_metrics_for_strategy(pac_total_df, "PAC", is_pac=True)
        st.write(f"--- DEBUG: Metriche PAC: {metrics_pac} ---")
        display_data_metrics = {"Metrica": list(metrics_pac.keys()), "PAC": list(metrics_pac.values())}
        if not lump_sum_df.empty:
            total_invested_val_pac = get_total_capital_invested(pac_total_df)
            metrics_ls = calculate_metrics_for_strategy(lump_sum_df, "Lump Sum", total_invested_override=total_invested_val_pac, is_pac=False)
            st.write(f"--- DEBUG: Metriche LS: {metrics_ls} ---")
            if metrics_ls and metrics_pac:
                 ls_values_aligned = [metrics_ls.get(key, "N/A") for key in metrics_pac.keys()]
                 display_data_metrics["Lump Sum"] = ls_values_aligned
        
        st.subheader("Metriche di Performance Riepilogative")
        if display_data_metrics.get("Metrica"):
            df_metrics_to_display = pd.DataFrame(display_data_metrics)
            if not df_metrics_to_display.empty:
                st.table(df_metrics_to_display.set_index("Metrica"))
                st.write("--- DEBUG: Tabella Metriche Riepilogative DOVREBBE ESSERE VISUALIZZATA ---")
            else: st.warning("DataFrame metriche riepilogative vuoto.")
        else: st.warning("Dati metriche riepilogative non pronti.")

        # --- GRAFICO EQUITY LINE (CON ESTENSIONE DATA E CASH BENCHMARK) ---
        st.subheader("Andamento Comparativo del Portafoglio")
        # Creiamo un DataFrame con un range di date completo per l'asse X
        # chart_display_end_date_dt è definita dopo il caricamento dati
        full_date_range_for_charts = pd.date_range(start=pac_start_date_dt, end=chart_display_end_date_dt, freq='B')
        combined_equity_df = pd.DataFrame(index=full_date_range_for_charts)
        combined_equity_df.index.name = 'Date'

        if not pac_total_df.empty and 'Date' in pac_total_df.columns:
            pac_plot_data = pac_total_df.set_index(pd.to_datetime(pac_total_df['Date']))
            combined_equity_df['PAC Valore Portafoglio'] = pac_plot_data['PortfolioValue']
            combined_equity_df['PAC Capitale Investito'] = pac_plot_data['InvestedCapital']
        
        if not lump_sum_df.empty and 'Date' in lump_sum_df.columns:
            ls_plot_data = lump_sum_df.set_index(pd.to_datetime(lump_sum_df['Date']))
            combined_equity_df['Lump Sum Valore Portafoglio'] = ls_plot_data['PortfolioValue']

        total_capital_for_cash_bm = get_total_capital_invested(pac_total_df) if not pac_total_df.empty else 0
        if total_capital_for_cash_bm > 0 :
            combined_equity_df['Cash (Valore Fisso 0%)'] = total_capital_for_cash_bm
        
        cols_to_ffill_equity = ['PAC Valore Portafoglio', 'PAC Capitale Investito', 'Lump Sum Valore Portafoglio', 'Cash (Valore Fisso 0%)']
        for col in cols_to_ffill_equity:
            if col in combined_equity_df.columns:
                combined_equity_df[col] = combined_equity_df[col].ffill()
        
        actual_cols_to_plot_equity = [col for col in cols_to_ffill_equity if col in combined_equity_df.columns and not combined_equity_df[col].isnull().all()]
        if actual_cols_to_plot_equity:
            st.line_chart(combined_equity_df[actual_cols_to_plot_equity])
            st.write("--- DEBUG: Grafico Equity Line DOVREBBE ESSERE VISUALIZZATO ---")
        else:
            st.warning("Dati insufficienti per il grafico equity comparativo.")

        # GRAFICO DRAWDOWN (come prima)
        st.subheader("Andamento del Drawdown nel Tempo")
        # ... (codice grafico drawdown come prima) ...
        drawdown_data_to_plot = {}
        if not pac_total_df.empty and 'PortfolioValue' in pac_total_df and 'Date' in pac_total_df:
            pac_portfolio_values = pac_total_df.set_index(pd.to_datetime(pac_total_df['Date']))['PortfolioValue']
            pac_drawdown_series = calculate_drawdown_series(pac_portfolio_values)
            if not pac_drawdown_series.empty: drawdown_data_to_plot['PAC Drawdown (%)'] = pac_drawdown_series
        if not lump_sum_df.empty and 'PortfolioValue' in lump_sum_df and 'Date' in lump_sum_df:
            ls_portfolio_values = lump_sum_df.set_index(pd.to_datetime(lump_sum_df['Date']))['PortfolioValue']
            ls_drawdown_series = calculate_drawdown_series(ls_portfolio_values)
            if not ls_drawdown_series.empty: drawdown_data_to_plot['Lump Sum Drawdown (%)'] = ls_drawdown_series
        if drawdown_data_to_plot:
            combined_drawdown_df_plot = pd.DataFrame(drawdown_data_to_plot) 
            if not combined_drawdown_df_plot.empty: st.line_chart(combined_drawdown_df_plot); st.write("--- DEBUG: Grafico Drawdown DOVREBBE ESSERE VISUALIZZATO ---")
            else: st.warning("Dati non sufficienti per il grafico drawdown.")
        else: st.info("Dati non sufficienti per calcolare la serie di drawdown.")


        # ISTOGRAMMA RENDIMENTI ANNUALI (come prima)
        st.subheader("Istogramma Rendimenti Annuali (%)")
        # ... (codice istogramma come prima) ...
        data_for_annual_hist = {}
        if not pac_total_df.empty and 'PortfolioValue' in pac_total_df and 'Date' in pac_total_df:
            pac_pv_series_for_annual = pac_total_df.set_index(pd.to_datetime(pac_total_df['Date']))['PortfolioValue']
            annual_returns_pac = calculate_annual_returns(pac_pv_series_for_annual)
            if not annual_returns_pac.empty:
                annual_returns_pac.index = annual_returns_pac.index.year 
                data_for_annual_hist["PAC"] = annual_returns_pac
        if not lump_sum_df.empty and 'PortfolioValue' in lump_sum_df and 'Date' in lump_sum_df:
            ls_pv_series_for_annual = lump_sum_df.set_index(pd.to_datetime(lump_sum_df['Date']))['PortfolioValue']
            annual_returns_ls = calculate_annual_returns(ls_pv_series_for_annual)
            if not annual_returns_ls.empty:
                annual_returns_ls.index = annual_returns_ls.index.year
                data_for_annual_hist["Lump Sum"] = annual_returns_ls
        if data_for_annual_hist:
            annual_hist_df = pd.DataFrame(data_for_annual_hist)
            annual_hist_df.dropna(how='all', inplace=True) 
            if not annual_hist_df.empty: st.bar_chart(annual_hist_df); st.write("--- DEBUG: Istogramma Rendimenti Annuali DOVREBBE ESSERE VISUALIZZATO ---")
            else: st.warning("Dati rendimenti annuali non sufficienti per istogramma.")
        else: st.warning("Non ci sono dati per istogramma rendimenti annuali.")


        # ROLLING METRICS (come prima)
        st.subheader(f"Analisi Rolling Metrics per PAC (Finestra: {rolling_window_months_input} mesi)")
        # ... (codice rolling metrics come prima) ...
        pac_portfolio_values_series_rm = pac_total_df.set_index(pd.to_datetime(pac_total_df['Date']))['PortfolioValue']
        pac_daily_returns_rm = calculate_portfolio_returns(pac_total_df.copy())
        approx_trading_days_per_month = 21 
        rolling_window_days = rolling_window_months_input * approx_trading_days_per_month
        if len(pac_daily_returns_rm) >= rolling_window_days and len(pac_portfolio_values_series_rm) >= rolling_window_days:
            st.write("--- DEBUG: Inizio calcolo Rolling Metrics ---")
            rolling_vol = calculate_rolling_volatility(pac_daily_returns_rm, window_days=rolling_window_days)
            if not rolling_vol.empty: st.markdown("##### Volatilità Annualizzata Mobile (%)"); st.line_chart(rolling_vol)
            rolling_sharpe = calculate_rolling_sharpe_ratio(pac_daily_returns_rm, window_days=rolling_window_days, risk_free_rate_annual=(risk_free_rate_input / 100.0))
            if not rolling_sharpe.empty: st.markdown("##### Sharpe Ratio Annualizzato Mobile"); st.line_chart(rolling_sharpe)
            rolling_cagr_values = calculate_rolling_cagr(pac_portfolio_values_series_rm, window_days=rolling_window_days)
            if not rolling_cagr_values.empty: st.markdown("##### CAGR Mobile (%)"); st.line_chart(rolling_cagr_values)
            st.write("--- DEBUG: Grafici Rolling Metrics DOVREBBERO ESSERE VISUALIZZATI ---")
        else: st.warning(f"Dati storici insufficienti per rolling metrics con finestra di {rolling_window_months_input} mesi.")


        # STACKED AREA CHART e TABELLE QUOTE/WAP
        st.write("--- DEBUG: Verifica dati per Stacked Area e Tabelle Quote/WAP ---")
        if asset_details_history_df is not None and not asset_details_history_df.empty:
            st.write(f"--- DEBUG: `asset_details_history_df` per Stacked Area ha {len(asset_details_history_df)} righe. Prime righe: ---")
            st.dataframe(asset_details_history_df.head(2)) # Mostra solo le prime 2 per brevità
            
            st.subheader("Allocazione Dinamica del Portafoglio PAC nel Tempo (Valore per Asset)")
            value_cols_to_plot_stacked = [f'{ticker}_value' for ticker in tickers_list if f'{ticker}_value' in asset_details_history_df.columns]
            st.write(f"--- DEBUG: Colonne per Stacked Area: {value_cols_to_plot_stacked} ---")
            if value_cols_to_plot_stacked:
                stacked_area_df_data = asset_details_history_df.copy()
                if 'Date' in stacked_area_df_data.columns:
                    stacked_area_df_data['Date'] = pd.to_datetime(stacked_area_df_data['Date'])
                    stacked_area_df_data = stacked_area_df_data.set_index('Date')
                actual_cols_in_stacked_df = [col for col in value_cols_to_plot_stacked if col in stacked_area_df_data.columns]
                if actual_cols_in_stacked_df: 
                    st.area_chart(stacked_area_df_data[actual_cols_in_stacked_df])
                    st.write("--- DEBUG: Stacked Area Chart DOVREBBE ESSERE VISUALIZZATO ---")
                else: st.warning("Nessuna colonna di valore per asset trovata per lo stacked area chart.")
            else: st.warning("Dati sui valori per asset non sufficienti per grafico allocazione.")

            st.subheader("Dettagli Finali per Asset nel PAC")
            # ... (logica tabella WAP come prima)
            final_asset_details_list_for_table = []
            last_day_asset_details = asset_details_history_df.iloc[-1] 
            for ticker_idx, ticker_name in enumerate(tickers_list):
                shares_col_name = f'{ticker_name}_shares'; capital_col_name = f'{ticker_name}_capital_invested'
                final_shares = last_day_asset_details.get(shares_col_name, 0.0)
                total_capital_for_asset = last_day_asset_details.get(capital_col_name, 0.0)
                wap = np.nan
                if final_shares > 1e-6 and total_capital_for_asset > 1e-6: wap = total_capital_for_asset / final_shares
                elif final_shares > 1e-6 and total_capital_for_asset <= 1e-6: wap = 0.0
                final_asset_details_list_for_table.append({
                    "Ticker": ticker_name, "Quote Finali": f"{final_shares:.4f}",
                    "Capitale Investito (Asset)": f"{total_capital_for_asset:,.2f}",
                    "Prezzo Medio Carico (WAP)": f"{wap:,.2f}" if pd.notna(wap) else "N/A" })
            if final_asset_details_list_for_table:
                details_summary_df = pd.DataFrame(final_asset_details_list_for_table)
                st.table(details_summary_df.set_index("Ticker"))
                st.write("--- DEBUG: Tabella Quote/WAP DOVREBBE ESSERE VISUALIZZATA ---")
        else: 
            st.warning("Dati storici dettagliati per asset (asset_details_history_df) non disponibili per Stacked Area Chart e Tabella Quote/WAP.")
        
        # CHECKBOX DATI DETTAGLIATI
        # ... (codice checkbox come prima) ...
        if st.checkbox("Mostra dati aggregati dettagliati del PAC", key="pac_total_data_detail_main_v4"):
            st.dataframe(pac_total_df)
        if asset_details_history_df is not None and not asset_details_history_df.empty and \
           st.checkbox("Mostra dati storici dettagliati per asset del PAC", key="pac_asset_data_detail_main_v4"):
            st.dataframe(asset_details_history_df)
        if not lump_sum_df.empty and \
           st.checkbox("Mostra dati dettagliati della simulazione Lump Sum", key="ls_data_detail_main_key_v4"):
            st.dataframe(lump_sum_df)
else: 
    st.info("Inserisci i parametri nella sidebar a sinistra e avvia la simulazione.")
    st.write("--- DEBUG: Pagina iniziale, pulsante non premuto ---")

st.sidebar.markdown("---")
st.sidebar.markdown("Progetto Didattico Kriterion Quant")
