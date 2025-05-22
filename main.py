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
    st.error(f"Errore critico import moduli utils: {IMPORT_ERROR_MESSAGE}")
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
    historical_data_map = {}; all_data_loaded_successfully = True; 
    
    latest_overall_data_date_ts = pd.Timestamp.min 
    for tkr in tickers_list:
        with st.spinner(f"Dati per {tkr}..."): data = load_historical_data_yf(tkr, data_fetch_start_date_str, data_fetch_end_date_str)
        if data.empty or data.index.min()>pac_start_date_dt or data.index.max()<(actual_pac_end_date_dt-pd.Timedelta(days=1)):
            st.error(f"Dati insufficienti per {tkr} per periodo PAC (da {pac_start_date_dt.date()} a {actual_pac_end_date_dt.date()})."); all_data_loaded_successfully=False; break
        historical_data_map[tkr] = data
        if data.index.max() > latest_overall_data_date_ts:
             latest_overall_data_date_ts = data.index.max()
    
    if not all_data_loaded_successfully: st.write("--- DEBUG: Caricamento dati fallito ---"); st.stop()
    st.success("Dati storici OK.")
    
    # Data fino a cui vogliamo POTENZIALMENTE estendere i grafici
    potential_chart_extension_end_date = min(latest_overall_data_date_ts, pd.Timestamp(datetime.today()))
    st.write(f"--- DEBUG: `latest_overall_data_date_ts` (ultima data tra tutti i ticker): {latest_overall_data_date_ts.date()} ---")
    st.write(f"--- DEBUG: `actual_pac_end_date_dt` (fine investimenti PAC): {actual_pac_end_date_dt.date()} ---")
    st.write(f"--- DEBUG: `potential_chart_extension_end_date` (usata per asse X grafici): {potential_chart_extension_end_date.date()} ---")

    base_chart_date_index = pd.DatetimeIndex([])
    chart_start_date_for_axis = pac_start_date_dt 
    if chart_start_date_for_axis <= potential_chart_extension_end_date:
        base_chart_date_index = pd.date_range(start=chart_start_date_for_axis, end=potential_chart_extension_end_date, freq='B')
        base_chart_date_index.name = 'Date'
        if not base_chart_date_index.empty:
             st.write(f"--- DEBUG: `base_chart_date_index` per grafici estesi: da {base_chart_date_index.min().date()} a {base_chart_date_index.max().date()} ({len(base_chart_date_index)} punti) ---")
        else:
            st.write(f"--- DEBUG: `base_chart_date_index` è VUOTO dopo date_range. Controllare start/end date. ---")
    else:
        st.write(f"--- DEBUG: Data inizio grafici ({chart_start_date_for_axis.date()}) successiva a data fine estensione ({potential_chart_extension_end_date.date()}). L'asse X sarà limitato al periodo PAC. ---")
        # Fallback a un indice che copre solo il periodo PAC se l'estensione non è possibile o richiesta
        if not pac_total_df.empty and 'Date' in pac_total_df.columns: # Assumendo che pac_total_df sia già calcolato o che lo calcoleremo
             base_chart_date_index = pd.to_datetime(pac_total_df['Date']).unique()
             base_chart_date_index.name = 'Date'
             st.write(f"--- DEBUG: `base_chart_date_index` di fallback limitato al periodo PAC: {len(base_chart_date_index)} punti ---")


    st.write("--- DEBUG: Dati storici caricati, prima di `run_pac_simulation` ---")
    pac_total_df, asset_details_history_df = pd.DataFrame(), pd.DataFrame()
    try:
        with st.spinner("Simulazione PAC..."): pac_total_df, asset_details_history_df = run_pac_simulation(historical_data_map, tickers_list, allocations_list_norm, monthly_investment_input, pac_start_date_str, duration_months_input, reinvest_dividends_input, rebalance_active_input, rebalance_frequency_input_str)
        st.write("--- DEBUG: `run_pac_simulation` COMPLETATA ---")
    except Exception as e: st.error(f"Errore CRITICO PAC: {e}"); import traceback; st.text(traceback.format_exc()); st.stop()
    # ... (Stampe DEBUG per pac_total_df e asset_details_history_df come prima) ...
    
    lump_sum_df = pd.DataFrame() # Definizione Lump Sum DF
    # ... (Logica simulazione Lump Sum come prima) ...
    if not pac_total_df.empty and 'PortfolioValue' in pac_total_df.columns and len(pac_total_df)>=2:
        total_invested_pac = get_total_capital_invested(pac_total_df)
        if total_invested_pac > 0:
            with st.spinner("Simulazione LS..."): 
                ls_sim_start_date = pd.to_datetime(pac_total_df['Date'].iloc[0])
                ls_sim_end_date = pd.to_datetime(pac_total_df['Date'].iloc[-1]) # LS finisce quando finisce il PAC
                lump_sum_df = run_lump_sum_simulation(historical_data_map, tickers_list, allocations_list_norm, total_invested_pac, pac_start_date_dt, ls_sim_start_date, ls_sim_end_date, reinvest_dividends_input)


    if pac_total_df.empty or 'PortfolioValue' not in pac_total_df.columns or len(pac_total_df)<2: st.error("Simulazione PAC fallita o dati insufficienti."); st.stop()
    st.success("Simulazioni OK. Elaborazione output.")

    # --- TABELLA METRICHE (come prima) ---
    # ... (codice tabella metriche come prima) ...
    def calculate_metrics_for_strategy(sim_df, strategy_name, total_invested_override=None, is_pac=False):
        # ... (contenuto della funzione helper come prima) ...
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


    # --- GRAFICO EQUITY LINE ---
    st.subheader("Andamento Comparativo del Portafoglio")
    if not base_chart_date_index.empty:
        combined_equity_plot_df = pd.DataFrame(index=base_chart_date_index)

        if not pac_total_df.empty:
            pac_plot = pac_total_df.set_index(pd.to_datetime(pac_total_df['Date']))
            combined_equity_plot_df = combined_equity_plot_df.join(pac_plot[['PortfolioValue', 'InvestedCapital']])
            combined_equity_plot_df.rename(columns={'PortfolioValue': 'PAC Valore Portafoglio', 'InvestedCapital': 'PAC Capitale Investito'}, inplace=True)
        
        if not lump_sum_df.empty:
            ls_plot = lump_sum_df.set_index(pd.to_datetime(lump_sum_df['Date']))
            combined_equity_plot_df = combined_equity_plot_df.join(ls_plot[['PortfolioValue']])
            combined_equity_plot_df.rename(columns={'PortfolioValue': 'Lump Sum Valore Portafoglio'}, inplace=True)
        
        cash_bm_val = get_total_capital_invested(pac_total_df) if not pac_total_df.empty else 0
        if cash_bm_val > 0: combined_equity_plot_df['Cash (Valore Fisso 0%)'] = cash_bm_val
        
        # Forward fill per estendere le linee
        cols_to_ffill_equity = ['PAC Valore Portafoglio', 'Lump Sum Valore Portafoglio', 'Cash (Valore Fisso 0%)']
        for col in cols_to_ffill_equity:
            if col in combined_equity_plot_df.columns: combined_equity_plot_df[col] = combined_equity_plot_df[col].ffill()

        # Gestione speciale per PAC Capitale Investito
        if 'PAC Capitale Investito' in combined_equity_plot_df.columns and not pac_total_df.empty:
            # Ffill iniziale per coprire i buchi durante il PAC
            combined_equity_plot_df['PAC Capitale Investito'] = combined_equity_plot_df['PAC Capitale Investito'].ffill()
            # Poi, mantieni l'ultimo valore costante dopo la fine del PAC
            last_pac_actual_date = pd.to_datetime(pac_total_df['Date'].iloc[-1])
            # Trova l'indice più vicino a last_pac_actual_date in base_chart_date_index
            if last_pac_actual_date in base_chart_date_index:
                actual_last_pac_date_in_index = last_pac_actual_date
            else: # Cerca il precedente più vicino se la data esatta non è nell'indice Bday
                try:
                    idx_loc = base_chart_date_index.get_indexer([last_pac_actual_date], method='ffill')[0]
                    # Se idx_loc è -1 (tutte le date dell'indice sono maggiori), usa il primo indice
                    actual_last_pac_date_in_index = base_chart_date_index[idx_loc if idx_loc != -1 else 0]
                except: # Fallback se get_indexer fallisce o l'indice è strano
                    actual_last_pac_date_in_index = base_chart_date_index[-1] if not base_chart_date_index.empty else None
            
            if actual_last_pac_date_in_index is not None:
                 last_known_invested_capital = combined_equity_plot_df.loc[actual_last_pac_date_in_index, 'PAC Capitale Investito']
                 if pd.notna(last_known_invested_capital):
                     combined_equity_plot_df.loc[combined_equity_plot_df.index > actual_last_pac_date_in_index, 'PAC Capitale Investito'] = last_known_invested_capital
        
        actual_cols_to_plot_equity = [c for c in ['PAC Valore Portafoglio', 'PAC Capitale Investito', 'Lump Sum Valore Portafoglio', 'Cash (Valore Fisso 0%)'] if c in combined_equity_plot_df.columns and not combined_equity_plot_df[c].isnull().all()]
        if actual_cols_to_plot_equity: st.line_chart(combined_equity_plot_df[actual_cols_to_plot_equity]); st.write("--- DEBUG: Grafico Equity VISUALIZZATO ---")
        else: st.warning("Dati insuff. per grafico equity.")
    else: st.warning("Indice base per grafici non creato, grafico equity saltato.")

    # --- GRAFICO DRAWDOWN ---
    st.subheader("Andamento del Drawdown nel Tempo")
    if not base_chart_date_index.empty:
        drawdown_data_to_plot = {}
        # ... (come prima)
        if not pac_total_df.empty:
            pac_portfolio_values = pac_total_df.set_index(pd.to_datetime(pac_total_df['Date']))['PortfolioValue']
            pac_drawdown_series = calculate_drawdown_series(pac_portfolio_values)
            if not pac_drawdown_series.empty: drawdown_data_to_plot['PAC Drawdown (%)'] = pac_drawdown_series
        if not lump_sum_df.empty:
            ls_portfolio_values = lump_sum_df.set_index(pd.to_datetime(lump_sum_df['Date']))['PortfolioValue']
            ls_drawdown_series = calculate_drawdown_series(ls_portfolio_values)
            if not ls_drawdown_series.empty: drawdown_data_to_plot['Lump Sum Drawdown (%)'] = ls_drawdown_series
        
        if drawdown_data_to_plot:
            dd_plot_df_temp = pd.DataFrame(drawdown_data_to_plot)
            dd_plot_df = pd.DataFrame(index=base_chart_date_index) # Inizia con l'indice completo
            for col in dd_plot_df_temp.columns: # Unisci e poi ffill
                dd_plot_df = dd_plot_df.join(dd_plot_df_temp[[col]], how='left')
                dd_plot_df[col] = dd_plot_df[col].ffill()

            if not dd_plot_df.empty and not dd_plot_df.isnull().all().all(): st.line_chart(dd_plot_df); st.write("--- DEBUG: Grafico Drawdown VISUALIZZATO ---")
            else: st.warning("DataFrame Drawdown vuoto o solo NaN dopo reindex/ffill.")
    else: st.warning("Indice base per grafici non creato, grafico drawdown saltato.")

# In main.py
# In main.py

        # --- ISTOGRAMMA RENDIMENTI ANNUALI ---
        st.subheader("Istogramma Rendimenti Annuali (%)")
        data_for_annual_hist = {}
        if not pac_total_df.empty and 'PortfolioValue' in pac_total_df.columns and 'Date' in pac_total_df.columns:
            pac_pv_ah = pac_total_df.set_index(pd.to_datetime(pac_total_df['Date']))['PortfolioValue']
            annual_returns_pac = calculate_annual_returns(pac_pv_ah) # Questa funzione ora restituisce una Serie con l'anno come indice
            if not annual_returns_pac.empty:
                # ar_pac.index = annual_returns_pac.index.year # <-- RIMUOVI QUESTA RIGA
                data_for_annual_hist["PAC"] = annual_returns_pac # L'indice è già l'anno

        if not lump_sum_df.empty and 'PortfolioValue' in lump_sum_df.columns and 'Date' in lump_sum_df.columns:
            ls_pv_ah = lump_sum_df.set_index(pd.to_datetime(lump_sum_df['Date']))['PortfolioValue']
            annual_returns_ls = calculate_annual_returns(ls_pv_ah) # Anche qui l'indice è l'anno
            if not annual_returns_ls.empty:
                # annual_returns_ls.index = annual_returns_ls.index.year # <-- RIMUOVI QUESTA RIGA
                data_for_annual_hist["Lump Sum"] = annual_returns_ls
        
        if data_for_annual_hist:
            annual_hist_df = pd.DataFrame(data_for_annual_hist).dropna(how='all')
            if not annual_hist_df.empty:
                st.bar_chart(annual_hist_df)
                st.write("--- DEBUG: Istogramma Rend. Ann. VISUALIZZATO ---")
            else:
                st.warning("Dati per Istogramma Rend. Ann. vuoti dopo dropna.")
        else:
            st.warning("Dati insuff. per Istogramma Rend. Ann.")


    # --- STACKED AREA CHART & TABELLE QUOTE/WAP ---
    st.write("--- DEBUG: Verifica dati per Stacked Area e Tabelle Quote/WAP ---")
    if asset_details_history_df is not None and not asset_details_history_df.empty and not base_chart_date_index.empty:
        st.write(f"--- DEBUG: `asset_details_history_df` per Stacked Area ha {len(asset_details_history_df)} righe.")
        st.subheader("Allocazione Dinamica Portafoglio PAC (Valore per Asset)")
        val_cols_stack = [f'{t}_value' for t in tickers_list if f'{t}_value' in asset_details_history_df.columns]
        if val_cols_stack:
            stack_df_data_temp = asset_details_history_df.set_index(pd.to_datetime(asset_details_history_df['Date']))[val_cols_stack]
            stack_df_data_reindexed = pd.DataFrame(index=base_chart_date_index) # Inizia con l'indice completo
            for col in stack_df_data_temp.columns: # Unisci e poi ffill per ogni colonna asset
                stack_df_data_reindexed = stack_df_data_reindexed.join(stack_df_data_temp[[col]], how='left')
                stack_df_data_reindexed[col] = stack_df_data_reindexed[col].ffill().fillna(0) # ffill e poi fillna(0)
            
            if not stack_df_data_reindexed.empty and not stack_df_data_reindexed.isnull().all().all():
                st.area_chart(stack_df_data_reindexed)
                st.write("--- DEBUG: Stacked Area VISUALIZZATO ---")
            else: st.warning("Dati per Stacked Area vuoti o solo NaN dopo reindex/ffill.")
        else: st.warning("Dati per Stacked Area non sufficienti (no colonne _value).")
        
        st.subheader("Dettagli Finali per Asset nel PAC")
        # ... (logica tabella WAP come prima) ...
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

    else: 
        st.warning("Dati storici dettagliati per asset o indice base non disponibili.")
    
    # CHECKBOX DATI DETTAGLIATI
    # ... (come prima) ...

else: 
    st.info("Inserisci parametri e avvia simulazione.")
    st.write("--- DEBUG: Pagina iniziale ---")
st.sidebar.markdown("---"); st.sidebar.markdown("Kriterion Quant")
