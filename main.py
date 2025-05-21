# simulatore_pac/main.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date

# Importazioni dai tuoi moduli utils
# Assicurati che questi file siano aggiornati e corretti su GitHub
try:
    from utils.data_loader import load_historical_data_yf
    from utils.pac_engine import run_pac_simulation # Deve restituire (pac_total_df, asset_details_history_df)
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
        # calculate_sortino_ratio_empyrical # Mantenuto commentato
    )
    IMPORT_SUCCESS = True
except ImportError as import_err:
    IMPORT_SUCCESS = False
    IMPORT_ERROR_MESSAGE = str(import_err)


st.set_page_config(page_title="Simulatore PAC Dettagliato", layout="wide")
st.title("📘 Simulatore PAC Dettagliato")
st.caption("Progetto Kriterion Quant - Analisi Multi-Asset, Ribilanciamento e Dettagli per Asset")

if not IMPORT_SUCCESS:
    st.error(f"Errore critico durante l'importazione dei moduli: {IMPORT_ERROR_MESSAGE}")
    st.error("Assicurati che tutti i file .py dei moduli utils siano presenti e corretti nel repository GitHub.")
    st.stop()

# --- Sidebar per Input Utente ---
st.sidebar.header("Parametri della Simulazione")

st.sidebar.subheader("Asset e Allocazioni")
tickers_input_str = st.sidebar.text_input("Tickers (separati da virgola, es. AAPL,MSFT,GOOG)", "AAPL,GOOG,MSFT")
allocations_input_str = st.sidebar.text_input("Allocazioni % (separate da virgola, es. 60,20,20)", "60,20,20")

st.sidebar.subheader("Parametri PAC")
monthly_investment_input = st.sidebar.number_input("Importo Versamento Mensile (€/$)", min_value=10.0, value=200.0, step=10.0)
default_start_date_pac = date(2020, 1, 1)
pac_start_date_input = st.sidebar.date_input("Data Inizio PAC", default_start_date_pac)
duration_months_input = st.sidebar.number_input("Durata PAC (in mesi)", min_value=6, value=36, step=1)
reinvest_dividends_input = st.sidebar.checkbox("Reinvesti Dividendi (per PAC e Lump Sum)?", value=True)

st.sidebar.subheader("Ribilanciamento Periodico (Solo per PAC)")
rebalance_active_input = st.sidebar.checkbox("Attiva Ribilanciamento (per PAC)?", value=False)
rebalance_frequency_input_str = None
if rebalance_active_input:
    rebalance_frequency_input_str = st.sidebar.selectbox(
        "Frequenza Ribilanciamento",
        ["Annuale", "Semestrale", "Trimestrale"], index=0
    )

st.sidebar.subheader("Parametri Metriche Avanzate")
risk_free_rate_input = st.sidebar.number_input("Tasso Risk-Free Annuale (%) per Sharpe", min_value=0.0, value=1.0, step=0.1, format="%.2f")
# mar_rate_input = st.sidebar.number_input(...) # Sortino ancora sospeso

run_simulation_button = st.sidebar.button("🚀 Avvia Simulazioni")

# --- Area Principale per Output ---
if run_simulation_button:
    st.write("--- DEBUG: Pulsante 'Avvia Simulazioni' PREMUTO ---")

    # --- VALIDAZIONE INPUT ---
    tickers_list = [ticker.strip().upper() for ticker in tickers_input_str.split(',') if ticker.strip()]
    error_in_input = False
    allocations_list_norm = []
    allocations_float_list_raw = [] 

    if not tickers_list:
        st.error("Errore: Devi inserire almeno un ticker.")
        error_in_input = True
    
    if not error_in_input:
        try:
            allocations_float_list_raw = [float(alloc.strip()) for alloc in allocations_input_str.split(',') if alloc.strip()]
            if len(tickers_list) != len(allocations_float_list_raw):
                st.error("Errore: Il numero di ticker deve corrispondere al numero di allocazioni.")
                error_in_input = True
            elif not np.isclose(sum(allocations_float_list_raw), 100.0):
                st.error(f"Errore: La somma delle allocazioni ({sum(allocations_float_list_raw)}%) deve essere 100%.")
                error_in_input = True
            else:
                allocations_list_norm = [alloc / 100.0 for alloc in allocations_float_list_raw]
        except ValueError:
            st.error("Errore: Le allocazioni devono essere numeri validi (es. 50, 30.5, 20).")
            error_in_input = True
    
    if error_in_input:
        st.write("--- DEBUG: Errore negli input, simulazione interrotta prima del caricamento dati. ---")
        st.stop() # Interrompe l'esecuzione se ci sono errori di input

    # Se siamo qui, gli input di base sono validi
    st.write(f"--- DEBUG: Input validati. Tickers: {tickers_list}, Allocazioni normalizzate: {allocations_list_norm} ---")
    st.header(f"Risultati Simulazione per: {', '.join(tickers_list)}")
    if allocations_float_list_raw:
        alloc_display_list = [f"{tickers_list[i]}: {allocations_float_list_raw[i]}%" for i in range(len(tickers_list))]
        st.write(f"Allocazioni Target: {', '.join(alloc_display_list)}")
    if rebalance_active_input:
        st.write(f"Ribilanciamento Attivo (PAC): Sì, Frequenza: {rebalance_frequency_input_str}")
    else:
        st.write("Ribilanciamento Attivo (PAC): No")

    # --- CARICAMENTO DATI ---
    pac_start_date_dt = pd.to_datetime(pac_start_date_input)
    pac_start_date_str = pac_start_date_dt.strftime('%Y-%m-%d')
    data_fetch_start_date = (pac_start_date_dt - pd.Timedelta(days=365*3)).strftime('%Y-%m-%d')
    sim_end_date_approx_dt = pac_start_date_dt + pd.DateOffset(months=duration_months_input)
    data_fetch_end_date = (sim_end_date_approx_dt + pd.Timedelta(days=180)).strftime('%Y-%m-%d')
    
    historical_data_map = {}
    all_data_loaded_successfully = True
    for ticker_to_load in tickers_list:
        with st.spinner(f"Caricamento dati per {ticker_to_load}..."):
            data = load_historical_data_yf(ticker_to_load, data_fetch_start_date, data_fetch_end_date)
            if data.empty or data.index.max() < sim_end_date_approx_dt or data.index.min() > pac_start_date_dt :
                st.error(f"Dati storici insufficienti per {ticker_to_load} per coprire l'intero periodo PAC.")
                all_data_loaded_successfully = False; break
            historical_data_map[ticker_to_load] = data
    
    if not all_data_loaded_successfully:
        st.write("--- DEBUG: Caricamento dati fallito per uno o più ticker. ---")
        st.stop()

    st.success("Dati storici caricati.")
    st.write("--- DEBUG: Dati storici caricati, prima della chiamata a run_pac_simulation ---")
    
    # --- ESECUZIONE SIMULAZIONI ---
    pac_total_df, asset_details_history_df = pd.DataFrame(), pd.DataFrame() # Inizializza
    
    try:
        with st.spinner("Esecuzione simulazione PAC..."):
            pac_total_df, asset_details_history_df = run_pac_simulation(
                historical_data_map=historical_data_map, 
                tickers=tickers_list, 
                allocations=allocations_list_norm,
                monthly_investment=monthly_investment_input, 
                start_date_pac=pac_start_date_str,
                duration_months=duration_months_input, 
                reinvest_dividends=reinvest_dividends_input,
                rebalance_active=rebalance_active_input, 
                rebalance_frequency=rebalance_frequency_input_str
            )
        st.write("--- DEBUG: Chiamata a run_pac_simulation COMPLETATA ---")
    except Exception as e_pac:
        st.error(f"Errore CRITICO durante run_pac_simulation: {e_pac}")
        st.write("--- DEBUG: ERRORE durante run_pac_simulation ---")
        import traceback
        st.text(traceback.format_exc())
        st.stop() # Interrompe se la simulazione PAC fallisce criticamente

    # Verifica asset_details_history_df
    st.subheader("--- DEBUG: Verifica Output da pac_engine ---")
    if pac_total_df is not None and not pac_total_df.empty:
        st.write("`pac_total_df` ricevuto e NON è vuoto.")
    else:
        st.write("`pac_total_df` ricevuto VUOTO o None.")

    if asset_details_history_df is not None and not asset_details_history_df.empty:
        st.write("`asset_details_history_df` ricevuto e NON è vuoto. Prime righe:")
        st.dataframe(asset_details_history_df.head())
        st.write(f"Colonne in `asset_details_history_df`: {asset_details_history_df.columns.tolist()}")
    elif asset_details_history_df is None:
        st.write("`asset_details_history_df` ricevuto come None.")
    else: 
        st.write("`asset_details_history_df` ricevuto VUOTO.")
    st.write("--- FINE DEBUG Output da pac_engine ---")


    # ESEGUI SIMULAZIONE LUMP SUM (solo se PAC ha prodotto risultati)
    lump_sum_df = pd.DataFrame()
    if not pac_total_df.empty and 'PortfolioValue' in pac_total_df.columns and len(pac_total_df) >= 2:
        total_invested_pac_for_ls = get_total_capital_invested(pac_total_df)
        if total_invested_pac_for_ls > 0:
             with st.spinner("Esecuzione simulazione Lump Sum..."):
                lump_sum_df = run_lump_sum_simulation(
                    historical_data_map=historical_data_map, tickers=tickers_list, allocations=allocations_list_norm,
                    total_investment_lump_sum=total_invested_pac_for_ls, lump_sum_investment_date=pac_start_date_dt,
                    simulation_start_date=pd.to_datetime(pac_total_df['Date'].iloc[0]),
                    simulation_end_date=pd.to_datetime(pac_total_df['Date'].iloc[-1]),
                    reinvest_dividends=reinvest_dividends_input)
                if not lump_sum_df.empty: st.success("Simulazione Lump Sum completata.")
                else: st.warning("Simulazione Lump Sum non ha prodotto risultati o è vuota.")

    # --- CONTROLLO RISULTATI E VISUALIZZAZIONE ---
    if pac_total_df.empty or 'PortfolioValue' not in pac_total_df.columns or len(pac_total_df) < 2:
        st.error("Simulazione PAC non ha prodotto risultati sufficienti per l'analisi dettagliata.")
    else:
        st.success("Inizio elaborazione metriche e grafici...")
        
        # CALCOLO METRICHE (funzione helper e tabella come prima)
        # ... (la tua funzione helper calculate_metrics_for_strategy e la visualizzazione della tabella st.table) ...
        def calculate_metrics_for_strategy(sim_df, strategy_name, total_invested_override=None, is_pac=False):
            metrics = {}
            if sim_df.empty or 'PortfolioValue' not in sim_df.columns or len(sim_df) < 2:
                keys = ["Capitale Investito", "Valore Finale", "Rend. Totale", "CAGR", "XIRR", "Volatilità Ann.", "Sharpe", "Max Drawdown"]
                return {k: "N/A" for k in keys}
            actual_total_invested = total_invested_override if total_invested_override is not None else get_total_capital_invested(sim_df)
            metrics["Capitale Investito"] = f"{actual_total_invested:,.2f}"
            final_portfolio_val = get_final_portfolio_value(sim_df)
            metrics["Valore Finale"] = f"{final_portfolio_val:,.2f}"
            metrics["Rend. Totale"] = f"{calculate_total_return_percentage(final_portfolio_val, actual_total_invested):.2f}%"
            duration_yrs_strat = get_duration_years(sim_df.copy()) # Passa copia
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
        display_data_metrics = {"Metrica": list(metrics_pac.keys()), "PAC": list(metrics_pac.values())}
        if not lump_sum_df.empty:
            total_invested_val_pac = get_total_capital_invested(pac_total_df)
            metrics_ls = calculate_metrics_for_strategy(lump_sum_df, "Lump Sum", total_invested_override=total_invested_val_pac, is_pac=False)
            ls_values_aligned = [metrics_ls.get(key, "N/A") for key in metrics_pac.keys()]
            display_data_metrics["Lump Sum"] = ls_values_aligned
        st.subheader("Metriche di Performance Riepilogative")
        st.table(pd.DataFrame(display_data_metrics).set_index("Metrica"))


        # GRAFICO COMPARATIVO EQUITY LINE (come prima)
        # ... (codice come prima) ...
        st.subheader("Andamento Comparativo del Portafoglio")
        combined_chart_df = pd.DataFrame()
        pac_total_df_copy = pac_total_df.copy() 
        pac_total_df_copy['Date'] = pd.to_datetime(pac_total_df_copy['Date'])
        combined_chart_df['Date'] = pac_total_df_copy['Date']
        combined_chart_df['PAC Valore Portafoglio'] = pac_total_df_copy['PortfolioValue']
        combined_chart_df['PAC Capitale Investito'] = pac_total_df_copy['InvestedCapital']
        if not lump_sum_df.empty:
            lump_sum_df_copy = lump_sum_df.copy()
            lump_sum_df_copy['Date'] = pd.to_datetime(lump_sum_df_copy['Date'])
            merged_df = pd.merge(combined_chart_df, lump_sum_df_copy[['Date', 'PortfolioValue']], on='Date', how='left', suffixes=('_pac', '_ls'))
            merged_df.rename(columns={'PortfolioValue': 'Lump Sum Valore Portafoglio'}, inplace=True)
            merged_df['Lump Sum Valore Portafoglio'] = merged_df['Lump Sum Valore Portafoglio'].ffill().bfill() 
            combined_chart_df = merged_df
        chart_to_plot_equity = combined_chart_df.set_index('Date')
        columns_to_plot_equity = ['PAC Valore Portafoglio', 'PAC Capitale Investito']
        if 'Lump Sum Valore Portafoglio' in chart_to_plot_equity.columns:
            columns_to_plot_equity.append('Lump Sum Valore Portafoglio')
        st.line_chart(chart_to_plot_equity[columns_to_plot_equity])

        # GRAFICO DRAWDOWN (come prima)
        # ... (codice come prima) ...
        st.subheader("Andamento del Drawdown nel Tempo")
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
            if not combined_drawdown_df_plot.empty: st.line_chart(combined_drawdown_df_plot)
            else: st.warning("Non è stato possibile generare i dati per il grafico drawdown.")
        else: st.info("Dati non sufficienti per calcolare la serie di drawdown.")


        # --- NUOVO: STACKED AREA CHART PER ALLOCAZIONE ASSET ---
        st.write("--- DEBUG: Tentativo di generare Stacked Area Chart ---") # DEBUG
        if asset_details_history_df is not None and not asset_details_history_df.empty:
            st.subheader("Allocazione Dinamica del Portafoglio PAC nel Tempo (Valore per Asset)")
            
            value_cols_to_plot_stacked = [f'{ticker}_value' for ticker in tickers_list if f'{ticker}_value' in asset_details_history_df.columns]
            st.write(f"--- DEBUG: Colonne per Stacked Area Chart: {value_cols_to_plot_stacked} ---") # DEBUG
            
            if value_cols_to_plot_stacked:
                stacked_area_df_data = asset_details_history_df.copy()
                if 'Date' in stacked_area_df_data.columns:
                    stacked_area_df_data['Date'] = pd.to_datetime(stacked_area_df_data['Date'])
                    stacked_area_df_data = stacked_area_df_data.set_index('Date')
                
                # Assicurati che le colonne selezionate esistano ancora dopo set_index
                actual_cols_in_stacked_df = [col for col in value_cols_to_plot_stacked if col in stacked_area_df_data.columns]
                if actual_cols_in_stacked_df:
                    st.area_chart(stacked_area_df_data[actual_cols_in_stacked_df])
                    st.write("--- DEBUG: Stacked Area Chart DOVREBBE ESSERE VISUALIZZATO ---") # DEBUG
                else:
                    st.warning("Nessuna colonna di valore per asset trovata per lo stacked area chart dopo aver impostato l'indice.")
            else:
                st.warning("Dati sui valori per asset non sufficienti per il grafico dell'allocazione dinamica (nessuna colonna '_value' trovata).")
        else:
            st.warning("Dati storici dettagliati per asset non disponibili (asset_details_history_df è vuoto o None) per il grafico dell'allocazione dinamica.")

        # --- NUOVO: TABELLE AGGIUNTIVE (QUOTE e WAP) ---
        st.write("--- DEBUG: Tentativo di generare Tabella Quote/WAP ---") # DEBUG
        if asset_details_history_df is not None and not asset_details_history_df.empty:
            st.subheader("Dettagli Finali per Asset nel PAC")
            final_asset_details_list_for_table = []
            last_day_asset_details = asset_details_history_df.iloc[-1] # Prendi l'ultima riga disponibile

            for ticker_idx, ticker_name in enumerate(tickers_list):
                # Nomi delle colonne basati su come li creiamo in pac_engine
                shares_col_name = f'{ticker_name}_shares'
                capital_col_name = f'{ticker_name}_capital_invested'
                
                final_shares = last_day_asset_details.get(shares_col_name, 0.0)
                total_capital_for_asset = last_day_asset_details.get(capital_col_name, 0.0)
                
                wap = np.nan
                if final_shares > 1e-6 and total_capital_for_asset > 1e-6:
                    wap = total_capital_for_asset / final_shares
                elif final_shares > 1e-6 and total_capital_for_asset <= 1e-6:
                    wap = 0.0
                
                final_asset_details_list_for_table.append({
                    "Ticker": ticker_name,
                    "Quote Finali": f"{final_shares:.4f}",
                    "Capitale Investito (Asset)": f"{total_capital_for_asset:,.2f}",
                    "Prezzo Medio Carico (WAP)": f"{wap:,.2f}" if pd.notna(wap) else "N/A"
                })
            
            if final_asset_details_list_for_table:
                details_summary_df = pd.DataFrame(final_asset_details_list_for_table)
                st.table(details_summary_df.set_index("Ticker"))
                st.write("--- DEBUG: Tabella Quote/WAP DOVREBBE ESSERE VISUALIZZATA ---") # DEBUG
            else:
                st.warning("Impossibile costruire la tabella dei dettagli finali per asset.")
        else:
            st.warning("Dati storici dettagliati per asset non disponibili (asset_details_history_df è vuoto o None) per la tabella Quote/WAP.")

        # Checkbox per dati dettagliati
        if st.checkbox("Mostra dati aggregati dettagliati del PAC", key="pac_total_data_detail_main"):
            st.dataframe(pac_total_df)
        if asset_details_history_df is not None and not asset_details_history_df.empty and \
           st.checkbox("Mostra dati storici dettagliati per asset del PAC", key="pac_asset_data_detail_main"):
            st.dataframe(asset_details_history_df)
        if not lump_sum_df.empty and \
           st.checkbox("Mostra dati dettagliati della simulazione Lump Sum", key="ls_data_detail_main_key"): # Chiave modificata
            st.dataframe(lump_sum_df)

else: # Se il pulsante non è stato premuto
    st.info("Inserisci i parametri nella sidebar a sinistra e avvia la simulazione.")
    st.write("--- DEBUG: Pagina iniziale, pulsante non premuto ---")

st.sidebar.markdown("---")
st.sidebar.markdown("Progetto Didattico Kriterion Quant")
