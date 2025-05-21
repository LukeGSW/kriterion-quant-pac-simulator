# simulatore_pac/main.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date

from utils.data_loader import load_historical_data_yf
from utils.pac_engine import run_pac_simulation
# NUOVA IMPORTAZIONE PER BENCHMARK
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
    generate_cash_flows_for_xirr,
    calculate_xirr_metric,
    # calculate_sortino_ratio_empyrical # Ancora commentato o rimosso
)

st.set_page_config(page_title="Simulatore PAC vs Lump Sum", layout="wide")
st.title("📘 Simulatore PAC vs Investimento Unico (Lump Sum)")
st.caption("Progetto Kriterion Quant")

# --- Sidebar (come prima) ---
st.sidebar.header("Parametri della Simulazione")
st.sidebar.subheader("Asset e Allocazioni")
tickers_input_str = st.sidebar.text_input("Tickers (separati da virgola, es. AAPL,MSFT,GOOG)", "AAPL,GOOG,MSFT")
allocations_input_str = st.sidebar.text_input("Allocazioni % (separate da virgola, es. 60,20,20)", "60,20,20")
st.sidebar.subheader("Parametri PAC")
monthly_investment_input = st.sidebar.number_input("Importo Versamento Mensile (€/$)", min_value=10.0, value=200.0, step=10.0)
default_start_date_pac = date(2020, 1, 1)
pac_start_date_input = st.sidebar.date_input("Data Inizio PAC", default_start_date_pac)
duration_months_input = st.sidebar.number_input("Durata PAC (in mesi)", min_value=6, value=36, step=1)
reinvest_dividends_input = st.sidebar.checkbox("Reinvesti Dividendi (per PAC e Lump Sum)?", value=True) # Ora si applica a entrambi
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

if run_simulation_button:
    # --- VALIDAZIONE INPUT (come prima) ---
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

    if not error_in_input:
        st.header(f"Risultati Simulazione per: {', '.join(tickers_list)}")
        # ... (logica caricamento dati `historical_data_map` come prima) ...
        pac_start_date_dt = pd.to_datetime(pac_start_date_input) # Converte in Timestamp per passarlo
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
        
        if all_data_loaded_successfully:
            st.success("Dati storici caricati.")
            
            # --- ESEGUI SIMULAZIONE PAC ---
            with st.spinner("Esecuzione simulazione PAC..."):
                pac_simulation_df = run_pac_simulation(
                    historical_data_map=historical_data_map, tickers=tickers_list, allocations=allocations_list_norm,
                    monthly_investment=monthly_investment_input, start_date_pac=pac_start_date_str,
                    duration_months=duration_months_input, reinvest_dividends=reinvest_dividends_input,
                    rebalance_active=rebalance_active_input, rebalance_frequency=rebalance_frequency_input_str
                )

            if pac_simulation_df.empty or 'PortfolioValue' not in pac_simulation_df.columns or len(pac_simulation_df) < 2:
                st.error("Simulazione PAC non ha prodotto risultati sufficienti.")
            else:
                st.success("Simulazione PAC completata.")
                total_invested_pac = get_total_capital_invested(pac_simulation_df)
                final_value_pac = get_final_portfolio_value(pac_simulation_df)
                
                # --- ESEGUI SIMULAZIONE LUMP SUM ---
                lump_sum_df = pd.DataFrame() # Inizializza
                if total_invested_pac > 0 : # Ha senso fare Lump Sum solo se c'è capitale PAC
                    st.subheader("Simulazione Investimento Unico (Lump Sum) di Confronto")
                    with st.spinner("Esecuzione simulazione Lump Sum..."):
                        # La simulazione Lump Sum deve tracciare l'equity line per lo stesso periodo del PAC
                        # simulation_start_date_for_ls_equity = pd.to_datetime(pac_simulation_df['Date'].iloc[0])
                        # simulation_end_date_for_ls_equity = pd.to_datetime(pac_simulation_df['Date'].iloc[-1])
                        # L'investimento LS avviene alla data di inizio del PAC
                        
                        lump_sum_df = run_lump_sum_simulation(
                            historical_data_map=historical_data_map,
                            tickers=tickers_list,
                            allocations=allocations_list_norm,
                            total_investment_lump_sum=total_invested_pac, # Usa il totale investito dal PAC
                            lump_sum_investment_date=pac_start_date_dt, # Data inizio PAC
                            simulation_start_date=pd.to_datetime(pac_simulation_df['Date'].iloc[0]), # Inizio equity line
                            simulation_end_date=pd.to_datetime(pac_simulation_df['Date'].iloc[-1]),   # Fine equity line
                            reinvest_dividends=reinvest_dividends_input 
                        )
                    if not lump_sum_df.empty:
                        st.success("Simulazione Lump Sum completata.")
                    else:
                        st.warning("Simulazione Lump Sum non ha prodotto risultati.")
                
                # --- CALCOLO E VISUALIZZAZIONE METRICHE (COMPARATIVE) ---
                st.subheader("Metriche di Performance Riepilogative")
                
                # Funzione helper per calcolare metriche per un df di simulazione
                def calculate_metrics_for_strategy(sim_df, total_invested_override=None):
                    metrics = {}
                    if sim_df.empty or 'PortfolioValue' not in sim_df.columns or len(sim_df) < 2:
                        return {k: "N/A" for k in ["Valore Finale", "Rend. Totale", "CAGR", "Volatilità Ann.", "Sharpe", "Max Drawdown", "XIRR"]}

                    metrics["Capitale Investito"] = total_invested_override if total_invested_override is not None else get_total_capital_invested(sim_df)
                    metrics["Valore Finale"] = get_final_portfolio_value(sim_df)
                    metrics["Rend. Totale"] = f"{calculate_total_return_percentage(metrics['Valore Finale'], metrics['Capitale Investito']):.2f}%"
                    
                    duration_yrs_strat = get_duration_years(sim_df)
                    cagr_strat = calculate_cagr(metrics['Valore Finale'], metrics['Capitale Investito'], duration_yrs_strat)
                    metrics["CAGR"] = f"{cagr_strat:.2f}%" if pd.notna(cagr_strat) else "N/A"
                    
                    returns_strat = calculate_portfolio_returns(sim_df)
                    vol_strat = calculate_annualized_volatility(returns_strat)
                    sharpe_strat = calculate_sharpe_ratio(returns_strat, risk_free_rate_annual=(risk_free_rate_input/100.0))
                    mdd_strat = calculate_max_drawdown(sim_df)
                    
                    metrics["Volatilità Ann."] = f"{vol_strat:.2f}%" if pd.notna(vol_strat) else "N/A"
                    metrics["Sharpe"] = f"{sharpe_strat:.2f}" if pd.notna(sharpe_strat) else "N/A"
                    metrics["Max Drawdown"] = f"{mdd_strat:.2f}%" if pd.notna(mdd_strat) else "N/A"

                    # XIRR specifico per PAC (usa monthly_investment e duration_months_input)
                    # Per Lump Sum, XIRR è simile a CAGR se non ci sono altri flussi.
                    if total_invested_override is None: # È PAC
                        xirr_dates, xirr_values = generate_cash_flows_for_xirr(sim_df, pac_start_date_str, duration_months_input, monthly_investment_input, metrics['Valore Finale'])
                        xirr_val = calculate_xirr_metric(xirr_dates, xirr_values)
                        metrics["XIRR"] = f"{xirr_val:.2f}%" if pd.notna(xirr_val) else "N/A"
                    else: # È Lump Sum, XIRR è meno significativo o uguale a CAGR semplice
                        # Per ora, usiamo CAGR anche per LS come XIRR proxy in questa tabella
                        metrics["XIRR"] = metrics["CAGR"]


                    return metrics

                metrics_pac = calculate_metrics_for_strategy(pac_simulation_df)
                
                display_data = {"Metrica": list(metrics_pac.keys()), "PAC": list(metrics_pac.values())}

                if not lump_sum_df.empty:
                    metrics_ls = calculate_metrics_for_strategy(lump_sum_df, total_invested_override=total_invested_pac)
                    display_data["Lump Sum"] = list(metrics_ls.values())
                
                st.table(pd.DataFrame(display_data).set_index("Metrica"))

                # --- GRAFICO COMPARATIVO ---
                st.subheader("Andamento Comparativo del Portafoglio")
                combined_chart_df = pd.DataFrame()
                pac_simulation_df['Date'] = pd.to_datetime(pac_simulation_df['Date'])
                combined_chart_df['Date'] = pac_simulation_df['Date']
                combined_chart_df['PAC Portfolio Value'] = pac_simulation_df['PortfolioValue']
                combined_chart_df['PAC Invested Capital'] = pac_simulation_df['InvestedCapital']

                if not lump_sum_df.empty:
                    lump_sum_df['Date'] = pd.to_datetime(lump_sum_df['Date'])
                    # Unisci basandoti sulle date per allineare correttamente
                    merged_df = pd.merge(combined_chart_df, lump_sum_df[['Date', 'PortfolioValue']], on='Date', how='left')
                    merged_df.rename(columns={'PortfolioValue': 'Lump Sum Portfolio Value'}, inplace=True)
                    # Potrebbe essere necessario un ffill se le date non combaciano perfettamente
                    merged_df['Lump Sum Portfolio Value'] = merged_df['Lump Sum Portfolio Value'].ffill().bfill() 
                    combined_chart_df = merged_df
                    
                chart_to_plot = combined_chart_df.set_index('Date')
                columns_to_plot = ['PAC Portfolio Value', 'PAC Invested Capital']
                if 'Lump Sum Portfolio Value' in chart_to_plot.columns:
                    columns_to_plot.append('Lump Sum Portfolio Value')
                
                st.line_chart(chart_to_plot[columns_to_plot])

                # ... (checkbox per dati dettagliati come prima, magari per entrambi i df)
                if st.checkbox("Mostra dati dettagliati simulazione PAC"):
                    st.dataframe(pac_simulation_df)
                if not lump_sum_df.empty and st.checkbox("Mostra dati dettagliati simulazione Lump Sum"):
                    st.dataframe(lump_sum_df)
        # ... fine blocco if all_data_loaded_successfully
    # ... fine blocco if not error_in_input
else: 
    st.info("Inserisci i parametri nella sidebar e avvia la simulazione.")

st.sidebar.markdown("---")
st.sidebar.markdown("Progetto Kriterion Quant")
