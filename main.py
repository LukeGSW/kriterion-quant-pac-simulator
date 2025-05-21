# simulatore_pac/main.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date

from utils.data_loader import load_historical_data_yf
from utils.pac_engine import run_pac_simulation
from utils.performance import (
    get_total_capital_invested,
    get_final_portfolio_value,
    calculate_total_return_percentage,
    calculate_cagr,
    get_duration_years
)

st.set_page_config(page_title="Simulatore PAC Multi-Asset", layout="wide")
st.title("📘 Simulatore PAC Multi-Asset con Ribilanciamento")
st.caption("Progetto Kriterion Quant - Gestione Multi-Asset e Ribilanciamento")

# --- Sidebar per Input Utente ---
st.sidebar.header("Parametri della Simulazione")

# Input Multi-Asset
st.sidebar.subheader("Asset e Allocazioni")
tickers_input_str = st.sidebar.text_input("Tickers (separati da virgola, es. AAPL,MSFT,GOOG)", "AAPL,GOOG,MSFT")
allocations_input_str = st.sidebar.text_input("Allocazioni % (separate da virgola, es. 60,20,20)", "60,20,20")

st.sidebar.subheader("Parametri PAC")
monthly_investment_input = st.sidebar.number_input("Importo Versamento Mensile (€/$)", min_value=10.0, value=200.0, step=10.0)
default_start_date_pac = date(2020, 1, 1)
pac_start_date_input = st.sidebar.date_input("Data Inizio PAC", default_start_date_pac)
duration_months_input = st.sidebar.number_input("Durata PAC (in mesi)", min_value=6, value=36, step=1)
reinvest_dividends_input = st.sidebar.checkbox("Reinvesti Dividendi?", value=True)

# Nuovi input per il Ribilanciamento
st.sidebar.subheader("Ribilanciamento Periodico")
rebalance_active_input = st.sidebar.checkbox("Attiva Ribilanciamento?", value=False)
rebalance_frequency_input_str = None # Inizializza
if rebalance_active_input:
    rebalance_frequency_input_str = st.sidebar.selectbox(
        "Frequenza Ribilanciamento",
        ["Annuale", "Semestrale", "Trimestrale"],
        index=0 # Default su "Annuale"
    )

run_simulation_button = st.sidebar.button("🚀 Avvia Simulazione PAC")

# --- Area Principale per Output ---
if run_simulation_button:
    # Processa e valida input tickers e allocazioni
    tickers_list = [ticker.strip().upper() for ticker in tickers_input_str.split(',') if ticker.strip()]
    
    error_in_input = False
    allocations_list_norm = [] # Inizializza qui per averla disponibile anche in caso di errore d'input

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

    if not error_in_input:
        st.header(f"Risultati Simulazione PAC per: {', '.join(tickers_list)}")
        alloc_display_list = [f"{tickers_list[i]}: {allocations_float_list_raw[i]}%" for i in range(len(tickers_list))]
        st.write(f"Allocazioni Target: {', '.join(alloc_display_list)}")
        if rebalance_active_input:
            st.write(f"Ribilanciamento Attivo: Sì, Frequenza: {rebalance_frequency_input_str}")
        else:
            st.write("Ribilanciamento Attivo: No")


        pac_start_date_str = pac_start_date_input.strftime('%Y-%m-%d')
        data_fetch_start_date = (pac_start_date_input - pd.Timedelta(days=365*3)).strftime('%Y-%m-%d') # 3 anni prima per storico lungo
        sim_end_date_approx = pd.to_datetime(pac_start_date_input) + pd.DateOffset(months=duration_months_input)
        data_fetch_end_date = (sim_end_date_approx + pd.Timedelta(days=180)).strftime('%Y-%m-%d')

        historical_data_map = {}
        all_data_loaded_successfully = True
        
        for ticker_to_load in tickers_list:
            with st.spinner(f"Caricamento dati per {ticker_to_load}..."):
                data = load_historical_data_yf(ticker_to_load, data_fetch_start_date, data_fetch_end_date)
                # Controllo più robusto sulla disponibilità dei dati per l'intero periodo PAC
                required_sim_start_date = pd.to_datetime(pac_start_date_str)
                required_sim_end_date = sim_end_date_approx 

                if data.empty or data.index.max() < required_sim_end_date or data.index.min() > required_sim_start_date :
                    st.error(f"Dati storici insufficienti per {ticker_to_load} per coprire l'intero periodo di simulazione PAC (da {required_sim_start_date.date()} a {required_sim_end_date.date()}).")
                    all_data_loaded_successfully = False
                    break
                historical_data_map[ticker_to_load] = data
        
        if all_data_loaded_successfully:
            st.success("Tutti i dati storici necessari sono stati caricati correttamente.")
            
            with st.spinner("Esecuzione simulazione PAC..."):
                pac_simulation_df = run_pac_simulation(
                    historical_data_map=historical_data_map,
                    tickers=tickers_list,
                    allocations=allocations_list_norm,
                    monthly_investment=monthly_investment_input,
                    start_date_pac=pac_start_date_str,
                    duration_months=duration_months_input,
                    reinvest_dividends=reinvest_dividends_input,
                    rebalance_active=rebalance_active_input, # NUOVO
                    rebalance_frequency=rebalance_frequency_input_str # NUOVO
                )

            if pac_simulation_df.empty or 'PortfolioValue' not in pac_simulation_df.columns:
                st.error("La simulazione PAC non ha prodotto risultati validi o è vuota.")
            else:
                st.success("Simulazione PAC completata.")

                total_invested = get_total_capital_invested(pac_simulation_df)
                final_value = get_final_portfolio_value(pac_simulation_df)
                total_return_perc = calculate_total_return_percentage(final_value, total_invested)
                duration_yrs = get_duration_years(pac_simulation_df)
                cagr_perc = calculate_cagr(final_value, total_invested, duration_yrs)
                total_dividends_cumulative = pac_simulation_df['DividendsReceivedCumulative'].iloc[-1] if 'DividendsReceivedCumulative' in pac_simulation_df.columns and not pac_simulation_df['DividendsReceivedCumulative'].empty else 0.0

                st.subheader("Metriche di Performance Riepilogative")
                num_metrics_cols = 4
                if reinvest_dividends_input and total_dividends_cumulative > 0:
                    num_metrics_cols = 5
                
                metric_cols = st.columns(num_metrics_cols)
                metric_cols[0].metric("Capitale Totale Investito", f"{total_invested:,.2f}")
                metric_cols[1].metric("Valore Finale Portafoglio", f"{final_value:,.2f}")
                
                current_metric_col_idx = 2
                if reinvest_dividends_input and total_dividends_cumulative > 0:
                    metric_cols[current_metric_col_idx].metric("Dividendi Reinvestiti", f"{total_dividends_cumulative:,.2f}")
                    current_metric_col_idx += 1
                
                metric_cols[current_metric_col_idx].metric("Rendimento Totale", f"{total_return_perc:.2f}%")
                current_metric_col_idx += 1
                if pd.notna(cagr_perc):
                    metric_cols[current_metric_col_idx].metric("CAGR", f"{cagr_perc:.2f}%")
                else:
                    metric_cols[current_metric_col_idx].metric("CAGR", "N/A")
                
                st.write(f"_Durata approssimativa della simulazione: {duration_yrs:.2f} anni._")
                if reinvest_dividends_input:
                    st.write(f"_I dividendi sono stati reinvestiti._")
                else:
                    st.write(f"_I dividendi NON sono stati reinvestiti._")

                st.subheader("Andamento del Portafoglio nel Tempo")
                chart_df = pac_simulation_df[['Date', 'PortfolioValue', 'InvestedCapital']].copy()
                if 'Date' in chart_df.columns:
                     chart_df['Date'] = pd.to_datetime(chart_df['Date'])
                     chart_df = chart_df.set_index('Date')
                
                if not chart_df.empty:
                    st.line_chart(chart_df)
                else:
                    st.warning("Non ci sono dati sufficienti per visualizzare il grafico.")
                
                if st.checkbox("Mostra dati dettagliati della simulazione PAC"):
                    formatters = {
                        "InvestedCapital": "{:,.2f}",
                        "PortfolioValue": "{:,.2f}"
                    }
                    if 'DividendsReceivedCumulative' in pac_simulation_df.columns:
                        formatters['DividendsReceivedCumulative'] = "{:,.2f}"
                    st.dataframe(pac_simulation_df.style.format(formatters))
        # else gestito dal blocco if all_data_loaded_successfully
    # else gestito dal blocco if not error_in_input

else: 
    st.info("Inserisci i parametri nella sidebar a sinistra e avvia la simulazione.")

st.sidebar.markdown("---")
st.sidebar.markdown("Progetto Didattico Kriterion Quant")
