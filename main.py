# simulatore_pac/main.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date

# Importazioni dai tuoi moduli utils
from utils.data_loader import load_historical_data_yf
from utils.pac_engine import run_pac_simulation
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
    # Nuove importazioni per XIRR e Sortino
    generate_cash_flows_for_xirr,
    calculate_xirr_metric,
    #calculate_sortino_ratio_empyrical
)

st.set_page_config(page_title="Simulatore PAC Completo", layout="wide")
st.title("📘 Simulatore PAC Completo con Metriche Avanzate")
st.caption("Progetto Kriterion Quant - Include XIRR e Sortino Ratio")

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
reinvest_dividends_input = st.sidebar.checkbox("Reinvesti Dividendi?", value=True)

st.sidebar.subheader("Ribilanciamento Periodico")
rebalance_active_input = st.sidebar.checkbox("Attiva Ribilanciamento?", value=False)
rebalance_frequency_input_str = None
if rebalance_active_input:
    rebalance_frequency_input_str = st.sidebar.selectbox(
        "Frequenza Ribilanciamento",
        ["Annuale", "Semestrale", "Trimestrale"],
        index=0
    )

st.sidebar.subheader("Parametri Metriche Avanzate")
risk_free_rate_input = st.sidebar.number_input("Tasso Risk-Free Annuale (%) per Sharpe", min_value=0.0, value=1.0, step=0.1, format="%.2f")
#mar_rate_input = st.sidebar.number_input("Tasso Rendimento Minimo Accettabile (MAR) Annuale (%) per Sortino", min_value=0.0, value=0.0, step=0.1, format="%.2f")


run_simulation_button = st.sidebar.button("🚀 Avvia Simulazione PAC")

if run_simulation_button:
    # Processa e valida input tickers e allocazioni
    tickers_list = [ticker.strip().upper() for ticker in tickers_input_str.split(',') if ticker.strip()]
    error_in_input = False
    allocations_list_norm = []
    allocations_float_list_raw = [] # Definita qui per averla nello scope successivo

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
        if allocations_float_list_raw: # Assicurati che sia stata popolata
            alloc_display_list = [f"{tickers_list[i]}: {allocations_float_list_raw[i]}%" for i in range(len(tickers_list))]
            st.write(f"Allocazioni Target: {', '.join(alloc_display_list)}")
        # ... (resto della UI e logica di caricamento dati come prima) ...
        pac_start_date_str = pac_start_date_input.strftime('%Y-%m-%d')
        data_fetch_start_date = (pac_start_date_input - pd.Timedelta(days=365*3)).strftime('%Y-%m-%d')
        sim_end_date_approx = pd.to_datetime(pac_start_date_input) + pd.DateOffset(months=duration_months_input)
        data_fetch_end_date = (sim_end_date_approx + pd.Timedelta(days=180)).strftime('%Y-%m-%d')
        historical_data_map = {}
        all_data_loaded_successfully = True
        for ticker_to_load in tickers_list:
            with st.spinner(f"Caricamento dati per {ticker_to_load}..."):
                data = load_historical_data_yf(ticker_to_load, data_fetch_start_date, data_fetch_end_date)
                required_sim_start_date = pd.to_datetime(pac_start_date_str)
                required_sim_end_date = sim_end_date_approx
                if data.empty or data.index.max() < required_sim_end_date or data.index.min() > required_sim_start_date :
                    st.error(f"Dati storici insufficienti per {ticker_to_load} per coprire l'intero periodo di simulazione PAC.")
                    all_data_loaded_successfully = False
                    break
                historical_data_map[ticker_to_load] = data
        
        if all_data_loaded_successfully:
            st.success("Tutti i dati storici necessari sono stati caricati.")
            with st.spinner("Esecuzione simulazione PAC..."):
                pac_simulation_df = run_pac_simulation(
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

            if pac_simulation_df.empty or 'PortfolioValue' not in pac_simulation_df.columns or len(pac_simulation_df) < 2:
                st.error("La simulazione PAC non ha prodotto risultati sufficienti per calcolare le metriche avanzate.")
            else:
                st.success("Simulazione PAC completata.")

                # Calcolo Metriche di Performance Base
                total_invested = get_total_capital_invested(pac_simulation_df)
                final_value = get_final_portfolio_value(pac_simulation_df)
                total_return_perc = calculate_total_return_percentage(final_value, total_invested)
                duration_yrs = get_duration_years(pac_simulation_df)
                cagr_perc = calculate_cagr(final_value, total_invested, duration_yrs)
                total_dividends_cumulative = pac_simulation_df['DividendsReceivedCumulative'].iloc[-1] if 'DividendsReceivedCumulative' in pac_simulation_df.columns and not pac_simulation_df['DividendsReceivedCumulative'].empty else 0.0

                # Calcolo Metriche Avanzate
                portfolio_daily_returns = calculate_portfolio_returns(pac_simulation_df)
                annual_volatility = calculate_annualized_volatility(portfolio_daily_returns)
                sharpe = calculate_sharpe_ratio(portfolio_daily_returns, risk_free_rate_annual=(risk_free_rate_input / 100.0))
                mdd = calculate_max_drawdown(pac_simulation_df)
                
                # Calcolo XIRR (con la semplificazione attuale)
                xirr_dates, xirr_values = generate_cash_flows_for_xirr(
                    pac_df=pac_simulation_df, 
                    start_date_pac_str=pac_start_date_str, # Data inizio PAC originale
                    duration_months=duration_months_input,
                    monthly_investment=monthly_investment_input,
                    final_portfolio_value=final_value
                )
                xirr_perc = calculate_xirr_metric(xirr_dates, xirr_values)

                # Calcolo Sortino Ratio
                #sortino = calculate_sortino_ratio_empyrical(portfolio_daily_returns, required_return_annual=(mar_rate_input / 100.0))


                st.subheader("Metriche di Performance Riepilogative")
                
                metrics_to_display = {
                    "Capitale Totale Investito": f"{total_invested:,.2f}",
                    "Valore Finale Portafoglio": f"{final_value:,.2f}",
                }
                if reinvest_dividends_input and total_dividends_cumulative > 0:
                    metrics_to_display["Dividendi Reinvestiti"] = f"{total_dividends_cumulative:,.2f}"
                
                metrics_to_display["Rendimento Totale"] = f"{total_return_perc:.2f}%"
                metrics_to_display["CAGR"] = f"{cagr_perc:.2f}%" if pd.notna(cagr_perc) else "N/A"
                #if pd.notna(xirr_perc): # Mostra XIRR se calcolato
                   # metrics_to_display["XIRR Annualizzato"] = f"{xirr_perc:.2f}%"
               # else:
                    #metrics_to_display["XIRR Annualizzato"] = "N/A"

                metrics_to_display["Volatilità Ann."] = f"{annual_volatility:.2f}%" if pd.notna(annual_volatility) else "N/A"
                metrics_to_display["Sharpe Ratio"] = f"{sharpe:.2f}" if pd.notna(sharpe) else "N/A"
                if pd.notna(sortino): # Mostra Sortino se calcolato
                     metrics_to_display["Sortino Ratio"] = f"{sortino:.2f}"
                else:
                    metrics_to_display["Sortino Ratio"] = "N/A"
                metrics_to_display["Max Drawdown"] = f"{mdd:.2f}%" if pd.notna(mdd) else "N/A"
                

                num_metrics_cols_to_show = len(metrics_to_display)
                # Cerca di organizzare le metriche in modo leggibile, ad es. 3 o 4 per riga
                # Se abbiamo molte metriche, potremmo aver bisogno di più righe o di un layout a griglia più flessibile.
                # Per ora, proviamo con un numero fisso di colonne per riga, es. 3 o 4.
                # Questo layout di st.columns potrebbe necessitare di aggiustamenti se le etichette sono lunghe.
                
                # Suddividiamo le metriche in gruppi di 3 per una migliore visualizzazione
                metric_items = list(metrics_to_display.items())
                num_metrics_per_row = 3
                
                for i in range(0, len(metric_items), num_metrics_per_row):
                    cols = st.columns(num_metrics_per_row)
                    for j in range(num_metrics_per_row):
                        if (i + j) < len(metric_items):
                            label, value = metric_items[i+j]
                            cols[j].metric(label, value)
                
                st.write(f"_Durata approssimativa della simulazione: {duration_yrs:.2f} anni._")
                # ... (altre info e grafico come prima) ...
                st.subheader("Andamento del Portafoglio nel Tempo")
                chart_df = pac_simulation_df[['Date', 'PortfolioValue', 'InvestedCapital']].copy()
                if 'Date' in chart_df.columns:
                     chart_df['Date'] = pd.to_datetime(chart_df['Date'])
                     chart_df = chart_df.set_index('Date')
                if not chart_df.empty:
                    st.line_chart(chart_df)

                if st.checkbox("Mostra dati dettagliati della simulazione PAC"):
                    st.dataframe(pac_simulation_df)
        # ... (fine blocco if all_data_loaded_successfully) ...
    # ... (fine blocco if not error_in_input) ...
else: 
    st.info("Inserisci i parametri nella sidebar a sinistra e avvia la simulazione.")

st.sidebar.markdown("---")
st.sidebar.markdown("Progetto Didattico Kriterion Quant")
