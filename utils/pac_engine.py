# simulatore_pac/main.py
import streamlit as st
import pandas as pd
import numpy as np # Aggiunto per np.isclose
from datetime import datetime, date

from utils.data_loader import load_historical_data_yf
from utils.pac_engine import run_pac_simulation # Nome funzione corretto
from utils.performance import (
    get_total_capital_invested,
    get_final_portfolio_value,
    calculate_total_return_percentage,
    calculate_cagr,
    get_duration_years
)

st.set_page_config(page_title="Simulatore PAC Multi-Asset", layout="wide")
st.title("📘 Simulatore PAC Multi-Asset")
st.caption("Progetto Kriterion Quant")

st.sidebar.header("Parametri della Simulazione")

# Input Multi-Asset
st.sidebar.subheader("Asset e Allocazioni")
tickers_input_str = st.sidebar.text_input("Tickers (separati da virgola)", "AAPL,MSFT,GOOG")
allocations_input_str = st.sidebar.text_input("Allocazioni % (separate da virgola, somma 100)", "60,20,20")

st.sidebar.subheader("Parametri PAC")
monthly_investment_input = st.sidebar.number_input("Importo Versamento Mensile (€/$)", min_value=10.0, value=200.0, step=10.0)
default_start_date_pac = date(2020, 1, 1)
pac_start_date_input = st.sidebar.date_input("Data Inizio PAC", default_start_date_pac)
duration_months_input = st.sidebar.number_input("Durata PAC (in mesi)", min_value=6, value=36, step=1)
reinvest_dividends_input = st.sidebar.checkbox("Reinvesti Dividendi?", value=True)

run_simulation_button = st.sidebar.button("🚀 Avvia Simulazione PAC")

if run_simulation_button:
    tickers_list = [ticker.strip().upper() for ticker in tickers_input_str.split(',') if ticker.strip()]
    
    try:
        allocations_float_list = [float(alloc.strip()) for alloc in allocations_input_str.split(',') if alloc.strip()]
        if not tickers_list:
            st.error("Nessun ticker inserito.")
        elif len(tickers_list) != len(allocations_float_list):
            st.error("Il numero di ticker deve corrispondere al numero di allocazioni.")
        elif not np.isclose(sum(allocations_float_list), 100.0):
            st.error(f"La somma delle allocazioni ({sum(allocations_float_list)}%) deve essere 100%.")
        else:
            allocations_list_norm = [alloc / 100.0 for alloc in allocations_float_list] # Normalizza a somma 1.0
            
            st.header(f"Risultati Simulazione PAC per: {', '.join(tickers_list)}")

            pac_start_date_str = pac_start_date_input.strftime('%Y-%m-%d')
            data_fetch_start_date = (pac_start_date_input - pd.Timedelta(days=180)).strftime('%Y-%m-%d') # Buffer più ampio
            sim_end_date_approx = pd.to_datetime(pac_start_date_input) + pd.DateOffset(months=duration_months_input)
            data_fetch_end_date = (sim_end_date_approx + pd.Timedelta(days=90)).strftime('%Y-%m-%d')

            historical_data_map = {}
            all_data_loaded = True
            min_data_points_needed = duration_months_input + 6 # Stima approssimativa
            
            for ticker_to_load in tickers_list:
                with st.spinner(f"Caricamento dati per {ticker_to_load}..."):
                    data = load_historical_data_yf(ticker_to_load, data_fetch_start_date, data_fetch_end_date)
                    if data.empty or len(data) < min_data_points_needed / (len(tickers_list)*0.5): # Controllo approssimativo
                        st.error(f"Dati insufficienti o mancanti per {ticker_to_load} nel periodo richiesto.")
                        all_data_loaded = False
                        break
                    historical_data_map[ticker_to_load] = data
            
            if not all_data_loaded:
                st.stop()

            st.success("Tutti i dati storici caricati correttamente.")
            
            with st.spinner("Esecuzione simulazione PAC multi-asset..."):
                pac_simulation_df = run_pac_simulation(
                    historical_data_map=historical_data_map,
                    tickers=tickers_list,
                    allocations=allocations_list_norm,
                    monthly_investment=monthly_investment_input,
                    start_date_pac=pac_start_date_str,
                    duration_months=duration_months_input,
                    reinvest_dividends=reinvest_dividends_input
                )

            if pac_simulation_df.empty or 'PortfolioValue' not in pac_simulation_df.columns:
                st.error("La simulazione PAC multi-asset non ha prodotto risultati validi.")
            else:
                st.success("Simulazione PAC multi-asset completata.")

                total_invested = get_total_capital_invested(pac_simulation_df)
                final_value = get_final_portfolio_value(pac_simulation_df)
                total_return_perc = calculate_total_return_percentage(final_value, total_invested)
                duration_yrs = get_duration_years(pac_simulation_df)
                cagr_perc = calculate_cagr(final_value, total_invested, duration_yrs)
                total_dividends_cumulative = pac_simulation_df['DividendsReceivedCumulative'].iloc[-1] if 'DividendsReceivedCumulative' in pac_simulation_df.columns else 0.0

                st.subheader("Metriche di Performance Riepilogative")
                num_cols = 4
                if reinvest_dividends_input and total_dividends_cumulative > 0:
                    num_cols = 5
                
                cols = st.columns(num_cols)
                cols[0].metric("Capitale Totale Investito", f"{total_invested:,.2f}")
                cols[1].metric("Valore Finale Portafoglio", f"{final_value:,.2f}")
                
                col_idx = 2
                if reinvest_dividends_input and total_dividends_cumulative > 0:
                    cols[col_idx].metric("Dividendi Reinvestiti", f"{total_dividends_cumulative:,.2f}")
                    col_idx += 1
                
                cols[col_idx].metric("Rendimento Totale", f"{total_return_perc:.2f}%")
                col_idx += 1
                if pd.notna(cagr_perc):
                    cols[col_idx].metric("CAGR", f"{cagr_perc:.2f}%")
                else:
                    cols[col_idx].metric("CAGR", "N/A")
                
                st.write(f"_Durata approssimativa della simulazione: {duration_yrs:.2f} anni._")
                # ... (resto della UI come prima per grafico e tabella dati)
                st.subheader("Andamento del Portafoglio nel Tempo")
                chart_df = pac_simulation_df[['Date', 'PortfolioValue', 'InvestedCapital']].copy()
                if 'Date' in chart_df.columns:
                     chart_df['Date'] = pd.to_datetime(chart_df['Date'])
                     chart_df = chart_df.set_index('Date')
                if not chart_df.empty:
                    st.line_chart(chart_df)

                if st.checkbox("Mostra dati dettagliati della simulazione PAC"):
                    st.dataframe(pac_simulation_df)


    except ValueError as ve:
        st.error(f"Errore nei valori di input per le allocazioni: {ve}. Assicurati che siano numeri.")
    except Exception as e:
        st.error(f"Si è verificato un errore imprevisto: {e}")
        import traceback
        st.text(traceback.format_exc())

else:
    st.info("Inserisci i parametri nella sidebar e avvia la simulazione.")

st.sidebar.markdown("---")
st.sidebar.markdown("Progetto Kriterion Quant")
