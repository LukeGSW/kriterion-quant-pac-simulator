# simulatore_pac/main.py

import streamlit as st
import pandas as pd
from datetime import datetime, date

# Importazioni dai tuoi moduli utils
# Assicurati che utils/pac_engine.py abbia la funzione 'run_pac_simulation'
# e che importi pandas al suo interno.
from utils.data_loader import load_historical_data_yf
from utils.pac_engine import run_pac_simulation # <--- NOME FUNZIONE CORRETTO
from utils.performance import (
    get_total_capital_invested,
    get_final_portfolio_value,
    calculate_total_return_percentage,
    calculate_cagr,
    get_duration_years
)

# Configurazione della pagina Streamlit
st.set_page_config(page_title="Simulatore PAC Avanzato", layout="wide")

st.title("📘 Simulatore di Piano di Accumulo Capitale (PAC)")
st.caption("Versione con Reinvestimento Dividendi - Progetto Kriterion Quant")

# --- Sidebar per Input Utente ---
st.sidebar.header("Parametri della Simulazione")

ticker_symbol = st.sidebar.text_input("Ticker Strumento Finanziario (es. AAPL, GOOGL, CSPX.AS, VWCE.DE)", "AAPL")
monthly_investment_input = st.sidebar.number_input("Importo Versamento Mensile (€/$)", min_value=10.0, value=150.0, step=10.0)

default_start_date_pac = date(2020, 1, 1)
pac_start_date_input = st.sidebar.date_input("Data Inizio PAC", default_start_date_pac)
duration_months_input = st.sidebar.number_input("Durata PAC (in mesi)", min_value=6, value=36, step=1)

# Nuovo input per il reinvestimento dei dividendi
reinvest_dividends_input = st.sidebar.checkbox("Reinvesti Dividendi?", value=True)

run_simulation_button = st.sidebar.button("🚀 Avvia Simulazione PAC")

# --- Area Principale per Output ---
if run_simulation_button:
    st.header(f"Risultati Simulazione PAC per: {ticker_symbol}")

    pac_start_date_str = pac_start_date_input.strftime('%Y-%m-%d')

    data_fetch_start_date = (pac_start_date_input - pd.Timedelta(days=90)).strftime('%Y-%m-%d')
    sim_end_date_approx = pd.to_datetime(pac_start_date_input) + pd.DateOffset(months=duration_months_input)
    data_fetch_end_date = (sim_end_date_approx + pd.Timedelta(days=60)).strftime('%Y-%m-%d') # Aumentato buffer per dati dividendi

    with st.spinner(f"Caricamento dati storici per {ticker_symbol}..."):
        historical_data = load_historical_data_yf(
            ticker=ticker_symbol,
            start_date=data_fetch_start_date,
            end_date=data_fetch_end_date
        )

    if historical_data.empty:
        st.error(f"Impossibile caricare i dati storici per {ticker_symbol}. Controlla il ticker o il periodo.")
    else:
        st.success(f"Dati storici per {ticker_symbol} caricati correttamente.")
        
        with st.spinner("Esecuzione simulazione PAC..."):
            # Chiamata alla funzione aggiornata con il nuovo parametro
            pac_simulation_df = run_pac_simulation( # <--- NOME FUNZIONE CORRETTO
                price_data=historical_data.copy(),
                monthly_investment=monthly_investment_input,
                start_date_pac=pac_start_date_str,
                duration_months=duration_months_input,
                reinvest_dividends=reinvest_dividends_input # <--- NUOVO PARAMETRO PASSATO
            )

        if pac_simulation_df.empty or 'PortfolioValue' not in pac_simulation_df.columns:
            st.error("La simulazione PAC non ha prodotto risultati validi.")
        else:
            st.success("Simulazione PAC completata.")

            total_invested = get_total_capital_invested(pac_simulation_df)
            final_value = get_final_portfolio_value(pac_simulation_df)
            total_return_perc = calculate_total_return_percentage(final_value, total_invested)
            
            duration_yrs = get_duration_years(pac_simulation_df) # Assicurati che pac_simulation_df abbia la colonna 'Date'
            cagr_perc = calculate_cagr(final_value, total_invested, duration_yrs)

            # Estrai i dividendi totali ricevuti/reinvestiti
            total_dividends_cumulative = 0.0
            if 'DividendsReceivedCumulative' in pac_simulation_df.columns:
                total_dividends_cumulative = pac_simulation_df['DividendsReceivedCumulative'].iloc[-1]


            st.subheader("Metriche di Performance Riepilogative")
            # Aggiungiamo una colonna per i dividendi se reinvestiti
            if reinvest_dividends_input and total_dividends_cumulative > 0:
                col1, col2, col3, col4, col5 = st.columns(5)
            else:
                col1, col2, col3, col4 = st.columns(4)

            col1.metric("Capitale Totale Investito", f"{total_invested:,.2f}")
            col2.metric("Valore Finale Portafoglio", f"{final_value:,.2f}")
            
            if reinvest_dividends_input and total_dividends_cumulative > 0:
                col3.metric("Dividendi Reinvestiti", f"{total_dividends_cumulative:,.2f}")
                col4.metric("Rendimento Totale", f"{total_return_perc:.2f}%")
                if pd.notna(cagr_perc):
                    col5.metric("CAGR", f"{cagr_perc:.2f}%")
                else:
                    col5.metric("CAGR", "N/A")
            else:
                col3.metric("Rendimento Totale", f"{total_return_perc:.2f}%")
                if pd.notna(cagr_perc):
                    col4.metric("CAGR", f"{cagr_perc:.2f}%")
                else:
                    col4.metric("CAGR", "N/A")
            
            st.write(f"_Durata approssimativa della simulazione: {duration_yrs:.2f} anni._")
            if reinvest_dividends_input:
                st.write(f"_I dividendi sono stati reinvestiti._")
            else:
                st.write(f"_I dividendi NON sono stati reinvestiti (se pagati, sarebbero stati incassati e non aggiunti al capitale)._")


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
                # Format columns for better readability
                formatters = {
                    "Price": "{:.2f}", "InvestedCapital": "{:,.2f}",
                    "SharesOwned": "{:.4f}", "CashHeld": "{:.2f}",
                    "PortfolioValue": "{:,.2f}"
                }
                if 'DividendsReceivedCumulative' in pac_simulation_df.columns:
                    formatters['DividendsReceivedCumulative'] = "{:,.2f}"
                
                # Rimuovi l'indice numerico se presente prima di passarlo a st.dataframe
                display_df = pac_simulation_df.copy()
                if isinstance(display_df.index, pd.RangeIndex):
                    display_df.set_index('Date', inplace=True) # Se Date è una colonna e vogliamo un DatetimeIndex per la visualizzazione
                
                st.dataframe(display_df.style.format(formatters))
else:
    st.info("Inserisci i parametri nella sidebar a sinistra e avvia la simulazione.")

st.sidebar.markdown("---")
st.sidebar.markdown("Progetto Didattico Kriterion Quant")
