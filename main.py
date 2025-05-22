# simulatore_pac/main.py

import streamlit as st
import pandas as pd
from datetime import datetime, date

# Assumendo che i tuoi moduli siano nella sottocartella 'utils'
# e che Streamlit sia eseguito dalla cartella 'simulatore_pac'
from utils.data_loader import load_historical_data_yf
from utils.pac_engine import run_basic_pac_simulation
from utils.performance import (
    get_total_capital_invested,
    get_final_portfolio_value,
    calculate_total_return_percentage,
    calculate_cagr,
    get_duration_years
)

# Configurazione della pagina Streamlit
st.set_page_config(page_title="Simulatore PAC Base", layout="wide")

st.title("📘 Simulatore di Piano di Accumulo Capitale (PAC)")
st.caption("Versione Base - Progetto Kriterion Quant")

# --- Sidebar per Input Utente ---
st.sidebar.header("Parametri della Simulazione")

ticker_symbol = st.sidebar.text_input("Ticker Strumento Finanziario (es. AAPL, GOOGL, CSPX.AS, VWCE.DE)", "AAPL")
monthly_investment_input = st.sidebar.number_input("Importo Versamento Mensile (€/$)", min_value=10.0, value=150.0, step=10.0)
# Per le date, è bene avere dei default ragionevoli
default_start_date_pac = date(2020, 1, 1)
default_end_date_data_calc = date.today() # Usato per calcolare una durata di default
# Calcola una durata di default, ad es. 3 anni o fino ad oggi se meno di 3 anni
duration_years_default = 3
pac_start_date_input = st.sidebar.date_input("Data Inizio PAC", default_start_date_pac)

# Durata in mesi
# Calcola la data di fine dei dati basata sulla data di inizio + durata
# Per semplicità, usiamo una durata fissa in mesi per ora.
duration_months_input = st.sidebar.number_input("Durata PAC (in mesi)", min_value=6, value=36, step=1)

# Pulsante per avviare la simulazione
run_simulation_button = st.sidebar.button("🚀 Avvia Simulazione PAC")

# --- Area Principale per Output ---
if run_simulation_button:
    st.header(f"Risultati Simulazione PAC per: {ticker_symbol}")

    # Converti le date degli input in stringhe formato 'YYYY-MM-DD' per le funzioni
    pac_start_date_str = pac_start_date_input.strftime('%Y-%m-%d')

    # Determina il periodo per scaricare i dati storici
    # Abbiamo bisogno di dati da un po' prima dell'inizio del PAC fino alla fine della durata del PAC
    data_fetch_start_date = (pac_start_date_input - pd.Timedelta(days=90)).strftime('%Y-%m-%d') # Un buffer prima
    # Calcola la data di fine approssimativa della simulazione per scaricare i dati
    sim_end_date_approx = pd.to_datetime(pac_start_date_input) + pd.DateOffset(months=duration_months_input)
    data_fetch_end_date = (sim_end_date_approx + pd.Timedelta(days=30)).strftime('%Y-%m-%d') # Un buffer dopo

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
            pac_simulation_df = run_basic_pac_simulation(
                price_data=historical_data.copy(), # Passa una copia
                monthly_investment=monthly_investment_input,
                start_date_pac=pac_start_date_str,
                duration_months=duration_months_input
            )

        if pac_simulation_df.empty or 'PortfolioValue' not in pac_simulation_df.columns:
            st.error("La simulazione PAC non ha prodotto risultati validi.")
        else:
            st.success("Simulazione PAC completata.")

            # Calcolo Metriche di Performance
            total_invested = get_total_capital_invested(pac_simulation_df)
            final_value = get_final_portfolio_value(pac_simulation_df)
            total_return_perc = calculate_total_return_percentage(final_value, total_invested)
            
            # La colonna 'Date' in pac_simulation_df è già stata convertita in datetime da pac_engine
            duration_yrs = get_duration_years(pac_simulation_df)
            cagr_perc = calculate_cagr(final_value, total_invested, duration_yrs)

            # Visualizzazione Metriche
            st.subheader("Metriche di Performance Riepilogative")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Capitale Totale Investito", f"{total_invested:,.2f}")
            col2.metric("Valore Finale Portafoglio", f"{final_value:,.2f}")
            col3.metric("Rendimento Totale", f"{total_return_perc:.2f}%")
            if pd.notna(cagr_perc):
                col4.metric("CAGR", f"{cagr_perc:.2f}%")
            else:
                col4.metric("CAGR", "N/A")
            
            st.write(f"_Durata approssimativa della simulazione: {duration_yrs:.2f} anni._")


            # Visualizzazione Grafico Equity Line
            st.subheader("Andamento del Portafoglio nel Tempo")
            
            # Prepara il DataFrame per st.line_chart
            # Deve avere la colonna 'Date' come indice (o essere la prima colonna datetime)
            chart_df = pac_simulation_df[['Date', 'PortfolioValue', 'InvestedCapital']].copy()
            chart_df['Date'] = pd.to_datetime(chart_df['Date'])
            chart_df = chart_df.set_index('Date')
            
            if not chart_df.empty:
                st.line_chart(chart_df)
            else:
                st.warning("Non ci sono dati sufficienti per visualizzare il grafico.")
            
            # Mostra il DataFrame con i risultati (opzionale, per debug o dettaglio)
            if st.checkbox("Mostra dati dettagliati della simulazione PAC"):
                st.dataframe(pac_simulation_df.style.format({
                    "Price": "{:.2f}", "InvestedCapital": "{:,.2f}",
                    "SharesOwned": "{:.4f}", "CashHeld": "{:.2f}",
                    "PortfolioValue": "{:,.2f}"
                }))

else:
    st.info("Inserisci i parametri nella sidebar a sinistra e avvia la simulazione.")

st.sidebar.markdown("---")
st.sidebar.markdown("Progetto Didattico Kriterion Quant")
