# simulatore_pac/main.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date

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
        calculate_annual_returns # NUOVA IMPORTAZIONE
        # calculate_sortino_ratio_empyrical # Mantenuto commentato
    )
    IMPORT_SUCCESS = True
except ImportError as import_err:
    IMPORT_SUCCESS = False
    IMPORT_ERROR_MESSAGE = str(import_err)


st.set_page_config(page_title="Simulatore PAC Completo", layout="wide")
st.title("📘 Simulatore PAC Completo con Benchmark e Analisi Avanzate")
st.caption("Progetto Kriterion Quant")

if not IMPORT_SUCCESS:
    st.error(f"Errore critico durante l'importazione dei moduli: {IMPORT_ERROR_MESSAGE}")
    st.error("Assicurati che tutti i file .py dei moduli utils siano presenti, corretti e che tutte le dipendenze in requirements.txt siano installate correttamente su Streamlit Cloud.")
    st.stop()

# --- Sidebar per Input Utente ---
st.sidebar.header("Parametri della Simulazione")

st.sidebar.subheader("Asset e Allocazioni")
tickers_input_str = st.sidebar.text_input("Tickers (separati da virgola, es. AAPL,MSFT,GOOG)", "AAPL,GOOG,MSFT")
allocations_input_str = st.sidebar.text_input("Allocazioni % (separate da virgola, es. 60,20,20)", "60,20,20")

st.sidebar.subheader("Parametri PAC")
monthly_investment_input = st.sidebar.number_input("Importo Versamento Mensile (€/$)", min_value=10.0, value=200.0, step=10.0)
# Data di default per l'inizio del PAC
default_start_date_pac_sidebar = date(2020, 1, 1)
# Data di default per la fine del PAC (per calcolare una durata di default)
# Ad esempio, 3 anni dal default_start_date_pac o fino ad oggi meno un po' se troppo breve
end_date_for_default_duration = default_start_date_pac_sidebar.replace(year=default_start_date_pac_sidebar.year + 3)
if end_date_for_default_duration > date.today():
    end_date_for_default_duration = date.today()

pac_start_date_input = st.sidebar.date_input("Data Inizio PAC", default_start_date_pac_sidebar)

# Calcolo durata mesi di default
default_duration_months = (end_date_for_default_duration.year - pac_start_date_input.year) * 12 + \
                          (end_date_for_default_duration.month - pac_start_date_input.month)
if default_duration_months < 6 : default_duration_months = 36 # Minimo 6 mesi, default 36 se troppo corto

duration_months_input = st.sidebar.number_input("Durata PAC (in mesi)", min_value=6, value=default_duration_months, step=1)

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

run_simulation_button = st.sidebar.button("🚀 Avvia Simulazioni")

# --- Area Principale per Output ---
if run_simulation_button:
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
        st.stop() 

    st.header(f"Risultati Simulazione per: {', '.join(tickers_list)}")
    if allocations_float_list_raw:
        alloc_display_list = [f"{tickers_list[i]}: {allocations_float_list_raw[i]}%" for i in range(len(tickers_list))]
        st.write(f"Allocazioni Target: {', '.join(alloc_display_list)}")
    # ... (Info su ribilanciamento)

    # --- CARICAMENTO DATI ---
    pac_start_date_dt = pd.to_datetime(pac_start_date_input)
    pac_start_date_str = pac_start_date_dt.strftime('%Y-%m-%d')
    
    # Calcola la data di fine effettiva del PAC
    actual_pac_end_date_dt = pac_start_date_dt + pd.DateOffset(months=duration_months_input)
    
    # Date per il fetch dei dati: un buffer attorno al periodo PAC effettivo
    data_fetch_start_date = (pac_start_date_dt - pd.Timedelta(days=365*1)).strftime('%Y-%m-%d') # 1 anno prima
    data_fetch_end_date = (actual_pac_end_date_dt + pd.Timedelta(days=31)).strftime('%Y-%m-%d') # 1 mese dopo la fine del PAC

    historical_data_map = {}
    all_data_loaded_successfully = True
    for ticker_to_load in tickers_list:
        with st.spinner(f"Caricamento dati per {ticker_to_load}..."):
            data = load_historical_data_yf(ticker_to_load, data_fetch_start_date, data_fetch_end_date)
            if data.empty or data.index.max() < (actual_pac_end_date_dt - pd.Timedelta(days=1)) or data.index.min() > pac_start_date_dt :
                st.error(f"Dati storici insufficienti per {ticker_to_load} per coprire l'intero periodo PAC (da {pac_start_date_dt.date()} a {actual_pac_end_date_dt.date()}).")
                all_data_loaded_successfully = False; break
            historical_data_map[ticker_to_load] = data
    
    if not all_data_loaded_successfully:
        st.stop()

    st.success("Dati storici caricati.")
    
    # --- ESECUZIONE SIMULAZIONI ---
    pac_total_df, asset_details_history_df = pd.DataFrame(), pd.DataFrame()
    
    try:
        with st.spinner("Esecuzione simulazione PAC..."):
            pac_total_df, asset_details_history_df = run_pac_simulation(
                historical_data_map=historical_data_map, tickers=tickers_list, allocations=allocations_list_norm,
                monthly_investment=monthly_investment_input, start_date_pac=pac_start_date_str,
                duration_months=duration_months_input, reinvest_dividends=reinvest_dividends_input,
                rebalance_active=rebalance_active_input, rebalance_frequency=rebalance_frequency_input_str )
    except Exception as e_pac:
        st.error(f"Errore CRITICO durante run_pac_simulation: {e_pac}")
        import traceback
        st.text(traceback.format_exc())
        st.stop()

    lump_sum_df = pd.DataFrame()
    if not pac_total_df.empty and 'PortfolioValue' in pac_total_df.columns and len(pac_total_df) >= 2:
        total_invested_pac_for_ls = get_total_capital_invested(pac_total_df)
        if total_invested_pac_for_ls > 0:
             with st.spinner("Esecuzione simulazione Lump Sum..."):
                lump_sum_df = run_lump_sum_simulation(
                    historical_data_map=historical_data_map, tickers=tickers_list, allocations=allocations_list_norm,
                    total_investment_lump_sum=total_invested_pac_for_ls, lump_sum_investment_date=pac_start_date_dt,
                    simulation_start_date=pd.to_datetime(pac_total_df['Date'].iloc[0]), # Usa date effettive dal PAC df
                    simulation_end_date=pd.to_datetime(pac_total_df['Date'].iloc[-1]),
                    reinvest_dividends=reinvest_dividends_input)
                if not lump_sum_df.empty: st.success("Simulazione Lump Sum completata.")
                else: st.warning("Simulazione Lump Sum non ha prodotto risultati o è vuota.")
    
    # --- CONTROLLO RISULTATI E VISUALIZZAZIONE ---
    if pac_total_df.empty or 'PortfolioValue' not in pac_total_df.columns or len(pac_total_df) < 2:
        st.error("Simulazione PAC non ha prodotto risultati sufficienti per l'analisi dettagliata.")
    else:
        st.success("Simulazioni completate. Inizio elaborazione output.")
        
        # FUNZIONE HELPER PER METRICHE (invariata)
        def calculate_metrics_for_strategy(sim_df, strategy_name, total_invested_override=None, is_pac=False):
            # ... (codice della funzione helper come prima)
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

        # TABELLA METRICHE (invariata)
        metrics_pac = calculate_metrics_for_strategy(pac_total_df, "PAC", is_pac=True)
        display_data_metrics = {"Metrica": list(metrics_pac.keys()), "PAC": list(metrics_pac.values())}
        if not lump_sum_df.empty:
            total_invested_val_pac = get_total_capital_invested(pac_total_df) # Questo è il capitale per il LS
            metrics_ls = calculate_metrics_for_strategy(lump_sum_df, "Lump Sum", total_invested_override=total_invested_val_pac, is_pac=False)
            ls_values_aligned = [metrics_ls.get(key, "N/A") for key in metrics_pac.keys()]
            display_data_metrics["Lump Sum"] = ls_values_aligned
        st.subheader("Metriche di Performance Riepilogative")
        st.table(pd.DataFrame(display_data_metrics).set_index("Metrica"))

        # GRAFICO EQUITY LINE (CON CASH BENCHMARK)
        st.subheader("Andamento Comparativo del Portafoglio")
        combined_chart_df = pd.DataFrame()
        if not pac_total_df.empty and 'Date' in pac_total_df.columns:
            pac_total_df_copy = pac_total_df.copy()
            pac_total_df_copy['Date'] = pd.to_datetime(pac_total_df_copy['Date'])
            combined_chart_df['Date'] = pac_total_df_copy['Date']
            combined_chart_df['PAC Valore Portafoglio'] = pac_total_df_copy['PortfolioValue']
            combined_chart_df['PAC Capitale Investito'] = pac_total_df_copy['InvestedCapital']

            # Aggiungi CASH BENCHMARK
            # Il valore del cash benchmark è il capitale totale del PAC, ma costante.
            # Prendiamo il capitale totale finale del PAC come riferimento per il cash benchmark
            total_capital_for_cash_benchmark = get_total_capital_invested(pac_total_df) # Totale investito dal PAC
            
            # Per il grafico, la linea "Cash" inizia con il primo versamento e cresce linearmente
            # con i versamenti PAC (se "Cash" = alternativa di accumulare i versamenti PAC senza investirli).
            # Oppure, se "Cash" = tenere l'importo totale del LS in contanti dall'inizio:
            # combined_chart_df['Cash (Benchmark 0%)'] = total_capital_for_cash_benchmark 
            # Questa seconda interpretazione crea una linea orizzontale.
            # Per il documento "Cash (benchmark a 0%)", una linea orizzontale ha senso se confrontato con LS.
            # Per il PAC, la linea "PAC Capitale Investito" già funge da costo base / cash accumulato.
            # Decidiamo per la linea orizzontale per "Cash" (alternativa a LS):
            if not combined_chart_df.empty:
                 combined_chart_df['Cash (Valore Fisso 0%)'] = total_capital_for_cash_benchmark


            if not lump_sum_df.empty:
                lump_sum_df_copy = lump_sum_df.copy()
                lump_sum_df_copy['Date'] = pd.to_datetime(lump_sum_df_copy['Date'])
                merged_df = pd.merge(combined_chart_df, lump_sum_df_copy[['Date', 'PortfolioValue']], on='Date', how='left', suffixes=('_pac', '_ls_val'))
                merged_df.rename(columns={'PortfolioValue': 'Lump Sum Valore Portafoglio'}, inplace=True)
                merged_df['Lump Sum Valore Portafoglio'] = merged_df['Lump Sum Valore Portafoglio'].ffill().bfill() 
                combined_chart_df = merged_df
            
            chart_to_plot_equity = combined_chart_df.set_index('Date')
            columns_to_plot_equity = ['PAC Valore Portafoglio', 'PAC Capitale Investito']
            if 'Lump Sum Valore Portafoglio' in chart_to_plot_equity.columns:
                columns_to_plot_equity.append('Lump Sum Valore Portafoglio')
            if 'Cash (Valore Fisso 0%)' in chart_to_plot_equity.columns:
                columns_to_plot_equity.append('Cash (Valore Fisso 0%)')

            if not chart_to_plot_equity.empty:
                st.line_chart(chart_to_plot_equity[columns_to_plot_equity])
            else:
                st.warning("Dati insufficienti per il grafico equity.")

        # GRAFICO DRAWDOWN (come prima)
        # ... (codice grafico drawdown come prima) ...
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


        # ISTOGRAMMA RENDIMENTI ANNUALI
        st.subheader("Istogramma Rendimenti Annuali (%)")
        data_for_annual_hist = {}
        if not pac_total_df.empty and 'PortfolioValue' in pac_total_df and 'Date' in pac_total_df:
            pac_pv_series_for_annual = pac_total_df.set_index(pd.to_datetime(pac_total_df['Date']))['PortfolioValue']
            annual_returns_pac = calculate_annual_returns(pac_pv_series_for_annual)
            if not annual_returns_pac.empty:
                annual_returns_pac.index = annual_returns_pac.index.year # Usa l'anno come etichetta
                data_for_annual_hist["PAC"] = annual_returns_pac

        if not lump_sum_df.empty and 'PortfolioValue' in lump_sum_df and 'Date' in lump_sum_df:
            ls_pv_series_for_annual = lump_sum_df.set_index(pd.to_datetime(lump_sum_df['Date']))['PortfolioValue']
            annual_returns_ls = calculate_annual_returns(ls_pv_series_for_annual)
            if not annual_returns_ls.empty:
                annual_returns_ls.index = annual_returns_ls.index.year
                data_for_annual_hist["Lump Sum"] = annual_returns_ls
        
        if data_for_annual_hist:
            annual_hist_df = pd.DataFrame(data_for_annual_hist)
            # Rimuovi anni con solo NaN per evitare problemi con st.bar_chart se un df è più corto
            annual_hist_df.dropna(how='all', inplace=True) 
            if not annual_hist_df.empty:
                st.bar_chart(annual_hist_df)
            else:
                st.warning("Dati sui rendimenti annuali non sufficienti per l'istogramma dopo la pulizia.")
        else:
            st.warning("Non ci sono dati sufficienti per l'istogramma dei rendimenti annuali.")

        # STACKED AREA CHART (come prima)
        # ... (codice come prima, usando asset_details_history_df) ...
        if asset_details_history_df is not None and not asset_details_history_df.empty:
            st.subheader("Allocazione Dinamica del Portafoglio PAC nel Tempo (Valore per Asset)")
            value_cols_to_plot_stacked = [f'{ticker}_value' for ticker in tickers_list if f'{ticker}_value' in asset_details_history_df.columns]
            if value_cols_to_plot_stacked:
                stacked_area_df_data = asset_details_history_df.copy()
                if 'Date' in stacked_area_df_data.columns:
                    stacked_area_df_data['Date'] = pd.to_datetime(stacked_area_df_data['Date'])
                    stacked_area_df_data = stacked_area_df_data.set_index('Date')
                actual_cols_in_stacked_df = [col for col in value_cols_to_plot_stacked if col in stacked_area_df_data.columns]
                if actual_cols_in_stacked_df:
                    st.area_chart(stacked_area_df_data[actual_cols_in_stacked_df])
                else: st.warning("Nessuna colonna di valore per asset trovata per lo stacked area chart.")
            else: st.warning("Dati sui valori per asset non sufficienti per il grafico allocazione dinamica.")
        else: st.warning("Dati storici dettagliati per asset non disponibili per grafico allocazione dinamica.")

        # TABELLE QUOTE e WAP (come prima)
        # ... (codice come prima, usando asset_details_history_df) ...
        if asset_details_history_df is not None and not asset_details_history_df.empty:
            st.subheader("Dettagli Finali per Asset nel PAC")
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
            else: st.warning("Impossibile costruire la tabella dei dettagli finali per asset.")
        else: st.warning("Dati storici dettagliati per asset non disponibili per tabella Quote/WAP.")

        # CHECKBOX DATI DETTAGLIATI (come prima)
        # ... (codice come prima) ...
        if st.checkbox("Mostra dati aggregati dettagliati del PAC", key="pac_total_data_detail_main_v2"):
            st.dataframe(pac_total_df)
        if asset_details_history_df is not None and not asset_details_history_df.empty and \
           st.checkbox("Mostra dati storici dettagliati per asset del PAC", key="pac_asset_data_detail_main_v2"):
            st.dataframe(asset_details_history_df)
        if not lump_sum_df.empty and \
           st.checkbox("Mostra dati dettagliati della simulazione Lump Sum", key="ls_data_detail_main_key_v2"):
            st.dataframe(lump_sum_df)
            
else: 
    st.info("Inserisci i parametri nella sidebar a sinistra e avvia la simulazione.")

st.sidebar.markdown("---")
st.sidebar.markdown("Progetto Didattico Kriterion Quant")
