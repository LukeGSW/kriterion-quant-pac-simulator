# simulatore_pac/main.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from io import BytesIO # Per salvare le immagini Matplotlib in memoria

# Importazioni per Matplotlib (necessarie per generare i grafici per il PDF)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Importazioni dai moduli utils
try:
    from utils.data_loader import load_historical_data_yf
    from utils.pac_engine import run_pac_simulation
    from utils.benchmark_engine import run_lump_sum_simulation
    from utils.report_generator import generate_pac_report_pdf # Per il PDF
    from utils.performance import (
        get_total_capital_invested, get_final_portfolio_value,
        calculate_total_return_percentage, calculate_cagr, get_duration_years,
        calculate_portfolio_returns, calculate_annualized_volatility,
        calculate_sharpe_ratio, calculate_max_drawdown, calculate_drawdown_series,
        generate_cash_flows_for_xirr, calculate_xirr_metric,
        get_final_asset_details, calculate_wap_for_assets,
        calculate_tracking_error
    )
    IMPORT_SUCCESS = True
except ImportError as import_err:
    IMPORT_SUCCESS = False
    IMPORT_ERROR_MESSAGE = str(import_err)

st.set_page_config(page_title="Simulatore PAC con Report PDF", layout="wide")
st.title("📘 Simulatore PAC con Report PDF")
st.caption("Progetto Kriterion Quant")

if not IMPORT_SUCCESS:
    st.error(f"Errore critico durante l'importazione dei moduli utils: {IMPORT_ERROR_MESSAGE}")
    st.stop()

# --- FUNZIONI HELPER PER CREARE FIGURE MATPLOTLIB ---
def create_equity_line_fig(equity_plot_df_input, pac_contribution_end_dt_input): # Rinominato per evitare conflitti
    if equity_plot_df_input.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 5)) 
    
    # Assicura che l'indice sia Datetime per il plot
    equity_plot_df_mpl = equity_plot_df_input.copy()
    if not isinstance(equity_plot_df_mpl.index, pd.DatetimeIndex):
        equity_plot_df_mpl.index = pd.to_datetime(equity_plot_df_mpl.index)

    cols_to_plot_mpl = []
    if 'PAC Valore Portafoglio' in equity_plot_df_mpl.columns: cols_to_plot_mpl.append('PAC Valore Portafoglio')
    if 'PAC Capitale Investito' in equity_plot_df_mpl.columns: cols_to_plot_mpl.append('PAC Capitale Investito')
    if 'Lump Sum Valore Portafoglio' in equity_plot_df_mpl.columns: cols_to_plot_mpl.append('Lump Sum Valore Portafoglio')
    if 'Cash (Valore Fisso 0%)' in equity_plot_df_mpl.columns: cols_to_plot_mpl.append('Cash (Valore Fisso 0%)')

    for col in cols_to_plot_mpl:
        if col in equity_plot_df_mpl.columns and not equity_plot_df_mpl[col].isnull().all():
            ax.plot(equity_plot_df_mpl.index, equity_plot_df_mpl[col], label=col)

    ax.set_title("Andamento Comparativo del Portafoglio")
    ax.set_xlabel("Data")
    ax.set_ylabel("Valore")
    ax.legend(loc='upper left', fontsize='small')
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig

def create_drawdown_fig(drawdown_plot_df_input): # Rinominato per evitare conflitti
    if drawdown_plot_df_input.empty or drawdown_plot_df_input.isnull().all().all():
        return None
    fig, ax = plt.subplots(figsize=(10, 4))

    # Assicura che l'indice sia Datetime per il plot
    drawdown_plot_df_mpl = drawdown_plot_df_input.copy()
    if not isinstance(drawdown_plot_df_mpl.index, pd.DatetimeIndex):
        drawdown_plot_df_mpl.index = pd.to_datetime(drawdown_plot_df_mpl.index)

    cols_to_plot_dd_mpl = [c for c in drawdown_plot_df_mpl.columns if not drawdown_plot_df_mpl[c].isnull().all()]
    if not cols_to_plot_dd_mpl: return None

    for col in cols_to_plot_dd_mpl:
        ax.plot(drawdown_plot_df_mpl.index, drawdown_plot_df_mpl[col], label=col)
        
    ax.set_title("Andamento del Drawdown nel Tempo (%)")
    ax.set_xlabel("Data")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc='lower left', fontsize='small')
    ax.grid(True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter('{:.0f}%'.format))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig

def create_stacked_area_fig(stacked_area_df_plot_input, tickers_list_input): # Rinominato
    if stacked_area_df_plot_input.empty or stacked_area_df_plot_input.isnull().all().all():
        return None
        
    value_cols_only_mpl = [f'{t}_value' for t in tickers_list_input if f'{t}_value' in stacked_area_df_plot_input.columns]
    if not value_cols_only_mpl: return None

    # Assicura che l'indice sia Datetime per il plot
    stacked_area_df_plot_mpl = stacked_area_df_plot_input.copy()
    if not isinstance(stacked_area_df_plot_mpl.index, pd.DatetimeIndex):
        stacked_area_df_plot_mpl.index = pd.to_datetime(stacked_area_df_plot_mpl.index)

    fig, ax = plt.subplots(figsize=(10, 5))
    # Prepara i dati per stackplot: lista di serie 1D o array 2D (N, M)
    data_to_stack = [stacked_area_df_plot_mpl[col].values for col in value_cols_only_mpl]
    
    ax.stackplot(stacked_area_df_plot_mpl.index, 
                 *data_to_stack, # Spacchetta la lista di array
                 labels=value_cols_only_mpl,
                 alpha=0.7)
    
    ax.set_title("Allocazione Dinamica Portafoglio PAC (Valore per Asset)")
    ax.set_xlabel("Data")
    ax.set_ylabel("Valore Asset")
    ax.legend(loc='upper left', fontsize='small')
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig


# --- Sidebar per Input Utente (come ultima versione stabile) ---
st.sidebar.header("Parametri Simulazione")
st.sidebar.subheader("Asset e Allocazioni")
tickers_input_str = st.sidebar.text_input("Tickers (virgola sep.)", "AAPL,GOOG,MSFT", key="ui_tickers_v8")
allocations_input_str = st.sidebar.text_input("Allocazioni % (virgola sep.)", "60,20,20", key="ui_allocations_v8")
st.sidebar.subheader("Parametri PAC")
monthly_investment_input = st.sidebar.number_input("Importo Mensile (€/$)", 10.0, value=200.0, step=10.0, key="ui_monthly_inv_v8")
default_start_date_pac_sidebar = date(2020, 1, 1)
pac_start_date_contributions_ui = st.sidebar.date_input("Data Inizio Contributi PAC", default_start_date_pac_sidebar, key="ui_pac_start_date_v8")
default_duration_months = 36
duration_months_contributions_input = st.sidebar.number_input("Durata Contributi PAC (mesi)", 6, value=default_duration_months, step=1, key="ui_duration_months_v8")
reinvest_dividends_input = st.sidebar.checkbox("Reinvesti Dividendi (per PAC e LS)?", True, key="ui_reinvest_div_v8")
st.sidebar.subheader("Ribilanciamento Periodico (Solo per PAC)")
rebalance_active_input = st.sidebar.checkbox("Attiva Ribilanciamento (per PAC)?", False, key="ui_rebalance_active_v8")
rebalance_frequency_input_str = None
if rebalance_active_input:
    rebalance_frequency_input_str = st.sidebar.selectbox("Frequenza", ["Annuale", "Semestrale", "Trimestrale"], 0, key="ui_rebalance_freq_v8")
st.sidebar.subheader("Parametri Metriche")
risk_free_rate_input = st.sidebar.number_input("Tasso Risk-Free Annuale (%) per Sharpe", 0.0, value=1.0, step=0.1, format="%.2f", key="ui_rf_rate_v8")
run_simulation_button = st.sidebar.button("🚀 Avvia Simulazioni", key="ui_run_button_v8")

st.sidebar.markdown("---") 
st.sidebar.subheader("Guida Rapida")
st.sidebar.caption(
    "1. **Asset**: Inserisci i simboli dei ticker (es. AAPL, VWCE.DE) separati da virgole.\n"
    "2. **Allocazioni**: Definisci le percentuali per ogni asset (es. 60,40). La somma deve essere 100.\n"
    "3. **PAC**: Imposta l'importo mensile, la data di inizio dei versamenti e la loro durata in mesi.\n"
    "4. **Opzioni**: Scegli se reinvestire i dividendi e se attivare il ribilanciamento periodico per il PAC.\n"
    "5. **Avvia**: Clicca 'Avvia Simulazioni' per vedere i risultati."
)
st.sidebar.markdown("---")
st.sidebar.markdown("Visita il nostro sito: [Kriterion Quant](https://kriterionquant.com/)", unsafe_allow_html=False)
st.sidebar.markdown("Progetto Didattico Kriterion Quant")


if run_simulation_button:
    # --- VALIDAZIONE INPUT (come ultima versione stabile) ---
    tickers_list = [t.strip().upper() for t in tickers_input_str.split(',') if t.strip()] # ... (resto validazione)
    error_in_input = False; allocations_list_norm = []; allocations_float_list_raw = []
    if not tickers_list: st.error("Errore: Devi inserire almeno un ticker."); error_in_input = True
    if not error_in_input:
        try:
            allocations_float_list_raw = [float(a.strip()) for a in allocations_input_str.split(',') if a.strip()]
            if len(tickers_list)!=len(allocations_float_list_raw): st.error("Errore: N. ticker diverso da N. allocazioni."); error_in_input=True
            elif not np.isclose(sum(allocations_float_list_raw),100.0): st.error(f"Errore: Somma allocazioni ({sum(allocations_float_list_raw)}%) != 100%."); error_in_input=True
            else: allocations_list_norm = [a/100.0 for a in allocations_float_list_raw]
        except ValueError: st.error("Errore: Allocazioni non numeriche."); error_in_input=True
    if error_in_input: st.stop()

    st.header(f"Risultati Simulazione per: {', '.join(tickers_list)}")
    # ... (Info allocazioni e ribilanciamento)

    # --- PREPARAZIONE DATE E CARICAMENTO DATI (come ultima versione stabile) ---
    pac_contribution_start_dt = pd.to_datetime(pac_start_date_contributions_ui) # ... (resto come prima)
    pac_contribution_start_str = pac_contribution_start_dt.strftime('%Y-%m-%d')
    pac_contribution_end_dt = pac_contribution_start_dt + pd.DateOffset(months=duration_months_contributions_input)
    data_download_start_str = (pac_contribution_start_dt - pd.Timedelta(days=365*1)).strftime('%Y-%m-%d')
    data_download_end_str = datetime.today().strftime('%Y-%m-%d')
    historical_data_map = {}; all_data_loaded_successfully = True; latest_overall_data_date_ts = pd.Timestamp.min
    for tkr in tickers_list:
        with st.spinner(f"Dati per {tkr}..."): data = load_historical_data_yf(tkr, data_download_start_str, data_download_end_str)
        if data.empty or data.index.min() > pac_contribution_start_dt or data.index.max() < (pac_contribution_end_dt - pd.Timedelta(days=1)):
            st.error(f"Dati storici insufficienti per {tkr} per periodo PAC. Ultima data per {tkr}: {data.index.max().date() if not data.empty else 'N/A'}.")
            all_data_loaded_successfully = False; break
        historical_data_map[tkr] = data
        if data.index.max() > latest_overall_data_date_ts: latest_overall_data_date_ts = data.index.max()
    if not all_data_loaded_successfully: st.stop()
    st.success("Dati storici caricati.")
    simulation_actual_end_dt = min(latest_overall_data_date_ts, pd.Timestamp(datetime.today()))
    base_chart_date_index = pd.DatetimeIndex([])
    if pac_contribution_start_dt <= simulation_actual_end_dt:
        base_chart_date_index = pd.date_range(start=pac_contribution_start_dt, end=simulation_actual_end_dt, freq='B')
        base_chart_date_index.name = 'Date'
    if base_chart_date_index.empty: st.warning("Indice date per grafici non generato.")

    # --- ESECUZIONE SIMULAZIONI (come ultima versione stabile) ---
    pac_total_df, asset_details_history_df = pd.DataFrame(), pd.DataFrame() # ... (come prima)
    try:
        with st.spinner("Simulazione PAC..."): pac_total_df, asset_details_history_df = run_pac_simulation(historical_data_map, tickers_list, allocations_list_norm, monthly_investment_input, pac_contribution_start_str, duration_months_contributions_input, reinvest_dividends_input, rebalance_active_input, rebalance_frequency_input_str)
    except Exception as e: st.error(f"Errore CRITICO PAC: {e}"); import traceback; st.text(traceback.format_exc()); st.stop()
    lump_sum_df = pd.DataFrame()
    if not pac_total_df.empty and 'PortfolioValue' in pac_total_df.columns and len(pac_total_df)>=2:
        total_invested_by_pac = get_total_capital_invested(pac_total_df)
        if total_invested_by_pac > 0:
            with st.spinner("Simulazione LS..."): lump_sum_df = run_lump_sum_simulation(historical_data_map, tickers_list, allocations_list_norm, total_invested_by_pac, pac_contribution_start_dt, reinvest_dividends_input)
            if not lump_sum_df.empty: st.success("Simulazione LS completata.")
    if pac_total_df.empty or 'PortfolioValue' not in pac_total_df.columns or len(pac_total_df)<2: st.error("Simulazione PAC fallita."); st.stop()
    st.success("Simulazioni OK. Elaborazione output.")

    # --- FUNZIONE HELPER PER METRICHE (come ultima versione stabile) ---
    def calculate_and_format_metrics(sim_df, strategy_name, total_invested_override=None, is_pac=False, benchmark_returns_for_te=None):
        # ... (COPIA QUI LA TUA ULTIMA VERSIONE FUNZIONANTE di calculate_and_format_metrics)
        metrics_display = {}; keys_ordered_pac = ["Capitale Investito", "Valore Finale", "Rend. Totale", "CAGR", "XIRR", "Sharpe", "Max Drawdown"];
        if is_pac and benchmark_returns_for_te is not None and not benchmark_returns_for_te.empty: keys_ordered_pac.append("Tracking Error (vs LS)")
        keys_ordered_ls = ["Capitale Investito", "Valore Finale", "Rend. Totale", "CAGR", "XIRR", "Volatilità Ann.", "Sharpe", "Max Drawdown"]
        current_keys = keys_ordered_pac if is_pac else keys_ordered_ls
        for k in current_keys: metrics_display[k] = "N/A"
        if sim_df.empty or 'PortfolioValue' not in sim_df.columns or len(sim_df) < 2: return metrics_display
        actual_total_invested = total_invested_override if total_invested_override is not None else get_total_capital_invested(sim_df)
        metrics_display["Capitale Investito"] = f"{actual_total_invested:,.2f}"; final_portfolio_val = get_final_portfolio_value(sim_df)
        metrics_display["Valore Finale"] = f"{final_portfolio_val:,.2f}"; rend_tot = calculate_total_return_percentage(final_portfolio_val, actual_total_invested)
        metrics_display["Rend. Totale"] = f"{rend_tot:.2f}%" if pd.notna(rend_tot) else "N/A"; duration_yrs_strat = get_duration_years(sim_df.copy()); cagr_strat = calculate_cagr(final_portfolio_val, actual_total_invested, duration_yrs_strat)
        metrics_display["CAGR"] = f"{cagr_strat:.2f}%" if pd.notna(cagr_strat) else "N/A"; returns_strat = calculate_portfolio_returns(sim_df.copy())
        vol_strat_numeric = calculate_annualized_volatility(returns_strat)
        if not is_pac: metrics_display["Volatilità Ann."] = f"{vol_strat_numeric:.2f}%" if pd.notna(vol_strat_numeric) else "N/A"
        sharpe_strat = calculate_sharpe_ratio(returns_strat, risk_free_rate_annual=(risk_free_rate_input/100.0))
        metrics_display["Sharpe"] = f"{sharpe_strat:.2f}" if pd.notna(sharpe_strat) else "N/A"; mdd_strat = calculate_max_drawdown(sim_df.copy())
        metrics_display["Max Drawdown"] = f"{mdd_strat:.2f}%" if pd.notna(mdd_strat) else "N/A"
        if is_pac:
            xirr_dates, xirr_values = generate_cash_flows_for_xirr(sim_df, pac_contribution_start_str, duration_months_contributions_input, monthly_investment_input, final_portfolio_val)
            xirr_val = calculate_xirr_metric(xirr_dates, xirr_values); metrics_display["XIRR"] = f"{xirr_val:.2f}%" if pd.notna(xirr_val) else "N/A"
            if benchmark_returns_for_te is not None and not benchmark_returns_for_te.empty and "Tracking Error (vs LS)" in metrics_display:
                te_strat = calculate_tracking_error(returns_strat, benchmark_returns_for_te); metrics_display["Tracking Error (vs LS)"] = f"{te_strat:.2f}%" if pd.notna(te_strat) else "N/A"
        else: metrics_display["XIRR"] = metrics_display["CAGR"] 
        return metrics_display
        
    ls_daily_returns_for_te_calc = pd.Series(dtype=float)
    if not lump_sum_df.empty: ls_daily_returns_for_te_calc = calculate_portfolio_returns(lump_sum_df.copy())
    metrics_pac_display = calculate_and_format_metrics(pac_total_df, "PAC", is_pac=True, benchmark_returns_for_te=ls_daily_returns_for_te_calc)
    desired_ordered_metrics = ["Capitale Investito", "Valore Finale", "Rend. Totale", "CAGR", "XIRR", "Volatilità Ann.", "Sharpe", "Max Drawdown", "Tracking Error (vs LS)"]
    df_for_table = pd.DataFrame(index=desired_ordered_metrics); df_for_table["PAC"] = df_for_table.index.map(metrics_pac_display).fillna("N/A")
    if not lump_sum_df.empty:
        metrics_ls_display = calculate_and_format_metrics(lump_sum_df, "Lump Sum", total_invested_override=get_total_capital_invested(pac_total_df), is_pac=False)
        df_for_table["Lump Sum"] = df_for_table.index.map(metrics_ls_display).fillna("N/A")
    if "Volatilità Ann." in df_for_table.index: df_for_table.loc["Volatilità Ann.", "PAC"] = "N/A"
    if "Lump Sum" not in df_for_table.columns and "Tracking Error (vs LS)" in df_for_table.index: df_for_table.drop("Tracking Error (vs LS)", inplace=True)
    elif "Tracking Error (vs LS)" in df_for_table.index and df_for_table.loc["Tracking Error (vs LS)", "PAC"] == "N/A":
         if "Lump Sum" not in df_for_table.columns: df_for_table.drop("Tracking Error (vs LS)", inplace=True)
    final_ordered_index = [label for label in desired_ordered_metrics if label in df_for_table.index]
    df_for_table = df_for_table.reindex(final_ordered_index).fillna("N/A")
    st.subheader("Metriche di Performance Riepilogative"); st.table(df_for_table)

    # --- PREPARAZIONE DATAFRAMES PER GRAFICI ESTESI (USANDO base_chart_date_index) ---
    equity_plot_df_final = pd.DataFrame(index=base_chart_date_index)
    drawdown_plot_df_final = pd.DataFrame(index=base_chart_date_index)
    stacked_area_df_final = pd.DataFrame(index=base_chart_date_index)

    if not pac_total_df.empty:
        pac_equity_data = pac_total_df.set_index(pd.to_datetime(pac_total_df['Date']))
        equity_plot_df_final = equity_plot_df_final.join(pac_equity_data[['PortfolioValue', 'InvestedCapital']])
        equity_plot_df_final.rename(columns={'PortfolioValue': 'PAC Valore Portafoglio', 'InvestedCapital': 'PAC Capitale Investito'}, inplace=True)
        pac_val_series_dd = pac_equity_data['PortfolioValue']
        drawdown_plot_df_final['PAC Drawdown (%)'] = calculate_drawdown_series(pac_val_series_dd)
    if not lump_sum_df.empty:
        ls_equity_data = lump_sum_df.set_index(pd.to_datetime(lump_sum_df['Date']))
        equity_plot_df_final = equity_plot_df_final.join(ls_equity_data[['PortfolioValue']])
        equity_plot_df_final.rename(columns={'PortfolioValue': 'Lump Sum Valore Portafoglio'}, inplace=True)
        ls_val_series_dd = ls_equity_data['PortfolioValue']
        drawdown_plot_df_final['Lump Sum Drawdown (%)'] = calculate_drawdown_series(ls_val_series_dd)
    cash_val_bm = get_total_capital_invested(pac_total_df) if not pac_total_df.empty else 0
    if cash_val_bm > 0: equity_plot_df_final['Cash (Valore Fisso 0%)'] = cash_val_bm
    
    cols_ffill_equity = ['PAC Valore Portafoglio', 'Lump Sum Valore Portafoglio', 'Cash (Valore Fisso 0%)', 'PAC Capitale Investito']
    for col in cols_ffill_equity:
        if col in equity_plot_df_final.columns: equity_plot_df_final[col] = equity_plot_df_final[col].ffill()
    if 'PAC Capitale Investito' in equity_plot_df_final.columns and not pac_total_df.empty:
        equity_plot_df_final['PAC Capitale Investito'] = equity_plot_df_final['PAC Capitale Investito'].ffill()
        last_pac_contrib_dt_fill = pac_contribution_start_dt + pd.DateOffset(months=duration_months_contributions_input -1)
        if not base_chart_date_index.empty:
             idx_loc_s = base_chart_date_index.get_indexer([last_pac_contrib_dt_fill], method='ffill')
             if idx_loc_s.size > 0 and idx_loc_s[0] != -1:
                actual_last_contrib_idx_dt = base_chart_date_index[idx_loc_s[0]]
                last_known_cap = equity_plot_df_final.loc[actual_last_contrib_idx_dt, 'PAC Capitale Investito']
                if pd.notna(last_known_cap): equity_plot_df_final.loc[equity_plot_df_final.index > actual_last_contrib_idx_dt, 'PAC Capitale Investito'] = last_known_cap
    
    for col in drawdown_plot_df_final.columns: drawdown_plot_df_final[col] = drawdown_plot_df_final[col].ffill()

    if asset_details_history_df is not None and not asset_details_history_df.empty:
        value_cols_stacked_list = [f'{t}_value' for t in tickers_list if f'{t}_value' in asset_details_history_df.columns]
        if value_cols_stacked_list:
            stacked_df_temp_data = asset_details_history_df.set_index(pd.to_datetime(asset_details_history_df['Date']))[value_cols_stacked_list]
            for col in stacked_df_temp_data.columns:
                stacked_area_df_final = stacked_area_df_final.join(stacked_df_temp_data[[col]], how='left')
                stacked_area_df_final[col] = stacked_area_df_final[col].ffill().fillna(0)

    # --- VISUALIZZAZIONE GRAFICI ---
    st.subheader("Andamento Comparativo del Portafoglio")
    cols_to_plot_equity_final = [c for c in equity_plot_df_final.columns if not equity_plot_df_final[c].isnull().all()]
    if cols_to_plot_equity_final and not base_chart_date_index.empty: st.line_chart(equity_plot_df_final[cols_to_plot_equity_final])
    else: st.warning("Dati insufficienti per grafico equity.")

    st.subheader("Andamento del Drawdown nel Tempo")
    cols_to_plot_dd_final = [c for c in drawdown_plot_df_final.columns if not drawdown_plot_df_final[c].isnull().all()]
    if cols_to_plot_dd_final and not base_chart_date_index.empty: st.line_chart(drawdown_plot_df_final[cols_to_plot_dd_final])
    else: st.warning("Dati insufficienti per grafico drawdown.")

    st.subheader("Allocazione Dinamica Portafoglio PAC (Valore per Asset)")
    actual_cols_in_stacked_df_final = [c for c in stacked_area_df_final.columns if not stacked_area_df_final[c].isnull().all()]
    if actual_cols_in_stacked_df_final and not base_chart_date_index.empty: st.area_chart(stacked_area_df_final[actual_cols_in_stacked_df_final])
    else: st.warning("Dati insufficienti per grafico allocazione dinamica.")
    
    # --- TABELLA QUOTE/WAP ---
    if asset_details_history_df is not None and not asset_details_history_df.empty:
        st.subheader("Dettagli Finali per Asset nel PAC")
        # ... (Logica tabella WAP come prima) ...
        final_asset_details_map = get_final_asset_details(asset_details_history_df, tickers_list)
        wap_map = calculate_wap_for_assets(final_asset_details_map)
        table_data_wap = []
        for ticker_wap in tickers_list:
            asset_info = final_asset_details_map.get(ticker_wap, {'shares': 0.0, 'capital_invested': 0.0})
            table_data_wap.append({"Ticker": ticker_wap, "Quote Finali": f"{asset_info['shares']:.4f}", "Capitale Investito (Asset)": f"{asset_info['capital_invested']:,.2f}", "Prezzo Medio Carico (WAP)": f"{wap_map.get(ticker_wap, np.nan):,.2f}" if pd.notna(wap_map.get(ticker_wap, np.nan)) else "N/A"})
        if table_data_wap: st.table(pd.DataFrame(table_data_wap).set_index("Ticker"))


    # --- SEZIONE DOWNLOAD DATI (inclusa preparazione per PDF) ---
    st.subheader("Download Dati e Report")
    # ... (Pulsanti CSV come prima) ...
    if 'df_for_table' in locals() and not df_for_table.empty: st.download_button(label="Scarica Metriche (CSV)", data=df_for_table.reset_index().to_csv(index=False).encode('utf-8'), file_name="metriche.csv", mime='text/csv', key='dl_m_v8')
    if not pac_total_df.empty: st.download_button(label="Scarica Evoluzione PAC (CSV)", data=pac_total_df.to_csv(index=False).encode('utf-8'), file_name="pac.csv", mime='text/csv', key='dl_pac_v8')
    if not lump_sum_df.empty: st.download_button(label="Scarica Evoluzione LS (CSV)", data=lump_sum_df.to_csv(index=False).encode('utf-8'), file_name="ls.csv", mime='text/csv', key='dl_ls_v8')
    if asset_details_history_df is not None and not asset_details_history_df.empty: st.download_button(label="Scarica Dettagli Asset PAC (CSV)", data=asset_details_history_df.to_csv(index=False).encode('utf-8'), file_name="pac_asset_details.csv", mime='text/csv', key='dl_ad_v8')
    
    # Preparazione dati e figure per PDF
    if not pac_total_df.empty and 'df_for_table' in locals() and not df_for_table.empty :
        pac_params_for_pdf = {
            "start_date": pac_start_date_contributions_ui.strftime('%Y-%m-%d'), "duration_months": duration_months_contributions_input,
            "monthly_investment": f"{monthly_investment_input:,.2f}", "reinvest_div": reinvest_dividends_input,
            "rebalance_active": rebalance_active_input, "rebalance_freq": rebalance_frequency_input_str if rebalance_active_input else "N/A"
        }
        asset_details_final_for_pdf_table = pd.DataFrame()
        if asset_details_history_df is not None and not asset_details_history_df.empty:
            # ... (logica per table_data_wap_pdf e asset_details_final_for_pdf_table come prima) ...
            final_asset_details_map_pdf = get_final_asset_details(asset_details_history_df, tickers_list)
            wap_map_pdf = calculate_wap_for_assets(final_asset_details_map_pdf)
            table_data_wap_pdf = []
            for ticker_wap_pdf in tickers_list:
                asset_info_pdf = final_asset_details_map_pdf.get(ticker_wap_pdf, {'shares': 0.0, 'capital_invested': 0.0})
                table_data_wap_pdf.append({"Ticker": ticker_wap_pdf, "Quote Finali": f"{asset_info_pdf['shares']:.4f}", "Cap.Inv.(Asset)": f"{asset_info_pdf['capital_invested']:,.2f}", "WAP": f"{wap_map_pdf.get(ticker_wap_pdf, np.nan):,.2f}" if pd.notna(wap_map_pdf.get(ticker_wap_pdf, np.nan)) else "N/A"})
            if table_data_wap_pdf: asset_details_final_for_pdf_table = pd.DataFrame(table_data_wap_pdf).set_index("Ticker")


        # Genera figure Matplotlib
        equity_fig_mpl, drawdown_fig_mpl, stacked_area_fig_mpl = None, None, None
        if 'equity_plot_df' in locals() and not equity_plot_df.empty and actual_cols_to_plot_equity:
            equity_fig_mpl = create_equity_line_fig(equity_plot_df[actual_cols_to_plot_equity].copy(), pac_contribution_end_dt) # Passa solo colonne valide
        
        if 'drawdown_plot_df' in locals() and not drawdown_plot_df.empty and cols_to_plot_dd: # cols_to_plot_dd è definito sopra
            drawdown_fig_mpl = create_drawdown_fig(drawdown_plot_df[cols_to_plot_dd].copy())

        if 'stacked_area_df_final' in locals() and not stacked_area_df_final.empty and actual_cols_in_stacked_df_final : # actual_cols_in_stacked_df_final definito sopra
             stacked_area_fig_mpl = create_stacked_area_fig(stacked_area_df_final[actual_cols_in_stacked_df_final].copy(), tickers_list)
        
        try:
            pdf_bytes = generate_pac_report_pdf(
                tickers_list=tickers_list, allocations_float_list_raw=allocations_float_list_raw,
                pac_params=pac_params_for_pdf, metrics_df=df_for_table.reset_index(), 
                asset_details_final_df=asset_details_final_for_pdf_table.reset_index(),
                equity_line_fig=equity_fig_mpl, drawdown_fig=drawdown_fig_mpl, stacked_area_fig=stacked_area_fig_mpl
            )
           # In main.py, dopo la chiamata a generate_pac_report_pdf


                
                # --- CHIUDI LE FIGURE MATPLOTLIB PER LIBERARE MEMORIA ---
            if equity_fig_mpl: plt.close(equity_fig_mpl)
            if drawdown_fig_mpl: plt.close(drawdown_fig_mpl)
            if stacked_area_fig_mpl: plt.close(stacked_area_fig_mpl)
                # --- FINE CHIUSURA FIGURE ---
                
        st.download_button(
            label="Scarica Report PDF Completo", # Etichetta aggiornata
            data=pdf_bytes,
            file_name=f"report_pac_{'_'.join(tickers_list)}.pdf",
            mime='application/pdf',
            key='dl_report_pdf_v9' # Chiave aggiornata
                )
        except Exception as e_pdf:
            st.warning(f"Impossibile generare il report PDF: {e_pdf}")
            import traceback
            st.text(f"Dettagli errore PDF:\n{traceback.format_exc()}") # Stampa il traceback sulla pagina
else: 
    st.info("Inserisci parametri e avvia simulazione.")

st.sidebar.markdown("---"); st.sidebar.markdown("Kriterion Quant")
