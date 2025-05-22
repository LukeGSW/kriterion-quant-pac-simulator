# simulatore_pac/utils/benchmark_engine.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta # timedelta non è usata qui, ma può rimanere
from dateutil.relativedelta import relativedelta # non usata qui, ma può rimanere

def run_lump_sum_simulation(
    historical_data_map: dict[str, pd.DataFrame],
    tickers: list[str],
    allocations: list[float], # Allocazioni target per il portafoglio Lump Sum
    total_investment_lump_sum: float,
    lump_sum_investment_date: pd.Timestamp, # Data effettiva dell'investimento LS (già pd.Timestamp)
    # simulation_start_date_tracking: pd.Timestamp, # Data da cui iniziare a TRACCIARE l'equity line
    # simulation_end_date_tracking: pd.Timestamp,   # Data fine per TRACCIARE l'equity line
    # I parametri sopra sono ora implicitamente gestiti dal range di historical_data_map
    reinvest_dividends: bool = True
) -> pd.DataFrame:
    """
    Simula un investimento Lump Sum e ne traccia l'evoluzione.
    La simulazione si estenderà per tutto il periodo coperto da historical_data_map[tickers[0]]
    a partire da lump_sum_investment_date (o la prima data di trading successiva).
    """
    if not (isinstance(historical_data_map, dict) and historical_data_map and tickers):
        # print("DEBUG (LS_engine): Mappa dati storici vuota o tickers mancanti.")
        return pd.DataFrame()
    if not all(ticker in historical_data_map for ticker in tickers):
        # print("DEBUG (LS_engine): Dati storici mancanti per alcuni ticker in mappa.")
        return pd.DataFrame()
    if total_investment_lump_sum <= 0:
        # print("DEBUG (LS_engine): Importo investimento Lump Sum non positivo.")
        return pd.DataFrame()
        
    portfolio_ls_details = {
        ticker: {'shares_owned': 0.0, 'dividends_cumulative_asset': 0.0, 'current_value': 0.0}
        for ticker in tickers
    }
    total_dividends_received_overall_ls = 0.0
    lump_sum_evolution_records = []
    actual_capital_invested_ls = 0.0 # Capitale effettivamente investito (potrebbe essere < total_investment_lump_sum se i prezzi non sono disponibili)

    # Usa il primo ticker come riferimento per il calendario di trading generale della simulazione LS
    reference_dates_df = historical_data_map[tickers[0]]

    # --- EFFETTUA L'INVESTIMENTO LUMP SUM INIZIALE ---
    # Trova il primo giorno di trading valido a partire da lump_sum_investment_date
    # nel DataFrame di riferimento. Questo sarà il giorno effettivo dell'investimento.
    potential_investment_days = reference_dates_df.index[reference_dates_df.index >= lump_sum_investment_date]
    if potential_investment_days.empty:
        # print(f"DEBUG (LS_engine): Nessuna data di trading valida trovata a partire da {lump_sum_investment_date.strftime('%Y-%m-%d')} per investimento Lump Sum nel reference ticker.")
        return pd.DataFrame()
    actual_investment_day_for_ls_setup = potential_investment_days[0]

    for i, ticker in enumerate(tickers):
        asset_data = historical_data_map[ticker]
        allocation_amount_for_ticker = total_investment_lump_sum * allocations[i]
        
        price_at_investment_for_ticker = np.nan
        # Verifica se actual_investment_day_for_ls_setup è un giorno di trading per QUESTO specifico asset
        if actual_investment_day_for_ls_setup in asset_data.index:
            price_at_investment_for_ticker = asset_data.loc[actual_investment_day_for_ls_setup, 'Adj Close']
        else: 
            # Se non lo è, cerca il *prossimo* giorno di trading disponibile per questo asset
            next_trading_day_for_this_asset_series = asset_data.index[asset_data.index >= actual_investment_day_for_ls_setup]
            if not next_trading_day_for_this_asset_series.empty:
                actual_investment_day_for_this_asset = next_trading_day_for_this_asset_series[0]
                price_at_investment_for_ticker = asset_data.loc[actual_investment_day_for_this_asset, 'Adj Close']
            # else: print(f"DEBUG (LS_engine): Nessun giorno di trading per {ticker} per l'investimento iniziale LS.")

        if pd.notna(price_at_investment_for_ticker) and price_at_investment_for_ticker > 0 and allocation_amount_for_ticker > 0:
            shares_bought = allocation_amount_for_ticker / price_at_investment_for_ticker
            portfolio_ls_details[ticker]['shares_owned'] = shares_bought
            actual_capital_invested_ls += allocation_amount_for_ticker # Somma solo se l'investimento avviene
        # else: print(f"DEBUG (LS_engine): Impossibile investire in {ticker} per LS il {actual_investment_day_for_ls_setup.strftime('%Y-%m-%d')}.")

    # --- TRACCIA L'EVOLUZIONE DEL PORTAFOGLIO LUMP SUM ---
    # Il periodo di tracciamento inizia dal giorno dell'investimento LS effettivo 
    # e va fino alla fine dei dati disponibili nel reference_dates_df.
    simulation_start_dt_ls_tracking = actual_investment_day_for_ls_setup
    simulation_end_dt_ls_tracking = reference_dates_df.index.max() # Fine dei dati del reference ticker

    tracking_period_dates_ls = reference_dates_df[
        (reference_dates_df.index >= simulation_start_dt_ls_tracking) &
        (reference_dates_df.index <= simulation_end_dt_ls_tracking)
    ].index

    if tracking_period_dates_ls.empty:
        # print("DEBUG (LS_engine): Nessuna data nel periodo di tracking per l'equity line LS.")
        return pd.DataFrame()

    for current_processing_date_ls in tracking_period_dates_ls:
        current_day_portfolio_value_ls = 0.0
        current_day_total_dividend_received_ls = 0.0

        for ticker in tickers:
            asset_data = historical_data_map[ticker]
            asset_ls_state = portfolio_ls_details[ticker]
            
            current_price_for_asset_ls = np.nan
            dividend_paid_by_asset_ls_today = 0.0

            if current_processing_date_ls in asset_data.index:
                current_price_for_asset_ls = asset_data.loc[current_processing_date_ls, 'Adj Close']
                dividend_paid_by_asset_ls_today = asset_data.loc[current_processing_date_ls, 'Dividend']
            else: 
                asset_data_up_to_today_ls = asset_data[asset_data.index <= current_processing_date_ls]
                if not asset_data_up_to_today_ls.empty:
                    current_price_for_asset_ls = asset_data_up_to_today_ls['Adj Close'].iloc[-1]
            
            # Reinvestimento Dividendi per Lump Sum
            if reinvest_dividends and dividend_paid_by_asset_ls_today > 0 and asset_ls_state['shares_owned'] > 0 and \
               pd.notna(current_price_for_asset_ls) and current_price_for_asset_ls > 0:
                cash_from_dividends_ls = asset_ls_state['shares_owned'] * dividend_paid_by_asset_ls_today
                asset_ls_state['dividends_cumulative_asset'] += cash_from_dividends_ls
                current_day_total_dividend_received_ls += cash_from_dividends_ls
                additional_shares_ls = cash_from_dividends_ls / current_price_for_asset_ls
                asset_ls_state['shares_owned'] += additional_shares_ls
            elif dividend_paid_by_asset_ls_today > 0 and asset_ls_state['shares_owned'] > 0:
                 cash_from_dividends_ls = asset_ls_state['shares_owned'] * dividend_paid_by_asset_ls_today
                 asset_ls_state['dividends_cumulative_asset'] += cash_from_dividends_ls
                 current_day_total_dividend_received_ls += cash_from_dividends_ls
            
            if pd.notna(current_price_for_asset_ls):
                asset_ls_state['current_value'] = asset_ls_state['shares_owned'] * current_price_for_asset_ls
            current_day_portfolio_value_ls += asset_ls_state.get('current_value', 0.0)

        total_dividends_received_overall_ls += current_day_total_dividend_received_ls

        lump_sum_evolution_records.append({
            'Date': current_processing_date_ls,
            'InvestedCapital': actual_capital_invested_ls, # Capitale investito rimane costante
            'PortfolioValue': current_day_portfolio_value_ls,
            'DividendsReceivedCumulative': total_dividends_received_overall_ls
        })

    if not lump_sum_evolution_records:
        # print("DEBUG (LS_engine): Nessun record di evoluzione del portafoglio LS generato.")
        return pd.DataFrame()

    final_df_ls = pd.DataFrame(lump_sum_evolution_records)
    if 'Date' in final_df_ls.columns: final_df_ls['Date'] = pd.to_datetime(final_df_ls['Date'])
    
    return final_df_ls.reset_index(drop=True)
