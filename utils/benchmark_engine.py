# simulatore_pac/utils/benchmark_engine.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def run_lump_sum_simulation(
    historical_data_map: dict[str, pd.DataFrame],
    tickers: list[str],
    allocations: list[float], # Allocazioni target per il portafoglio Lump Sum
    total_investment_lump_sum: float,
    lump_sum_investment_date: pd.Timestamp, # Data effettiva (già pd.Timestamp)
    simulation_start_date: pd.Timestamp, # Data da cui iniziare a tracciare l'equity line
    simulation_end_date: pd.Timestamp,   # Data fine per l'equity line
    reinvest_dividends: bool = True
) -> pd.DataFrame:
    """
    Simula un investimento Lump Sum e ne traccia l'evoluzione.
    """
    if not tickers or not historical_data_map:
        print("ERRORE (Lump Sum): Lista ticker o dati storici mancanti.")
        return pd.DataFrame()

    # Inizializzazione stato portafoglio per ogni asset
    portfolio_ls_details = {
        ticker: {'shares_owned': 0.0, 'dividends_cumulative_asset': 0.0, 'current_value': 0.0}
        for ticker in tickers
    }
    total_dividends_received_overall_ls = 0.0
    lump_sum_evolution_records = []

    # --- EFFETTUA L'INVESTIMENTO LUMP SUM INIZIALE ---
    capital_invested_ls = 0.0 # Tracciamo il capitale effettivamente investito se un asset non ha prezzo
    
    # Trova il primo giorno di trading valido per l'investimento Lump Sum
    # (dovrebbe essere la stessa logica usata per il primo investimento PAC)
    # Per semplicità, assumiamo che lump_sum_investment_date sia un giorno di trading valido
    # o che useremo il prezzo del primo giorno di trading valido subito dopo.
    
    # Usiamo il DataFrame del primo ticker come riferimento per le date di trading generali
    reference_dates_df = historical_data_map[tickers[0]]
    
    # Trova il primo giorno di trading effettivo a partire da lump_sum_investment_date
    # nel DataFrame di riferimento
    actual_investment_date_series = reference_dates_df.index[reference_dates_df.index >= lump_sum_investment_date]
    if actual_investment_date_series.empty:
        print(f"ERRORE (Lump Sum): Nessuna data di trading valida trovata a partire da {lump_sum_investment_date.strftime('%Y-%m-%d')} per l'investimento Lump Sum.")
        return pd.DataFrame()
    actual_investment_day = actual_investment_date_series[0]


    for i, ticker in enumerate(tickers):
        asset_data = historical_data_map[ticker]
        allocation_amount = total_investment_lump_sum * allocations[i]
        
        price_at_investment = np.nan
        if actual_investment_day in asset_data.index:
            price_at_investment = asset_data.loc[actual_investment_day, 'Adj Close']
        else: # Se actual_investment_day non è un giorno di trading per QUESTO asset, cerca il prossimo
            next_trading_day_asset_series = asset_data.index[asset_data.index >= actual_investment_day]
            if not next_trading_day_asset_series.empty:
                actual_investment_day_asset = next_trading_day_asset_series[0]
                price_at_investment = asset_data.loc[actual_investment_day_asset, 'Adj Close']
            else: # Non ci sono giorni di trading successivi per questo asset nel range di dati
                print(f"ATTENZIONE (Lump Sum): Nessun giorno di trading per {ticker} trovato a partire da {actual_investment_day.strftime('%Y-%m-%d')}. Impossibile investire in questo asset.")

        if pd.notna(price_at_investment) and price_at_investment > 0 and allocation_amount > 0:
            shares_bought = allocation_amount / price_at_investment
            portfolio_ls_details[ticker]['shares_owned'] = shares_bought
            capital_invested_ls += allocation_amount # Solo se l'investimento è andato a buon fine
            # print(f"Debug LS: Investito in {ticker} il {actual_investment_day.date()}: {shares_bought:.4f} quote a {price_at_investment:.2f}")
        else:
            print(f"ATTENZIONE (Lump Sum): Impossibile investire {allocation_amount:.2f} in {ticker} il {actual_investment_day.strftime('%Y-%m-%d')} (prezzo: {price_at_investment}).")

    # --- TRACCIA L'EVOLUZIONE DEL PORTAFOGLIO LUMP SUM ---
    # Definiamo il periodo per cui tracciare l'equity line del Lump Sum
    # Deve iniziare da simulation_start_date (che è l'inizio del PAC)
    # e finire a simulation_end_date (fine del PAC).
    # L'investimento effettivo del LS è avvenuto su actual_investment_day.
    
    tracking_period_dates = reference_dates_df[
        (reference_dates_df.index >= simulation_start_date) & # Inizia a tracciare da quando inizia il PAC
        (reference_dates_df.index <= simulation_end_date)
    ].index

    if tracking_period_dates.empty:
        print("ERRORE (Lump Sum): Nessuna data nel periodo di tracking per l'equity line.")
        return pd.DataFrame()


    for current_date in tracking_period_dates:
        portfolio_value_today_ls = 0.0
        daily_total_dividend_received_ls = 0.0

        for ticker in tickers:
            asset_data = historical_data_map[ticker]
            asset_portfolio_ls = portfolio_ls_details[ticker]

            current_price_asset = np.nan
            dividend_asset_today = 0.0

            if current_date in asset_data.index:
                current_price_asset = asset_data.loc[current_date, 'Adj Close']
                dividend_asset_today = asset_data.loc[current_date, 'Dividend']
            else: # Se non è un giorno di trading per questo asset, prendi ultimo prezzo noto
                asset_data_before_or_on_current = asset_data[asset_data.index <= current_date]
                if not asset_data_before_or_on_current.empty:
                    current_price_asset = asset_data_before_or_on_current['Adj Close'].iloc[-1]
                # dividend_asset_today rimane 0.0

            # Reinvestimento Dividendi per Lump Sum
            if reinvest_dividends and dividend_asset_today > 0 and asset_portfolio_ls['shares_owned'] > 0 and \
               pd.notna(current_price_asset) and current_price_asset > 0:
                cash_from_dividends = asset_portfolio_ls['shares_owned'] * dividend_asset_today
                asset_portfolio_ls['dividends_cumulative_asset'] += cash_from_dividends
                daily_total_dividend_received_ls += cash_from_dividends
                
                additional_shares = cash_from_dividends / current_price_asset
                asset_portfolio_ls['shares_owned'] += additional_shares
            elif dividend_asset_today > 0 and asset_portfolio_ls['shares_owned'] > 0: # Traccia dividendi anche se non reinvestiti
                 cash_from_dividends = asset_portfolio_ls['shares_owned'] * dividend_asset_today
                 asset_portfolio_ls['dividends_cumulative_asset'] += cash_from_dividends
                 daily_total_dividend_received_ls += cash_from_dividends

            # Valore della posizione
            if pd.notna(current_price_asset):
                current_asset_ls_value = asset_portfolio_ls['shares_owned'] * current_price_asset
                asset_portfolio_ls['current_value'] = current_asset_ls_value
                portfolio_value_today_ls += current_asset_ls_value
            else:
                portfolio_value_today_ls += asset_portfolio_ls.get('current_value', 0) # Usa ultimo valore noto

        total_dividends_received_overall_ls += daily_total_dividend_received_ls

        lump_sum_evolution_records.append({
            'Date': current_date,
            'InvestedCapital': capital_invested_ls, # Il capitale investito è fisso dopo l'investimento iniziale
            'PortfolioValue': portfolio_value_today_ls,
            'DividendsReceivedCumulative': total_dividends_received_overall_ls
        })

    if not lump_sum_evolution_records:
        return pd.DataFrame()

    final_df_ls = pd.DataFrame(lump_sum_evolution_records)
    return final_df_ls.reset_index(drop=True)
