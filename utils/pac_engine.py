# simulatore_pac/utils/pac_engine.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def run_pac_simulation(
    historical_data_map: dict[str, pd.DataFrame],
    tickers: list[str],
    allocations: list[float], # Allocazioni target
    monthly_investment: float,
    start_date_pac: str,
    duration_months: int,
    reinvest_dividends: bool = True,
    rebalance_active: bool = False,
    rebalance_frequency: str = None
) -> tuple[pd.DataFrame, pd.DataFrame]: # MODIFICATO TIPO DI RITORNO

    # --- VALIDAZIONE INPUT (come prima) ---
    if not isinstance(historical_data_map, dict) or not historical_data_map:
        print("ERRORE (PAC): historical_data_map deve essere un dizionario non vuoto.")
        return pd.DataFrame(), pd.DataFrame() # MODIFICATO
    # ... (tutti gli altri controlli di input)
    if not tickers:
        print("INFO (PAC): Lista ticker vuota.")
        return pd.DataFrame(), pd.DataFrame() # MODIFICATO

    try:
        pac_start_dt = pd.to_datetime(start_date_pac)
        if pac_start_dt.tzinfo is not None:
            pac_start_dt = pac_start_dt.tz_localize(None)
    except ValueError:
        print(f"ERRORE (PAC): Formato data inizio PAC non valido: {start_date_pac}.")
        return pd.DataFrame(), pd.DataFrame() # MODIFICATO

    # --- PREPARAZIONE DATI e STATO PORTAFOGLIO ---
    portfolio_details = {
        ticker: {
            'shares_owned': 0.0, 
            'capital_invested_asset': 0.0, # Per WAP
            'dividends_cumulative_asset': 0.0, 
            'current_value': 0.0
        }
        for ticker in tickers
    }
    total_capital_invested_overall = 0.0
    total_dividends_received_overall = 0.0
    
    # NUOVO: Lista per registrare i dettagli per asset giornalieri
    asset_daily_records = [] 
    
    portfolio_total_evolution_records = [] # Per il portafoglio aggregato
    month_counter_for_investment = 0
    
    last_rebalance_date = None
    next_rebalance_date = None
    if rebalance_active and rebalance_frequency:
        # ... (logica next_rebalance_date come prima) ...
        if rebalance_frequency == "Annuale": next_rebalance_date = pac_start_dt + relativedelta(years=1)
        elif rebalance_frequency == "Semestrale": next_rebalance_date = pac_start_dt + relativedelta(months=6)
        elif rebalance_frequency == "Trimestrale": next_rebalance_date = pac_start_dt + relativedelta(months=3)


    # --- DETERMINA RANGE DATE SIMULAZIONE (come prima) ---
    reference_dates_df = historical_data_map[tickers[0]]
    actual_simulation_start_date = pac_start_dt
    last_potential_investment_date = pac_start_dt + relativedelta(months=duration_months - 1)
    max_end_date_from_ref_data = reference_dates_df.index.max()
    _end_month_target = last_potential_investment_date + relativedelta(day=31)
    actual_simulation_end_date = min(_end_month_target, max_end_date_from_ref_data)
    simulation_period_dates = reference_dates_df[
        (reference_dates_df.index >= actual_simulation_start_date) &
        (reference_dates_df.index <= actual_simulation_end_date)
    ].index
    if simulation_period_dates.empty and actual_simulation_start_date <= actual_simulation_end_date:
        simulation_period_dates = pd.date_range(start=actual_simulation_start_date, end=actual_simulation_end_date, freq='B')
        if actual_simulation_start_date == actual_simulation_end_date and actual_simulation_start_date not in simulation_period_dates:
             if actual_simulation_start_date in reference_dates_df.index:
                 simulation_period_dates = pd.DatetimeIndex([actual_simulation_start_date])

    if simulation_period_dates.empty: # Aggiunto controllo se ancora vuoto
        print("ERRORE (PAC): Nessuna data valida nel periodo di simulazione.")
        return pd.DataFrame(), pd.DataFrame()


    # --- LOOP DI SIMULAZIONE GIORNALIERO ---
    for current_date in simulation_period_dates:
        portfolio_value_today_total = 0.0
        daily_total_dividend_received = 0.0
        
        # Record per i dettagli degli asset di oggi
        current_day_asset_details_record = {'Date': current_date}

        # --- LOGICA DI INVESTIMENTO PAC MENSILE (come prima) ---
        investment_date_target_for_this_month = pac_start_dt + relativedelta(months=month_counter_for_investment)
        is_pac_investment_day_for_current_month_target = False
        if current_date >= investment_date_target_for_this_month and month_counter_for_investment < duration_months:
            potential_investment_trigger_date = simulation_period_dates[simulation_period_dates >= investment_date_target_for_this_month]
            if not potential_investment_trigger_date.empty and current_date == potential_investment_trigger_date[0]:
                is_pac_investment_day_for_current_month_target = True

        if is_pac_investment_day_for_current_month_target:
            total_capital_invested_overall += monthly_investment
            for i, ticker in enumerate(tickers):
                asset_data = historical_data_map[ticker]
                allocation = allocations[i]
                investment_for_this_asset = monthly_investment * allocation

                price_for_investment = np.nan
                if current_date in asset_data.index:
                    price_for_investment = asset_data.loc[current_date, 'Adj Close']
                else: 
                    asset_data_before_or_on_current = asset_data[asset_data.index <= current_date]
                    if not asset_data_before_or_on_current.empty:
                        price_for_investment = asset_data_before_or_on_current['Adj Close'].iloc[-1]
                
                if pd.notna(price_for_investment) and price_for_investment > 0 and allocation > 0:
                    shares_bought = investment_for_this_asset / price_for_investment
                    portfolio_details[ticker]['shares_owned'] += shares_bought
                    portfolio_details[ticker]['capital_invested_asset'] += investment_for_this_asset # TRACCIA CAPITALE PER ASSET
                # else: gestisci impossibilità di investire
            month_counter_for_investment += 1

        # --- LOGICA DI RIBILANCIAMENTO (come prima) ---
        perform_rebalance_today = False
        if rebalance_active and next_rebalance_date and current_date >= next_rebalance_date:
            perform_rebalance_today = True
        
        if perform_rebalance_today:
            # ... (logica di ribilanciamento come prima, che aggiorna portfolio_details[ticker]['shares_owned']) ...
            # È importante che capital_invested_asset NON venga modificato dal ribilanciamento
            # perché il ribilanciamento è solo una riallocazione del valore esistente.
            print(f"INFO (PAC): Ribilanciamento in data {current_date.strftime('%Y-%m-%d')}")
            current_total_portfolio_value_for_rebalance = 0
            temp_asset_values_for_rebalance = {}
            for ticker_rebal in tickers:
                asset_data_rebal = historical_data_map[ticker_rebal]
                asset_portfolio_rebal = portfolio_details[ticker_rebal]
                price_for_rebalance = np.nan
                if current_date in asset_data_rebal.index: price_for_rebalance = asset_data_rebal.loc[current_date, 'Adj Close']
                else:
                    asset_data_before = asset_data_rebal[asset_data_rebal.index <= current_date]
                    if not asset_data_before.empty: price_for_rebalance = asset_data_before['Adj Close'].iloc[-1]
                
                if pd.notna(price_for_rebalance):
                    value = asset_portfolio_rebal['shares_owned'] * price_for_rebalance
                    temp_asset_values_for_rebalance[ticker_rebal] = {'value': value, 'price': price_for_rebalance}
                    current_total_portfolio_value_for_rebalance += value
                else:
                    temp_asset_values_for_rebalance[ticker_rebal] = {'value': asset_portfolio_rebal.get('current_value',0), 'price': np.nan}
                    current_total_portfolio_value_for_rebalance += asset_portfolio_rebal.get('current_value',0)

            if current_total_portfolio_value_for_rebalance > 0:
                for i_rebal, ticker_rebal in enumerate(tickers):
                    target_allocation = allocations[i_rebal]
                    asset_portfolio_rebal = portfolio_details[ticker_rebal]
                    target_value_asset = current_total_portfolio_value_for_rebalance * target_allocation
                    current_value_asset = temp_asset_values_for_rebalance[ticker_rebal]['value']
                    price_asset = temp_asset_values_for_rebalance[ticker_rebal]['price']
                    value_difference = target_value_asset - current_value_asset
                    if pd.notna(price_asset) and price_asset > 0:
                        shares_to_transact = value_difference / price_asset
                        if shares_to_transact < 0 and abs(shares_to_transact) > asset_portfolio_rebal['shares_owned']:
                            shares_to_transact = -asset_portfolio_rebal['shares_owned']
                        asset_portfolio_rebal['shares_owned'] += shares_to_transact
            
            last_rebalance_date = current_date
            if rebalance_frequency == "Annuale": next_rebalance_date = last_rebalance_date + relativedelta(years=1)
            elif rebalance_frequency == "Semestrale": next_rebalance_date = last_rebalance_date + relativedelta(months=6)
            elif rebalance_frequency == "Trimestrale": next_rebalance_date = last_rebalance_date + relativedelta(months=3)


        # --- GESTIONE DIVIDENDI, VALORE PORTAFOGLIO GIORNALIERO E DETTAGLI ASSET ---
        portfolio_value_today_total = 0 # Ricalcola dopo eventuale ribilanciamento
        
        for ticker in tickers:
            asset_data = historical_data_map[ticker]
            asset_portfolio = portfolio_details[ticker] # Riferimento al dizionario per questo ticker
            current_price_asset = np.nan
            dividend_asset_today = 0.0

            if current_date in asset_data.index:
                current_price_asset = asset_data.loc[current_date, 'Adj Close']
                dividend_asset_today = asset_data.loc[current_date, 'Dividend']
            else: # Se non è un giorno di trading per questo asset, prendi ultimo prezzo noto
                asset_data_before_or_on_current = asset_data[asset_data.index <= current_date]
                if not asset_data_before_or_on_current.empty:
                    current_price_asset = asset_data_before_or_on_current['Adj Close'].iloc[-1]

            # Reinvestimento Dividendi (aggiorna shares_owned e dividends_cumulative_asset)
            if reinvest_dividends and dividend_asset_today > 0 and asset_portfolio['shares_owned'] > 0 and \
               pd.notna(current_price_asset) and current_price_asset > 0:
                cash_from_dividends = asset_portfolio['shares_owned'] * dividend_asset_today
                asset_portfolio['dividends_cumulative_asset'] += cash_from_dividends
                daily_total_dividend_received += cash_from_dividends # Per il totale del portafoglio
                additional_shares = cash_from_dividends / current_price_asset
                asset_portfolio['shares_owned'] += additional_shares # Aggiorna le quote dell'asset
            elif dividend_asset_today > 0 and asset_portfolio['shares_owned'] > 0: # Traccia dividendi anche se non reinvestiti
                 cash_from_dividends = asset_portfolio['shares_owned'] * dividend_asset_today
                 asset_portfolio['dividends_cumulative_asset'] += cash_from_dividends
                 daily_total_dividend_received += cash_from_dividends
            
            # Valore della posizione dell'asset e aggiornamento per il totale
            if pd.notna(current_price_asset):
                current_asset_value = asset_portfolio['shares_owned'] * current_price_asset
                asset_portfolio['current_value'] = current_asset_value # Aggiorna l'ultimo valore noto dell'asset
            else: # Se il prezzo non è disponibile, usa l'ultimo valore noto per questo asset
                 current_asset_value = asset_portfolio.get('current_value', 0.0) 
            
            portfolio_value_today_total += current_asset_value

            # NUOVO: Registra dettagli per asset
            current_day_asset_details_record[f'{ticker}_shares'] = asset_portfolio['shares_owned']
            current_day_asset_details_record[f'{ticker}_value'] = current_asset_value
            current_day_asset_details_record[f'{ticker}_capital_invested'] = asset_portfolio['capital_invested_asset']


        asset_daily_records.append(current_day_asset_details_record)
        total_dividends_received_overall += daily_total_dividend_received

        portfolio_total_evolution_records.append({
            'Date': current_date,
            'InvestedCapital': total_capital_invested_overall,
            'PortfolioValue': portfolio_value_today_total,
            'DividendsReceivedCumulative': total_dividends_received_overall
        })

    # --- FINE LOOP DI SIMULAZIONE ---

    if not portfolio_total_evolution_records:
        return pd.DataFrame(), pd.DataFrame() # MODIFICATO

    pac_total_df = pd.DataFrame(portfolio_total_evolution_records).reset_index(drop=True)
    
    # NUOVO: Crea DataFrame per i dettagli degli asset
    asset_details_history_df = pd.DataFrame(asset_daily_records)
    if not asset_details_history_df.empty:
        asset_details_history_df['Date'] = pd.to_datetime(asset_details_history_df['Date'])
        # Potrebbe essere utile impostare 'Date' come indice anche qui o lasciarla come colonna
        # Per lo stacked area chart, un indice Date è utile.
        # asset_details_history_df = asset_details_history_df.set_index('Date')


    return pac_total_df, asset_details_history_df # MODIFICATO
