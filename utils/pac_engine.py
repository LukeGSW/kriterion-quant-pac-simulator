# simulatore_pac/utils/pac_engine.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def run_pac_simulation(
    historical_data_map: dict[str, pd.DataFrame],
    tickers: list[str],
    allocations: list[float],
    monthly_investment: float,
    start_date_pac: str, # Stringa YYYY-MM-DD
    duration_months: int, # Durata del PERIODO DI CONTRIBUZIONE PAC
    reinvest_dividends: bool = True,
    rebalance_active: bool = False,
    rebalance_frequency: str = None,
    # NUOVO: Data finale effettiva per l'intera simulazione (tracking post-PAC)
    # Questa sarà determinata da main.py (es. potential_chart_extension_end_date)
    # e usata per definire fino a quando il loop deve andare.
    # Per ora, il loop andrà fino alla fine dei dati disponibili in historical_data_map.
    # Assicuriamoci che historical_data_map contenga dati fino alla fine desiderata.
) -> tuple[pd.DataFrame, pd.DataFrame]:

    # --- VALIDAZIONE INPUT (come prima) ---
    if not isinstance(historical_data_map, dict) or not historical_data_map:
        return pd.DataFrame(), pd.DataFrame()
    if not tickers: return pd.DataFrame(), pd.DataFrame()
    # ... (altri controlli di input) ...
    try:
        pac_start_dt = pd.to_datetime(start_date_pac)
        if pac_start_dt.tzinfo is not None: pac_start_dt = pac_start_dt.tz_localize(None)
    except ValueError: return pd.DataFrame(), pd.DataFrame()

    # --- PREPARAZIONE DATI e STATO PORTAFOGLIO ---
    portfolio_details = {
        ticker: {'shares_owned': 0.0, 'capital_invested_asset': 0.0, 
                   'dividends_cumulative_asset': 0.0, 'current_value': 0.0}
        for ticker in tickers
    }
    total_capital_invested_overall = 0.0 # Questo si ferma dopo duration_months
    total_dividends_received_overall = 0.0
    asset_daily_records = [] 
    portfolio_total_evolution_records = []
    month_counter_for_investment = 0
    
    last_rebalance_date = None
    next_rebalance_date = None
    if rebalance_active and rebalance_frequency:
        if rebalance_frequency == "Annuale": next_rebalance_date = pac_start_dt + relativedelta(years=1)
        elif rebalance_frequency == "Semestrale": next_rebalance_date = pac_start_dt + relativedelta(months=6)
        elif rebalance_frequency == "Trimestrale": next_rebalance_date = pac_start_dt + relativedelta(months=3)

    # --- DETERMINA RANGE DATE SIMULAZIONE ---
    # Il loop ora va fino alla fine dei dati disponibili nel reference_dates_df,
    # che main.py dovrebbe aver caricato fino a potential_chart_extension_end_date.
    reference_dates_df = historical_data_map[tickers[0]]
    
    # La simulazione inizia alla data di inizio del PAC.
    # L'actual_simulation_end_date è l'ultima data disponibile nel reference ticker.
    # main.py si occuperà di passare dati fino alla data desiderata.
    simulation_start_date_dt = pac_start_dt
    simulation_end_date_dt = reference_dates_df.index.max() # Fine dei dati disponibili

    simulation_period_dates = reference_dates_df[
        (reference_dates_df.index >= simulation_start_date_dt) &
        (reference_dates_df.index <= simulation_end_date_dt)
    ].index
    
    if simulation_period_dates.empty:
        print("ERRORE (PAC): Nessuna data valida nel periodo di simulazione.")
        return pd.DataFrame(), pd.DataFrame()

    # --- LOOP DI SIMULAZIONE GIORNALIERO ---
    for current_date in simulation_period_dates:
        portfolio_value_today_total = 0.0
        daily_total_dividend_received = 0.0
        current_day_asset_details_record = {'Date': current_date}

        # --- LOGICA DI INVESTIMENTO PAC MENSILE ---
        # Avviene solo se siamo ancora nel periodo di contribuzione del PAC (duration_months)
        if month_counter_for_investment < duration_months:
            investment_date_target_for_this_month = pac_start_dt + relativedelta(months=month_counter_for_investment)
            is_pac_investment_day_for_current_month_target = False
            if current_date >= investment_date_target_for_this_month:
                # Verifica se è il primo giorno di trading valido per questo versamento
                # (considerando che simulation_period_dates sono già giorni di trading del reference)
                potential_investment_trigger_date = simulation_period_dates[simulation_period_dates >= investment_date_target_for_this_month]
                if not potential_investment_trigger_date.empty and current_date == potential_investment_trigger_date[0]:
                    is_pac_investment_day_for_current_month_target = True

            if is_pac_investment_day_for_current_month_target:
                total_capital_invested_overall += monthly_investment # Incrementa solo durante i versamenti
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
                        portfolio_details[ticker]['capital_invested_asset'] += investment_for_this_asset
                month_counter_for_investment += 1

        # --- LOGICA DI RIBILANCIAMENTO (continua anche dopo la fine dei versamenti PAC) ---
        perform_rebalance_today = False
        if rebalance_active and next_rebalance_date and current_date >= next_rebalance_date:
            perform_rebalance_today = True
        
        if perform_rebalance_today:
            # ... (logica di ribilanciamento come prima, INVARIATA) ...
            # (calcola current_total_portfolio_value_for_rebalance, aggiorna shares_owned per ogni ticker)
            # (aggiorna next_rebalance_date)
            print(f"INFO (PAC): Ribilanciamento in data {current_date.strftime('%Y-%m-%d')}")
            current_total_portfolio_value_for_rebalance = 0; temp_asset_values_for_rebalance = {}
            for ticker_rebal in tickers:
                asset_data_rebal = historical_data_map[ticker_rebal]; asset_portfolio_rebal = portfolio_details[ticker_rebal]; price_for_rebalance = np.nan
                if current_date in asset_data_rebal.index: price_for_rebalance = asset_data_rebal.loc[current_date, 'Adj Close']
                else:
                    asset_data_before = asset_data_rebal[asset_data_rebal.index <= current_date]
                    if not asset_data_before.empty: price_for_rebalance = asset_data_before['Adj Close'].iloc[-1]
                if pd.notna(price_for_rebalance): value = asset_portfolio_rebal['shares_owned'] * price_for_rebalance; temp_asset_values_for_rebalance[ticker_rebal] = {'value': value, 'price': price_for_rebalance}; current_total_portfolio_value_for_rebalance += value
                else: temp_asset_values_for_rebalance[ticker_rebal] = {'value': asset_portfolio_rebal.get('current_value',0), 'price': np.nan}; current_total_portfolio_value_for_rebalance += asset_portfolio_rebal.get('current_value',0)
            if current_total_portfolio_value_for_rebalance > 0:
                for i_rebal, ticker_rebal in enumerate(tickers):
                    target_allocation = allocations[i_rebal]; asset_portfolio_rebal = portfolio_details[ticker_rebal]
                    target_value_asset = current_total_portfolio_value_for_rebalance * target_allocation
                    current_value_asset = temp_asset_values_for_rebalance[ticker_rebal]['value']; price_asset = temp_asset_values_for_rebalance[ticker_rebal]['price']
                    value_difference = target_value_asset - current_value_asset
                    if pd.notna(price_asset) and price_asset > 0:
                        shares_to_transact = value_difference / price_asset
                        if shares_to_transact < 0 and abs(shares_to_transact) > asset_portfolio_rebal['shares_owned']: shares_to_transact = -asset_portfolio_rebal['shares_owned']
                        asset_portfolio_rebal['shares_owned'] += shares_to_transact
            last_rebalance_date = current_date
            if rebalance_frequency == "Annuale": next_rebalance_date = last_rebalance_date + relativedelta(years=1)
            elif rebalance_frequency == "Semestrale": next_rebalance_date = last_rebalance_date + relativedelta(months=6)
            elif rebalance_frequency == "Trimestrale": next_rebalance_date = last_rebalance_date + relativedelta(months=3)

        # --- GESTIONE DIVIDENDI E VALORE PORTAFOGLIO GIORNALIERO (continua anche dopo la fine dei versamenti PAC) ---
        portfolio_value_today_total = 0
        for ticker in tickers:
            asset_data = historical_data_map[ticker]
            asset_portfolio = portfolio_details[ticker]
            current_price_asset = np.nan; dividend_asset_today = 0.0
            if current_date in asset_data.index:
                current_price_asset = asset_data.loc[current_date, 'Adj Close']
                dividend_asset_today = asset_data.loc[current_date, 'Dividend']
            else:
                asset_data_before_or_on_current = asset_data[asset_data.index <= current_date]
                if not asset_data_before_or_on_current.empty:
                    current_price_asset = asset_data_before_or_on_current['Adj Close'].iloc[-1]
            
            if reinvest_dividends and dividend_asset_today > 0 and asset_portfolio['shares_owned'] > 0 and pd.notna(current_price_asset) and current_price_asset > 0:
                cash_from_dividends = asset_portfolio['shares_owned'] * dividend_asset_today
                asset_portfolio['dividends_cumulative_asset'] += cash_from_dividends
                daily_total_dividend_received += cash_from_dividends
                additional_shares = cash_from_dividends / current_price_asset
                asset_portfolio['shares_owned'] += additional_shares
            elif dividend_asset_today > 0 and asset_portfolio['shares_owned'] > 0:
                 cash_from_dividends = asset_portfolio['shares_owned'] * dividend_asset_today
                 asset_portfolio['dividends_cumulative_asset'] += cash_from_dividends
                 daily_total_dividend_received += cash_from_dividends
            
            if pd.notna(current_price_asset):
                current_asset_value = asset_portfolio['shares_owned'] * current_price_asset
                asset_portfolio['current_value'] = current_asset_value
            else: current_asset_value = asset_portfolio.get('current_value', 0.0) 
            portfolio_value_today_total += current_asset_value

            current_day_asset_details_record[f'{ticker}_shares'] = asset_portfolio['shares_owned']
            current_day_asset_details_record[f'{ticker}_value'] = current_asset_value
            current_day_asset_details_record[f'{ticker}_capital_invested'] = asset_portfolio['capital_invested_asset']
        
        asset_daily_records.append(current_day_asset_details_record)
        total_dividends_received_overall += daily_total_dividend_received

        portfolio_total_evolution_records.append({
            'Date': current_date,
            'InvestedCapital': total_capital_invested_overall, # Questo non cresce più dopo la fine del PAC
            'PortfolioValue': portfolio_value_today_total,
            'DividendsReceivedCumulative': total_dividends_received_overall
        })

    # --- FINE LOOP DI SIMULAZIONE ---
    if not portfolio_total_evolution_records: return pd.DataFrame(), pd.DataFrame()
    pac_total_df = pd.DataFrame(portfolio_total_evolution_records).reset_index(drop=True)
    asset_details_history_df = pd.DataFrame(asset_daily_records)
    if not asset_details_history_df.empty:
        asset_details_history_df['Date'] = pd.to_datetime(asset_details_history_df['Date'])
    return pac_total_df, asset_details_history_df
