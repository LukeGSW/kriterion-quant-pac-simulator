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
    rebalance_active: bool = False, # Nuovo parametro
    rebalance_frequency: str = None # Es. "Annuale", "Semestrale", "Trimestrale"
) -> pd.DataFrame:

    # --- VALIDAZIONE INPUT ---
    # ... (come prima, aggiungi controlli per i nuovi parametri se necessario) ...
    if not isinstance(historical_data_map, dict) or not historical_data_map:
        print("ERRORE (PAC): historical_data_map deve essere un dizionario non vuoto.")
        return pd.DataFrame()
    # ... (altri controlli di input esistenti)

    try:
        pac_start_dt = pd.to_datetime(start_date_pac)
        if pac_start_dt.tzinfo is not None:
            pac_start_dt = pac_start_dt.tz_localize(None)
    except ValueError:
        print(f"ERRORE (PAC): Formato data inizio PAC non valido: {start_date_pac}.")
        return pd.DataFrame()

    # --- PREPARAZIONE DATI e STATO PORTAFOGLIO ---
    portfolio_details = {
        ticker: {'shares_owned': 0.0, 'capital_invested_asset': 0.0, 
                   'dividends_cumulative_asset': 0.0, 'current_value': 0.0}
        for ticker in tickers
    }
    total_capital_invested_overall = 0.0
    total_dividends_received_overall = 0.0
    portfolio_evolution_records = []
    month_counter_for_investment = 0
    
    # --- GESTIONE DATE RIBILANCIAMENTO ---
    last_rebalance_date = None # Non abbiamo ancora ribilanciato
    next_rebalance_date = None

    if rebalance_active and rebalance_frequency:
        # Imposta la prima data di ribilanciamento
        if rebalance_frequency == "Annuale":
            next_rebalance_date = pac_start_dt + relativedelta(years=1)
        elif rebalance_frequency == "Semestrale":
            next_rebalance_date = pac_start_dt + relativedelta(months=6)
        elif rebalance_frequency == "Trimestrale":
            next_rebalance_date = pac_start_dt + relativedelta(months=3)
        # else: gestisci errore o frequenza non valida

    # --- DETERMINA RANGE DATE SIMULAZIONE ---
    if not tickers:
        print("INFO (PAC): Lista ticker vuota.")
        return pd.DataFrame()
    reference_dates_df = historical_data_map[tickers[0]] # Usiamo il primo ticker per il calendario principale
    actual_simulation_start_date = pac_start_dt
    last_potential_investment_date = pac_start_dt + relativedelta(months=duration_months - 1)
    max_end_date_from_ref_data = reference_dates_df.index.max()
    _end_month_target = last_potential_investment_date + relativedelta(day=31)
    actual_simulation_end_date = min(_end_month_target, max_end_date_from_ref_data)
    simulation_period_dates = reference_dates_df[
        (reference_dates_df.index >= actual_simulation_start_date) &
        (reference_dates_df.index <= actual_simulation_end_date)
    ].index
    if simulation_period_dates.empty and actual_simulation_start_date <= actual_simulation_end_date: # Fallback
        simulation_period_dates = pd.date_range(start=actual_simulation_start_date, end=actual_simulation_end_date, freq='B')
        if actual_simulation_start_date == actual_simulation_end_date and actual_simulation_start_date not in simulation_period_dates:
             if actual_simulation_start_date in reference_dates_df.index:
                 simulation_period_dates = pd.DatetimeIndex([actual_simulation_start_date])

    # --- LOOP DI SIMULAZIONE GIORNALIERO ---
    for current_date in simulation_period_dates:
        portfolio_value_today_total = 0.0
        daily_total_dividend_received = 0.0
        
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
                # ... (logica di acquisto mensile per asset come prima) ...
                asset_data = historical_data_map[ticker]
                allocation = allocations[i] # Allocazione target
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
                # else: gestisci impossibilità di investire
            month_counter_for_investment += 1


        # --- LOGICA DI RIBILANCIAMENTO ---
        perform_rebalance_today = False
        if rebalance_active and next_rebalance_date and current_date >= next_rebalance_date:
            # Assicurati che sia un giorno di trading valido per la maggior parte degli asset o per il reference
            # Per semplicità, se current_date è nel nostro simulation_period_dates (basato sul reference ticker), procediamo.
            perform_rebalance_today = True
        
        if perform_rebalance_today:
            print(f"INFO (PAC): Ribilanciamento in data {current_date.strftime('%Y-%m-%d')}")
            # 1. Calcola il valore attuale di ogni posizione e il valore totale del portafoglio
            current_total_portfolio_value_for_rebalance = 0
            temp_asset_values_for_rebalance = {}

            for ticker in tickers:
                asset_data = historical_data_map[ticker]
                asset_portfolio = portfolio_details[ticker]
                price_for_rebalance = np.nan
                if current_date in asset_data.index:
                    price_for_rebalance = asset_data.loc[current_date, 'Adj Close']
                else:
                    asset_data_before_or_on_current = asset_data[asset_data.index <= current_date]
                    if not asset_data_before_or_on_current.empty:
                        price_for_rebalance = asset_data_before_or_on_current['Adj Close'].iloc[-1]
                
                if pd.notna(price_for_rebalance):
                    value = asset_portfolio['shares_owned'] * price_for_rebalance
                    temp_asset_values_for_rebalance[ticker] = {'value': value, 'price': price_for_rebalance}
                    current_total_portfolio_value_for_rebalance += value
                else: # Se un asset non ha prezzo, non può essere ribilanciato
                    temp_asset_values_for_rebalance[ticker] = {'value': asset_portfolio.get('current_value',0), 'price': np.nan} # Usa ultimo valore noto
                    current_total_portfolio_value_for_rebalance += asset_portfolio.get('current_value',0)


            # 2. Determina le transazioni di ribilanciamento
            if current_total_portfolio_value_for_rebalance > 0: # Solo se il portafoglio ha valore
                for i, ticker in enumerate(tickers):
                    target_allocation = allocations[i] # Allocazione target originale
                    asset_portfolio = portfolio_details[ticker]
                    
                    target_value_asset = current_total_portfolio_value_for_rebalance * target_allocation
                    current_value_asset = temp_asset_values_for_rebalance[ticker]['value']
                    price_asset = temp_asset_values_for_rebalance[ticker]['price']
                    
                    value_difference = target_value_asset - current_value_asset
                    
                    if pd.notna(price_asset) and price_asset > 0:
                        shares_to_transact = value_difference / price_asset
                        # Se shares_to_transact > 0, dobbiamo comprare. Se < 0, dobbiamo vendere.
                        # Assicurati di non vendere più quote di quelle possedute (anche se con float può essere ok)
                        if shares_to_transact < 0 and abs(shares_to_transact) > asset_portfolio['shares_owned']:
                            shares_to_transact = -asset_portfolio['shares_owned'] # Vendi tutto
                        
                        asset_portfolio['shares_owned'] += shares_to_transact
                        # Nota: non stiamo tracciando cash da ribilanciamento o commissioni.
                        # Il capitale investito per asset non cambia con il ribilanciamento, solo le quote.
                    else:
                        print(f"ATTENZIONE (PAC): Impossibile ribilanciare {ticker} il {current_date.strftime('%Y-%m-%d')} per prezzo non valido.")

            # Aggiorna la prossima data di ribilanciamento
            last_rebalance_date = current_date # O next_rebalance_date
            if rebalance_frequency == "Annuale":
                next_rebalance_date = last_rebalance_date + relativedelta(years=1)
            elif rebalance_frequency == "Semestrale":
                next_rebalance_date = last_rebalance_date + relativedelta(months=6)
            elif rebalance_frequency == "Trimestrale":
                next_rebalance_date = last_rebalance_date + relativedelta(months=3)

        # --- GESTIONE DIVIDENDI E VALORE PORTAFOGLIO GIORNALIERO (come prima, ma dopo il ribilanciamento) ---
        portfolio_value_today_total = 0 # Ricalcola dopo eventuale ribilanciamento
        for ticker in tickers:
            asset_data = historical_data_map[ticker]
            asset_portfolio = portfolio_details[ticker]
            current_price_asset = np.nan
            dividend_asset_today = 0.0

            if current_date in asset_data.index:
                current_price_asset = asset_data.loc[current_date, 'Adj Close']
                dividend_asset_today = asset_data.loc[current_date, 'Dividend']
            else:
                asset_data_before_or_on_current = asset_data[asset_data.index <= current_date]
                if not asset_data_before_or_on_current.empty:
                    current_price_asset = asset_data_before_or_on_current['Adj Close'].iloc[-1]

            # Reinvestimento Dividendi
            if reinvest_dividends and dividend_asset_today > 0 and asset_portfolio['shares_owned'] > 0 and \
               pd.notna(current_price_asset) and current_price_asset > 0:
                cash_from_dividends = asset_portfolio['shares_owned'] * dividend_asset_today
                asset_portfolio['dividends_cumulative_asset'] += cash_from_dividends
                daily_total_dividend_received += cash_from_dividends
                additional_shares = cash_from_dividends / current_price_asset
                asset_portfolio['shares_owned'] += additional_shares
            elif dividend_asset_today > 0 and asset_portfolio['shares_owned'] > 0:
                 cash_from_dividends = asset_portfolio['shares_owned'] * dividend_asset_today
                 asset_portfolio['dividends_cumulative_asset'] += cash_from_dividends
                 daily_total_dividend_received += cash_from_dividends
            
            # Valore della posizione dell'asset
            if pd.notna(current_price_asset):
                current_asset_value = asset_portfolio['shares_owned'] * current_price_asset
                asset_portfolio['current_value'] = current_asset_value
                portfolio_value_today_total += current_asset_value
            else:
                 portfolio_value_today_total += asset_portfolio.get('current_value', 0)
        
        total_dividends_received_overall += daily_total_dividend_received

        portfolio_evolution_records.append({
            'Date': current_date,
            'InvestedCapital': total_capital_invested_overall,
            'PortfolioValue': portfolio_value_today_total,
            'DividendsReceivedCumulative': total_dividends_received_overall
        })

    # --- FINE LOOP DI SIMULAZIONE ---

    if not portfolio_evolution_records:
        return pd.DataFrame()

    final_df = pd.DataFrame(portfolio_evolution_records)
    return final_df.reset_index(drop=True)
