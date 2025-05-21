# simulatore_pac/utils/pac_engine.py
import pandas as pd
import numpy as np # Aggiunto per np.isclose e np.nan
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def run_pac_simulation(
    historical_data_map: dict[str, pd.DataFrame],
    tickers: list[str],
    allocations: list[float],
    monthly_investment: float,
    start_date_pac: str,
    duration_months: int,
    reinvest_dividends: bool = True
) -> pd.DataFrame:

    # --- VALIDAZIONE INPUT ---
    if not isinstance(historical_data_map, dict) or not historical_data_map:
        print("ERRORE (PAC): historical_data_map deve essere un dizionario non vuoto.")
        return pd.DataFrame()
    if not all(ticker in historical_data_map for ticker in tickers):
        print("ERRORE (PAC): Non tutti i ticker hanno dati storici.")
        return pd.DataFrame()
    if len(tickers) != len(allocations):
        print("ERRORE (PAC): Numero di ticker e allocazioni non corrispondono.")
        return pd.DataFrame()
    if tickers and not np.isclose(sum(allocations), 1.0):
        print(f"ERRORE (PAC): La somma delle allocazioni ({sum(allocations)}) deve essere 1.0.")
        return pd.DataFrame()
    if not isinstance(monthly_investment, (int, float)) or monthly_investment <= 0:
        print("ERRORE (PAC): monthly_investment deve essere un numero positivo.")
        return pd.DataFrame()
    if not isinstance(duration_months, int) or duration_months <= 0:
        print("ERRORE (PAC): duration_months deve essere un intero positivo.")
        return pd.DataFrame()

    try:
        pac_start_dt = pd.to_datetime(start_date_pac)
        if pac_start_dt.tzinfo is not None:
            pac_start_dt = pac_start_dt.tz_localize(None)
    except ValueError:
        print(f"ERRORE (PAC): Formato data inizio PAC non valido: {start_date_pac}.")
        return pd.DataFrame()

    # --- PREPARAZIONE DATI ---
    portfolio_details = {
        ticker: {'shares_owned': 0.0, 'capital_invested_asset': 0.0, 'dividends_cumulative_asset': 0.0, 'current_value': 0.0}
        for ticker in tickers
    }
    total_capital_invested_overall = 0.0
    total_dividends_received_overall = 0.0
    portfolio_evolution_records = []
    month_counter_for_investment = 0

    if not tickers: # Se la lista dei ticker è vuota, non c'è nulla da simulare
        print("INFO (PAC): Lista ticker vuota, nessuna simulazione da eseguire.")
        return pd.DataFrame()

    # Determina il range di date per la simulazione
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
             # Handle case where start and end date are the same and not a business day
             if actual_simulation_start_date in reference_dates_df.index: # Check if it's in original reference
                 simulation_period_dates = pd.DatetimeIndex([actual_simulation_start_date])
             else: # If not, and it's a single day, it's tricky. For now, let it be potentially empty.
                 pass


    for current_date in simulation_period_dates:
        portfolio_value_today_total = 0.0
        daily_total_dividend_received = 0.0
        
        # Flag per assicurarsi che l'investimento per un dato "mese target" avvenga solo una volta
        # Questo deve essere gestito con più attenzione. L'approccio migliore è verificare se
        # il `month_counter_for_investment` è già stato processato per la `current_date`.
        # Semplificazione: l'investimento per il mese target avviene il primo giorno di trading valido
        # a partire dalla data target del mese.
        
        investment_date_target_for_this_month = pac_start_dt + relativedelta(months=month_counter_for_investment)
        
        # Determina se è un giorno di investimento per il PAC
        # Deve essere il primo giorno di trading in simulation_period_dates che è >= investment_date_target_for_this_month
        # e per il quale non abbiamo ancora investito (month_counter_for_investment < duration_months)
        is_pac_investment_day_for_current_month_target = False
        if current_date >= investment_date_target_for_this_month and month_counter_for_investment < duration_months:
            # Per evitare investimenti multipli, verifichiamo se questo è il *primo* giorno utile per questo mese target
            # Questa logica può essere complessa. Un modo è marcare il mese target come "investito"
            # dopo il primo investimento utile per quel mese target.
            # Alternativa più semplice: se la data corrente è la prima data in simulation_period_dates
            # che soddisfa la condizione >= investment_date_target_for_this_month, allora è un giorno di investimento.
            
            # Per ora, assumiamo che se current_date >= target e non abbiamo raggiunto la durata, investiamo,
            # e l'incremento di month_counter_for_investment gestirà il passaggio al mese successivo.
            # Questo significa che l'investimento per un mese target avverrà sul primo giorno di trading
            # del nostro `simulation_period_dates` che è uguale o successivo a `investment_date_target_for_this_month`.
            
            # Troviamo il PRIMO giorno di trading nel nostro calendario che è >= target
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
                if current_date in asset_data.index: # Il nostro current_date è un giorno di trading per il reference ticker
                    price_for_investment = asset_data.loc[current_date, 'Adj Close']
                else: # Se current_date non è un giorno di trading per QUESTO asset, cerca l'ultimo prezzo
                    asset_data_before_or_on_current = asset_data[asset_data.index <= current_date]
                    if not asset_data_before_or_on_current.empty:
                        price_for_investment = asset_data_before_or_on_current['Adj Close'].iloc[-1]
                
                if pd.notna(price_for_investment) and price_for_investment > 0 and allocation > 0:
                    shares_bought = investment_for_this_asset / price_for_investment
                    portfolio_details[ticker]['shares_owned'] += shares_bought
                    portfolio_details[ticker]['capital_invested_asset'] += investment_for_this_asset
                else:
                    print(f"ATTENZIONE (PAC): Impossibile investire in {ticker} il {current_date.strftime('%Y-%m-%d')}. Prezzo: {price_for_investment if pd.notna(price_for_investment) else 'NaN'}")
            
            month_counter_for_investment += 1 # Incrementa dopo aver processato l'investimento per questo mese target


        # Gestione Dividendi e Valore Portafoglio per ogni asset
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
                # dividend_asset_today rimane 0.0 se non è un giorno di trading per questo asset

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

            if pd.notna(current_price_asset):
                current_asset_value = asset_portfolio['shares_owned'] * current_price_asset
                asset_portfolio['current_value'] = current_asset_value # Aggiorna ultimo valore noto
                portfolio_value_today_total += current_asset_value
            else: # Se il prezzo non è disponibile, usa l'ultimo valore noto per questo asset
                 portfolio_value_today_total += asset_portfolio.get('current_value', 0)


        total_dividends_received_overall += daily_total_dividend_received

        portfolio_evolution_records.append({
            'Date': current_date,
            'InvestedCapital': total_capital_invested_overall,
            'PortfolioValue': portfolio_value_today_total,
            'DividendsReceivedCumulative': total_dividends_received_overall
        })

    if not portfolio_evolution_records:
        return pd.DataFrame()

    final_df = pd.DataFrame(portfolio_evolution_records)
    return final_df.reset_index(drop=True)
