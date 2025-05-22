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
    start_date_pac_str: str, # Data inizio contributi PAC (stringa YYYY-MM-DD)
    duration_months_contributions: int, # Durata del PERIODO DI CONTRIBUZIONE PAC
    reinvest_dividends: bool = True,
    rebalance_active: bool = False,
    rebalance_frequency: str = None
    # La data di fine simulazione è implicitamente la fine dei dati in historical_data_map per il reference ticker
) -> tuple[pd.DataFrame, pd.DataFrame]:

    # --- VALIDAZIONE INPUT ---
    if not (isinstance(historical_data_map, dict) and historical_data_map and tickers):
        # print("DEBUG (pac_engine): historical_data_map vuoto o tickers mancanti.")
        return pd.DataFrame(), pd.DataFrame()
    if not all(ticker in historical_data_map for ticker in tickers):
        # print("DEBUG (pac_engine): Dati storici mancanti per alcuni ticker.")
        return pd.DataFrame(), pd.DataFrame()
    if len(tickers) != len(allocations):
        # print("DEBUG (pac_engine): Discrepanza tickers/allocazioni.")
        return pd.DataFrame(), pd.DataFrame()
    if tickers and not np.isclose(sum(allocations), 1.0):
        # print(f"DEBUG (pac_engine): Somma allocazioni ({sum(allocations)}) != 1.0.")
        return pd.DataFrame(), pd.DataFrame()
    if not (isinstance(monthly_investment, (int, float)) and monthly_investment > 0):
        # print("DEBUG (pac_engine): monthly_investment non valido.")
        return pd.DataFrame(), pd.DataFrame()
    if not (isinstance(duration_months_contributions, int) and duration_months_contributions > 0):
        # print("DEBUG (pac_engine): duration_months_contributions non valida.")
        return pd.DataFrame(), pd.DataFrame()

    try:
        pac_contribution_start_dt = pd.to_datetime(start_date_pac_str)
        if pac_contribution_start_dt.tzinfo is not None:
            pac_contribution_start_dt = pac_contribution_start_dt.tz_localize(None)
    except ValueError:
        # print(f"DEBUG (pac_engine): Formato start_date_pac_str non valido: {start_date_pac_str}.")
        return pd.DataFrame(), pd.DataFrame()

    # --- PREPARAZIONE DATI e STATO PORTAFOGLIO ---
    portfolio_details = {
        ticker: {
            'shares_owned': 0.0,
            'capital_invested_asset': 0.0, # Capitale versato specificamente in questo asset
            'dividends_cumulative_asset': 0.0,
            'current_value': 0.0 # Valore di mercato corrente dell'asset
        } for ticker in tickers
    }
    total_capital_contributed_overall = 0.0 # Solo i versamenti PAC
    total_dividends_received_overall = 0.0
    
    asset_daily_records = []
    portfolio_total_evolution_records = []
    
    month_contribution_counter = 0 # Contatore per i versamenti mensili PAC
    
    # --- GESTIONE DATE RIBILANCIAMENTO ---
    last_rebalance_date = None
    next_rebalance_date = None
    if rebalance_active and rebalance_frequency and tickers: # Aggiunto controllo tickers
        # Imposta la prima data di ribilanciamento basata sull'inizio dei contributi PAC
        if rebalance_frequency == "Annuale": next_rebalance_date = pac_contribution_start_dt + relativedelta(years=1)
        elif rebalance_frequency == "Semestrale": next_rebalance_date = pac_contribution_start_dt + relativedelta(months=6)
        elif rebalance_frequency == "Trimestrale": next_rebalance_date = pac_contribution_start_dt + relativedelta(months=3)

    # --- DETERMINA RANGE DATE SIMULAZIONE ---
    # Usa il primo ticker come riferimento per il calendario di trading.
    # La simulazione si estenderà per tutte le date disponibili in questo DataFrame di riferimento.
    reference_dates_df = historical_data_map[tickers[0]]
    # La simulazione inizia alla data di inizio dei contributi PAC.
    simulation_start_dt = pac_contribution_start_dt
    # La simulazione finisce all'ultima data disponibile nel DataFrame di riferimento.
    simulation_actual_end_dt = reference_dates_df.index.max()

    # Prendi solo i giorni di trading del ticker di riferimento che cadono nel periodo di simulazione.
    simulation_period_dates = reference_dates_df[
        (reference_dates_df.index >= simulation_start_dt) &
        (reference_dates_df.index <= simulation_actual_end_dt)
    ].index
    
    if simulation_period_dates.empty:
        # print("DEBUG (pac_engine): Nessuna data valida nel periodo di simulazione calcolato.")
        return pd.DataFrame(), pd.DataFrame()

    # --- LOOP DI SIMULAZIONE GIORNALIERO ---
    for current_processing_date in simulation_period_dates:
        current_day_portfolio_value_total = 0.0
        current_day_total_dividend_received = 0.0
        current_day_asset_details_log = {'Date': current_processing_date}

        # --- 1. VERSAMENTI PAC MENSILI (solo durante il periodo di contribuzione) ---
        if month_contribution_counter < duration_months_contributions:
            target_contribution_date_this_month = pac_contribution_start_dt + relativedelta(months=month_contribution_counter)
            
            # L'investimento avviene il primo giorno di trading (current_processing_date)
            # che è >= del target_contribution_date_this_month
            if current_processing_date >= target_contribution_date_this_month:
                # Assicurati che sia il *primo* giorno di trading che soddisfa questa condizione per il mese target
                # Questo è gestito implicitamente dal fatto che incrementiamo month_contribution_counter
                # e che current_processing_date avanza.
                
                total_capital_contributed_overall += monthly_investment
                for i, ticker in enumerate(tickers):
                    asset_data = historical_data_map[ticker]
                    allocation_pct = allocations[i]
                    investment_for_this_asset = monthly_investment * allocation_pct

                    # Prezzo per l'investimento: usa il prezzo di current_processing_date se disponibile per l'asset,
                    # altrimenti non investire in quell'asset in questa data (semplificazione).
                    # Una logica più avanzata potrebbe accumulare il cash e investirlo dopo.
                    if current_processing_date in asset_data.index:
                        price_for_investment = asset_data.loc[current_processing_date, 'Adj Close']
                        if pd.notna(price_for_investment) and price_for_investment > 0 and allocation_pct > 0:
                            shares_bought = investment_for_this_asset / price_for_investment
                            portfolio_details[ticker]['shares_owned'] += shares_bought
                            portfolio_details[ticker]['capital_invested_asset'] += investment_for_this_asset
                        # else: print(f"DEBUG (pac_engine): Prezzo non valido per {ticker} il {current_processing_date}, investimento PAC saltato per questo asset.")
                    # else: print(f"DEBUG (pac_engine): {current_processing_date} non è un giorno di trading per {ticker}, investimento PAC saltato per questo asset.")
                month_contribution_counter += 1 # Passa al prossimo mese di contribuzione


        # --- 2. RIBILANCIAMENTO (continua per tutta la simulazione, se attivo) ---
        if rebalance_active and next_rebalance_date and current_processing_date >= next_rebalance_date:
            # ... (Logica di ribilanciamento COMPLETA come nella versione precedente)
            # Questa logica aggiorna portfolio_details[ticker]['shares_owned']
            # Non modifica portfolio_details[ticker]['capital_invested_asset']
            # print(f"DEBUG (pac_engine): Ribilanciamento il {current_processing_date.strftime('%Y-%m-%d')}")
            # (Incolla qui la logica di ribilanciamento testata e funzionante)
            temp_total_value_for_rebal = 0
            asset_current_prices_for_rebal = {}
            for ticker_rb in tickers: # Calcola valore attuale di ogni asset e totale
                asset_data_rb = historical_data_map[ticker_rb]
                price_rb = np.nan
                if current_processing_date in asset_data_rb.index: price_rb = asset_data_rb.loc[current_processing_date, 'Adj Close']
                else:
                    subset_rb = asset_data_rb[asset_data_rb.index <= current_processing_date]
                    if not subset_rb.empty: price_rb = subset_rb['Adj Close'].iloc[-1]
                asset_current_prices_for_rebal[ticker_rb] = price_rb
                if pd.notna(price_rb): temp_total_value_for_rebal += portfolio_details[ticker_rb]['shares_owned'] * price_rb
                else: temp_total_value_for_rebal += portfolio_details[ticker_rb]['current_value'] # Usa ultimo valore noto
            
            if temp_total_value_for_rebal > 0:
                for i_rb, ticker_rb in enumerate(tickers):
                    target_value_asset = temp_total_value_for_rebal * allocations[i_rb] # allocations sono le target
                    current_price_rb = asset_current_prices_for_rebal[ticker_rb]
                    if pd.notna(current_price_rb) and current_price_rb > 0:
                        current_value_asset = portfolio_details[ticker_rb]['shares_owned'] * current_price_rb
                        value_diff = target_value_asset - current_value_asset
                        shares_to_transact = value_diff / current_price_rb
                        # Non vendere più di quanto possiedi
                        if shares_to_transact < 0 and abs(shares_to_transact) > portfolio_details[ticker_rb]['shares_owned']:
                            shares_to_transact = -portfolio_details[ticker_rb]['shares_owned']
                        portfolio_details[ticker_rb]['shares_owned'] += shares_to_transact
            
            last_rebalance_date = current_processing_date
            if rebalance_frequency == "Annuale": next_rebalance_date = last_rebalance_date + relativedelta(years=1)
            elif rebalance_frequency == "Semestrale": next_rebalance_date = last_rebalance_date + relativedelta(months=6)
            elif rebalance_frequency == "Trimestrale": next_rebalance_date = last_rebalance_date + relativedelta(months=3)


        # --- 3. GESTIONE DIVIDENDI, VALORE PORTAFOGLIO E DETTAGLI ASSET (per tutta la simulazione) ---
        current_day_portfolio_value_total = 0.0 # Ricalcola il valore totale oggi
        current_day_total_dividend_received_for_overall_tracking = 0.0

        for ticker in tickers:
            asset_data = historical_data_map[ticker]
            asset_state = portfolio_details[ticker] # Riferimento al dizionario per questo ticker
            
            current_price_for_asset_today = np.nan
            dividend_paid_by_asset_today = 0.0

            if current_processing_date in asset_data.index:
                current_price_for_asset_today = asset_data.loc[current_processing_date, 'Adj Close']
                dividend_paid_by_asset_today = asset_data.loc[current_processing_date, 'Dividend']
            else: # Giorno non di trading per questo asset, usa l'ultimo prezzo noto per valutazione
                asset_data_up_to_today = asset_data[asset_data.index <= current_processing_date]
                if not asset_data_up_to_today.empty:
                    current_price_for_asset_today = asset_data_up_to_today['Adj Close'].iloc[-1]
            
            # Reinvestimento Dividendi
            if reinvest_dividends and dividend_paid_by_asset_today > 0 and asset_state['shares_owned'] > 0:
                cash_from_dividends_this_asset = asset_state['shares_owned'] * dividend_paid_by_asset_today
                asset_state['dividends_cumulative_asset'] += cash_from_dividends_this_asset
                current_day_total_dividend_received_for_overall_tracking += cash_from_dividends_this_asset
                
                if pd.notna(current_price_for_asset_today) and current_price_for_asset_today > 0:
                    additional_shares_from_dividends = cash_from_dividends_this_asset / current_price_for_asset_today
                    asset_state['shares_owned'] += additional_shares_from_dividends
                # else: print(f"DEBUG (pac_engine): Prezzo non valido per {ticker} il {current_processing_date}, dividendo non reinvestito.")
            elif dividend_paid_by_asset_today > 0 and asset_state['shares_owned'] > 0: # Traccia dividendi anche se non reinvestiti
                 cash_from_dividends_this_asset = asset_state['shares_owned'] * dividend_paid_by_asset_today
                 asset_state['dividends_cumulative_asset'] += cash_from_dividends_this_asset
                 current_day_total_dividend_received_for_overall_tracking += cash_from_dividends_this_asset

            # Valore attuale della posizione dell'asset
            if pd.notna(current_price_for_asset_today):
                asset_state['current_value'] = asset_state['shares_owned'] * current_price_for_asset_today
            # else: asset_state['current_value'] rimane l'ultimo valore calcolato (implicito dal non aggiornamento)
            
            current_day_portfolio_value_total += asset_state['current_value']

            # Registra dettagli per asset per questo giorno
            current_day_asset_details_log[f'{ticker}_shares'] = asset_state['shares_owned']
            current_day_asset_details_log[f'{ticker}_value'] = asset_state['current_value']
            current_day_asset_details_log[f'{ticker}_capital_invested'] = asset_state['capital_invested_asset']
        
        asset_daily_records.append(current_day_asset_details_log)
        total_dividends_received_overall += current_day_total_dividend_received_for_overall_tracking

        portfolio_total_evolution_records.append({
            'Date': current_processing_date,
            'InvestedCapital': total_capital_contributed_overall, # Capitale versato
            'PortfolioValue': current_day_portfolio_value_total,
            'DividendsReceivedCumulative': total_dividends_received_overall
        })

    # --- FINE LOOP DI SIMULAZIONE ---
    if not portfolio_total_evolution_records:
        # print("DEBUG (pac_engine): Nessun record di evoluzione del portafoglio totale generato.")
        return pd.DataFrame(), pd.DataFrame()

    pac_total_df = pd.DataFrame(portfolio_total_evolution_records)
    if 'Date' in pac_total_df.columns: pac_total_df['Date'] = pd.to_datetime(pac_total_df['Date'])
    
    asset_details_history_df = pd.DataFrame(asset_daily_records)
    if not asset_details_history_df.empty and 'Date' in asset_details_history_df.columns:
        asset_details_history_df['Date'] = pd.to_datetime(asset_details_history_df['Date'])
    
    return pac_total_df, asset_details_history_df
