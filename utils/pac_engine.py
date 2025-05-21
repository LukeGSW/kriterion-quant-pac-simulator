# simulatore_pac/utils/pac_engine.py
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def run_pac_simulation(
    historical_data_map: dict[str, pd.DataFrame], # Dizionario: ticker -> DataFrame dei suoi dati
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
        print("ERRORE (PAC): Non tutti i ticker hanno dati storici corrispondenti in historical_data_map.")
        return pd.DataFrame()
    if len(tickers) != len(allocations):
        print("ERRORE (PAC): Il numero di ticker deve corrispondere al numero di allocazioni.")
        return pd.DataFrame()
    if not np.isclose(sum(allocations), 1.0): # Usare numpy.isclose per confronti float
        print("ERRORE (PAC): La somma delle allocazioni deve essere 1.0 (o 100%).")
        return pd.DataFrame()
    # ... (altri controlli di input esistenti per monthly_investment, duration_months, start_date_pac) ...

    try:
        pac_start_dt = pd.to_datetime(start_date_pac)
        if pac_start_dt.tzinfo is not None:
            pac_start_dt = pac_start_dt.tz_localize(None)
    except ValueError:
        # ... (gestione errore data) ...
        return pd.DataFrame()

    # --- PREPARAZIONE DATI ---
    # Trova il range di date comune a tutti i DataFrame dei prezzi
    # e assicurati che tutti gli indici siano timezone-naive.
    # Questa parte è cruciale e può essere complessa. Per ora assumiamo che i dati siano già
    # stati pre-processati per avere un indice comune o che li allineeremo qui.
    # Un modo semplice (ma che potrebbe perdere dati se gli orari di trading sono molto diversi)
    # è fare un inner join degli indici, o trovare il più piccolo range comune.
    
    # Creiamo un DataFrame "master" per le date di trading unendo gli indici di tutti i ticker
    # e prendendo solo le date in cui TUTTI i mercati erano aperti.
    # In alternativa, potremmo usare un approccio per cui ogni asset opera sui suoi giorni di trading.
    # Per ora, semplifichiamo e assumiamo che gli investimenti avvengano se il mercato USA è aperto (se SPY è un ticker di riferimento)
    # o se il primo ticker della lista è negoziabile. Una gestione più robusta è necessaria qui.

    # Per ora, lavoriamo con l'idea che il loop principale itera su un calendario di date
    # e poi, per ogni data, controlliamo quali asset sono negoziabili.

    # Inizializzazione stato portafoglio per ogni asset
    portfolio_details = {
        ticker: {'shares_owned': 0.0, 'capital_invested_asset': 0.0, 'dividends_cumulative_asset': 0.0}
        for ticker in tickers
    }
    
    total_capital_invested_overall = 0.0
    total_dividends_received_overall = 0.0 # Tracciamo i dividendi totali per il portafoglio
    
    portfolio_evolution_records = []

    # --- LOGICA DI SIMULAZIONE GIORNALIERA ---
    # (Simile a prima, ma ora deve ciclare e applicare logica per ogni asset)

    month_counter_for_investment = 0
    
    # Determina il range di date per la simulazione
    actual_simulation_start_date = pac_start_dt
    last_potential_investment_date = pac_start_dt + relativedelta(months=duration_months - 1)
    # Troviamo l'ultima data disponibile tra tutti gli asset per definire la fine della simulazione
    max_end_date_from_data = max(historical_data_map[ticker].index.max() for ticker in tickers)
    
    _end_month_target = last_potential_investment_date + relativedelta(day=31)
    actual_simulation_end_date = min(_end_month_target, max_end_date_from_data)
    
    # Genera un range di date di calendario per il loop principale
    # (poi filtreremo per i giorni di trading effettivi di ciascun asset)
    all_simulation_dates = pd.date_range(start=actual_simulation_start_date, end=actual_simulation_end_date, freq='B') # 'B' per business days
    if all_simulation_dates.empty and actual_simulation_start_date == actual_simulation_end_date: # Caso di durata molto breve
        all_simulation_dates = pd.DatetimeIndex([actual_simulation_start_date])


    for current_simulation_date_nominal in all_simulation_dates:
        portfolio_value_today = 0.0
        new_overall_dividend_today = 0.0

        # 1. Gestione Versamento PAC Mensile
        investment_date_target_for_this_month = pac_start_dt + relativedelta(months=month_counter_for_investment)
        
        # Il versamento avviene se è il primo giorno di business del mese target del PAC
        # o il primo giorno di business dopo, e non abbiamo superato la durata
        is_investment_day = False
        if current_simulation_date_nominal >= investment_date_target_for_this_month and \
           month_counter_for_investment < duration_months:
            # Verifichiamo se non abbiamo già investito per questo 'month_counter_for_investment'
            # Questo si fa controllando se current_simulation_date_nominal è "vicino" a investment_date_target_for_this_month
            # e se abbiamo incrementato month_counter_for_investment.
            # Una logica più precisa: l'investimento per month_counter_for_investment avviene
            # il primo giorno di trading >= investment_date_target_for_this_month.
            
            # Per evitare investimenti multipli per lo stesso "target mensile",
            # l'incremento di month_counter_for_investment è cruciale.
            is_investment_day = True # Marcatore per questo giorno
            total_capital_invested_overall += monthly_investment 
            # (il capitale è allocato globalmente, poi diviso per asset)
            # print(f"Debug: {current_simulation_date_nominal.date()} - GIORNO DI INVESTIMENTO PAC per mese target {month_counter_for_investment+1}")


        for i, ticker in enumerate(tickers):
            asset_data = historical_data_map[ticker]
            allocation = allocations[i]
            
            # Trova il giorno di trading effettivo per questo asset <= current_simulation_date_nominal
            # o il più vicino possibile se current_simulation_date_nominal non è un giorno di trading per QUESTO asset
            # Questo è importante se i mercati hanno giorni di chiusura diversi.
            
            # Semplificazione: usiamo current_simulation_date_nominal se è nell'indice dell'asset,
            # altrimenti cerchiamo l'ultimo prezzo valido (ffill).
            if current_simulation_date_nominal in asset_data.index:
                actual_trading_date_for_asset = current_simulation_date_nominal
                current_price = asset_data.loc[actual_trading_date_for_asset, 'Adj Close']
                dividend_asset_today = asset_data.loc[actual_trading_date_for_asset, 'Dividend']
            else:
                # Se non è un giorno di trading per l'asset, prendiamo l'ultimo prezzo e dividendo noto (forward fill)
                # per calcolare il valore del portafoglio, ma non per transazioni.
                # Le transazioni (acquisti/dividendi) dovrebbero avvenire solo nei giorni di trading effettivi.
                # Questa parte della logica necessita di molta attenzione.
                # Per ora, se non è un giorno di trading, il prezzo è NaN per le transazioni
                # ma per la valutazione usiamo l'ultimo noto.
                
                # Se current_simulation_date_nominal NON è un giorno di trading per l'asset,
                # non ci possono essere acquisti o dividendi pagati DA QUESTO ASSET in questa data nominale.
                # Il valore sarà basato sull'ultimo prezzo noto.
                
                # Selezioniamo i dati fino alla data nominale corrente
                asset_data_subset_to_date = asset_data[asset_data.index <= current_simulation_date_nominal]
                if not asset_data_subset_to_date.empty:
                    last_known_data_asset = asset_data_subset_to_date.iloc[-1]
                    current_price = last_known_data_asset['Adj Close'] # Per valutazione
                    dividend_asset_today = 0 # Non ci sono dividendi se il mercato è chiuso
                    actual_trading_date_for_asset = last_known_data_asset.name # Data dell'ultimo prezzo noto
                else: # Nessun dato storico prima o a questa data per l'asset
                    current_price = np.nan
                    dividend_asset_today = 0
                    actual_trading_date_for_asset = pd.NaT


            # A. Investimento periodico per questo asset
            if is_investment_day and allocation > 0:
                investment_for_this_asset = monthly_investment * allocation
                if pd.notna(current_price) and current_price > 0 and current_simulation_date_nominal in asset_data.index: # Assicurati che sia un giorno di trading
                    shares_bought = investment_for_this_asset / current_price
                    portfolio_details[ticker]['shares_owned'] += shares_bought
                    portfolio_details[ticker]['capital_invested_asset'] += investment_for_this_asset
                    # print(f"Debug: ...Investito in {ticker}: {investment_for_this_asset:.2f}, Quote: {shares_bought:.4f}")
                else:
                    # Cosa fare se non si può investire? Accumulare cash per l'asset?
                    # Per ora, l'investimento "globale" è stato contato, ma non allocato a questo asset in quote.
                    print(f"ATTENZIONE (PAC): Prezzo non valido o giorno non di trading per {ticker} il {current_simulation_date_nominal.strftime('%Y-%m-%d')} per investimento. Allocazione non convertita in quote.")


            # B. Reinvestimento dividendi per questo asset
            if reinvest_dividends and dividend_asset_today > 0 and portfolio_details[ticker]['shares_owned'] > 0 and current_simulation_date_nominal in asset_data.index: # Assicurati che sia un giorno di trading
                cash_from_dividends_asset = portfolio_details[ticker]['shares_owned'] * dividend_asset_today
                portfolio_details[ticker]['dividends_cumulative_asset'] += cash_from_dividends_asset
                new_overall_dividend_today += cash_from_dividends_asset # Sommiamo al totale di oggi
                
                if pd.notna(current_price) and current_price > 0:
                    additional_shares = cash_from_dividends_asset / current_price
                    portfolio_details[ticker]['shares_owned'] += additional_shares
                    # print(f"Debug: ...Reinvestito dividendo per {ticker}: {cash_from_dividends_asset:.2f}, Nuove quote: {additional_shares:.4f}")

                else:
                    print(f"ATTENZIONE (PAC): Prezzo non valido per {ticker} il {current_simulation_date_nominal.strftime('%Y-%m-%d')} per reinvestimento dividendi.")


            # C. Calcolo valore della posizione di questo asset
            if pd.notna(current_price): # Usa l'ultimo prezzo noto per la valutazione
                portfolio_value_today += portfolio_details[ticker]['shares_owned'] * current_price
            # Se current_price è NaN (nessun dato per questo asset fino a questa data), il suo contributo è 0
            # o potremmo prendere l'ultimo valore valido, ma current_price già fa questo se non è un giorno di trading

        # Dopo aver processato tutti gli asset per il current_simulation_date_nominal
        if is_investment_day:
            month_counter_for_investment += 1 # Incrementa solo dopo che il giorno di investimento è stato processato per tutti gli asset

        total_dividends_received_overall += new_overall_dividend_today

        portfolio_evolution_records.append({
            'Date': current_simulation_date_nominal, # Usiamo la data nominale del business day
            'InvestedCapital': total_capital_invested_overall,
            'PortfolioValue': portfolio_value_today,
            'DividendsReceivedCumulative': total_dividends_received_overall
            # Potremmo aggiungere qui i dettagli per ogni asset se necessario per debug o output avanzati
        })

    if not portfolio_evolution_records:
        return pd.DataFrame()

    final_df = pd.DataFrame(portfolio_evolution_records)
    final_df.set_index('Date', inplace=True)
    
    # Potrebbe essere necessario un ffill se alcune date di business non avevano prezzi per NESSUN asset
    # e quindi portfolio_value_today era 0 o basato su dati molto vecchi.
    # Per ora, il dataframe è basato sulle date di business calcolate.
    
    return final_df.reset_index()
