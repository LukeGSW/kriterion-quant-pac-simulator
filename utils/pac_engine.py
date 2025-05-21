# simulatore_pac/utils/pac_engine.py

import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# Assicurati che pandas sia importato
# import pandas as pd (già fatto nella tua versione precedente, ma per sicurezza)

def run_pac_simulation( # Rinominata da run_basic_pac_simulation per riflettere nuove funzionalità
    price_data: pd.DataFrame,
    monthly_investment: float,
    start_date_pac: str,
    duration_months: int,
    reinvest_dividends: bool = True # Nuovo parametro
) -> pd.DataFrame:
    if not isinstance(price_data, pd.DataFrame) or price_data.empty:
        print("ERRORE (PAC): price_data non è un DataFrame valido o è vuoto.")
        return pd.DataFrame()
    if 'Adj Close' not in price_data.columns or 'Dividend' not in price_data.columns:
        print("ERRORE (PAC): Colonne 'Adj Close' o 'Dividend' mancanti in price_data.")
        return pd.DataFrame()
    # ... (altri controlli di input come prima) ...
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
        print(f"ERRORE (PAC): Formato data inizio PAC non valido: {start_date_pac}. Usare 'YYYY-MM-DD'.")
        return pd.DataFrame()

    if price_data.index.tz is not None:
        print(f"ERRORE CRITICO (PAC): price_data.index è ancora timezone-aware ({price_data.index.tz})!")
        return pd.DataFrame()

    sim_end_approx = pac_start_dt + relativedelta(months=duration_months)
    mask_start = price_data.index >= (pac_start_dt - timedelta(days=90)) # Buffer più ampio per dati iniziali e dividendi
    mask_end = price_data.index <= (sim_end_approx + timedelta(days=60)) # Buffer più ampio
    relevant_price_data = price_data[mask_start & mask_end].copy()

    if relevant_price_data.empty:
        print(f"INFO (PAC): Nessun dato di prezzo disponibile in relevant_price_data per il periodo PAC.")
        return pd.DataFrame()

    total_shares_owned = 0.0
    total_capital_invested = 0.0
    cash_held = 0.0 # Potremmo usarlo per i dividendi non immediatamente reinvestiti, ma per ora reinvestiamo subito
    total_dividends_received = 0.0 # Tracciamo i dividendi totali
    
    # Creiamo un DataFrame per l'evoluzione giornaliera, partendo dall'indice di relevant_price_data
    # Questo ci aiuterà a gestire gli eventi (investimenti, dividendi) giorno per giorno
    # e poi a costruire il daily_portfolio finale.
    
    # Determiniamo il range di date completo per la simulazione effettiva
    actual_simulation_start_date = pac_start_dt
    # L'ultimo giorno di investimento possibile
    last_potential_investment_date = pac_start_dt + relativedelta(months=duration_months -1)
    # L'ultimo giorno per cui potremmo avere dati rilevanti (es. per il valore del portafoglio)
    # dovrebbe essere circa un mese dopo l'ultimo investimento, o la fine di relevant_price_data
    if relevant_price_data.index[-1] < (last_potential_investment_date + relativedelta(days=1)):
         actual_simulation_end_date = relevant_price_data.index[-1]
    else:
         # Cerchiamo di estendere fino alla fine del mese dell'ultimo investimento o poco oltre
         # per catturare il valore finale del portafoglio
         _end_month_target = last_potential_investment_date + relativedelta(day=31)
         if _end_month_target > relevant_price_data.index[-1]:
             actual_simulation_end_date = relevant_price_data.index[-1]
         else:
             actual_simulation_end_date = _end_month_target
             
    # Filtriamo relevant_price_data per il range effettivo della simulazione per l'equity line
    # Ma per i calcoli di investimento e dividendi, usiamo il relevant_price_data più ampio
    # per trovare i giorni di trading.
    if actual_simulation_start_date > actual_simulation_end_date:
        print("INFO (PAC): La data di inizio simulazione è successiva alla data di fine. Nessuna simulazione possibile.")
        return pd.DataFrame()
        
    portfolio_evolution_records = [] # Lista di dizionari per registrare gli stati

    # Ciclo attraverso ogni giorno nel range di dati storici rilevanti per la simulazione
    # Iniziamo dal primo giorno in cui potremmo fare un investimento o ricevere un dividendo
    
    # Troviamo la prima data di investimento effettiva
    first_investment_date_target = pac_start_dt
    first_investment_date_series = relevant_price_data.index[relevant_price_data.index >= first_investment_date_target]
    if first_investment_date_series.empty:
        print("INFO (PAC): Nessuna data di trading valida trovata per il primo investimento.")
        return pd.DataFrame()
    
    current_simulation_date = first_investment_date_series[0] # Iniziamo da qui
    
    last_date_in_data = relevant_price_data.index.max()
    month_counter_for_investment = 0 # Contatore per i versamenti mensili

    while current_simulation_date <= actual_simulation_end_date and current_simulation_date <= last_date_in_data:
        # Assicurati che current_simulation_date sia un giorno di trading
        if current_simulation_date not in relevant_price_data.index:
            # Passa al prossimo giorno di trading disponibile in relevant_price_data
            next_trading_day_series = relevant_price_data.index[relevant_price_data.index > current_simulation_date]
            if next_trading_day_series.empty:
                break # Non ci sono più giorni di trading
            current_simulation_date = next_trading_day_series[0]
            if current_simulation_date > actual_simulation_end_date: # Non andare oltre la fine della simulazione
                 break
            continue # Riprova il ciclo con la nuova current_simulation_date

        current_price = relevant_price_data.loc[current_simulation_date, 'Adj Close']
        dividend_today = relevant_price_data.loc[current_simulation_date, 'Dividend']

        # 1. Gestione Versamento PAC Mensile
        # Il versamento avviene il primo giorno di trading del mese target del PAC
        investment_date_target_for_this_month = pac_start_dt + relativedelta(months=month_counter_for_investment)
        
        if current_simulation_date >= investment_date_target_for_this_month and month_counter_for_investment < duration_months:
            # Verifica se questo è il primo giorno di trading valido per questo versamento target
            # (potrebbe essere stato saltato un giorno festivo, quindi current_simulation_date è il primo disponibile)
            
            # Per evitare doppi investimenti nello stesso mese target se il loop è giornaliero:
            # Assicuriamoci che questo investimento non sia già stato fatto
            # Questo si ottiene incrementando month_counter_for_investment DOPO l'investimento.
            # La condizione current_simulation_date >= investment_date_target_for_this_month
            # combinata con l'incremento di month_counter_for_investment gestisce questo.
            
            if pd.notna(current_price) and current_price > 0:
                shares_bought_monthly = monthly_investment / current_price
                total_shares_owned += shares_bought_monthly
                total_capital_invested += monthly_investment
                # print(f"Debug: {current_simulation_date.date()} - Investimento PAC: {monthly_investment:.2f}, Quote: {shares_bought_monthly:.4f}, Prezzo: {current_price:.2f}")
            else:
                print(f"ATTENZIONE (PAC): Prezzo non valido ({current_price}) il {current_simulation_date.strftime('%Y-%m-%d')} per investimento PAC. Salto.")
                # Il capitale viene comunque considerato "allocato" per quel mese
                total_capital_invested += monthly_investment
                # In una versione più avanzata, potremmo accumulare questo cash_held e investirlo dopo
            
            month_counter_for_investment += 1 # Passa al prossimo mese target per l'investimento

        # 2. Gestione e Reinvestimento Dividendi
        if reinvest_dividends and dividend_today > 0 and total_shares_owned > 0:
            cash_from_dividends = total_shares_owned * dividend_today
            total_dividends_received += cash_from_dividends
            # print(f"Debug: {current_simulation_date.date()} - Dividendo ricevuto: {cash_from_dividends:.2f} ({total_shares_owned:.4f} quote * {dividend_today:.4f}/quota)")

            if pd.notna(current_price) and current_price > 0:
                additional_shares_from_dividends = cash_from_dividends / current_price
                total_shares_owned += additional_shares_from_dividends
                # print(f"Debug: {current_simulation_date.date()} - Reinvestimento dividendi: {additional_shares_from_dividends:.4f} quote acquistate, Prezzo: {current_price:.2f}")
            else:
                # Se il prezzo non è valido il giorno del dividendo, cosa fare?
                # Per ora, non reinvestiamo e perdiamo l'opportunità (semplificazione).
                # In futuro, potremmo accumulare questo cash e reinvestirlo al prossimo prezzo valido.
                print(f"ATTENZIONE (PAC): Prezzo non valido ({current_price}) il {current_simulation_date.strftime('%Y-%m-%d')} per reinvestimento dividendi. Dividendo non reinvestito in questa data.")


        # Registra lo stato giornaliero
        current_portfolio_value = total_shares_owned * current_price if pd.notna(current_price) else (total_shares_owned * relevant_price_data.loc[:current_simulation_date, 'Adj Close'].ffill().iloc[-1] if total_shares_owned > 0 else 0)

        portfolio_evolution_records.append({
            'Date': current_simulation_date,
            'Price': current_price,
            'InvestedCapital': total_capital_invested,
            'SharesOwned': total_shares_owned,
            'PortfolioValue': current_portfolio_value,
            'CashHeld': cash_held, # Non ancora usato attivamente
            'DividendsReceivedCumulative': total_dividends_received
        })
        
        # Passa al prossimo giorno di trading
        next_trading_day_series = relevant_price_data.index[relevant_price_data.index > current_simulation_date]
        if next_trading_day_series.empty:
            break 
        current_simulation_date = next_trading_day_series[0]


    if not portfolio_evolution_records:
        print("INFO (PAC): Nessun record di evoluzione del portafoglio generato.")
        return pd.DataFrame()

    # Converte i record in DataFrame
    daily_portfolio_df = pd.DataFrame(portfolio_evolution_records)
    daily_portfolio_df['Date'] = pd.to_datetime(daily_portfolio_df['Date'])
    
    # Assicurati che il DataFrame copra l'intero periodo se necessario,
    # ma dato che iteriamo sui giorni di trading e registriamo, dovrebbe già essere abbastanza denso.
    # Se vogliamo una riga per OGNI giorno del calendario (non solo trading days),
    # dovremmo fare un reindex e ffill, ma per ora va bene così.
    
    return daily_portfolio_df.reset_index(drop=True) # Rimuove l'indice numerico e mantiene 'Date' come colonna
