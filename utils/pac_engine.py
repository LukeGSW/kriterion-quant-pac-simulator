# --- Contenuto di utils/pac_engine.py (CON CORREZIONE PER FUTUREWARNING) ---
def run_basic_pac_simulation(
    price_data: pd.DataFrame,
    monthly_investment: float,
    start_date_pac: str,
    duration_months: int
) -> pd.DataFrame:
    if not isinstance(price_data, pd.DataFrame) or price_data.empty:
        print("ERRORE (PAC): price_data non è un DataFrame valido o è vuoto.")
        return pd.DataFrame()
    if 'Adj Close' not in price_data.columns:
        print("ERRORE (PAC): La colonna 'Adj Close' è mancante in price_data.")
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
        print(f"ERRORE (PAC): Formato data inizio PAC non valido: {start_date_pac}. Usare 'YYYY-MM-DD'.")
        return pd.DataFrame()

    if price_data.index.tz is not None:
        print(f"ERRORE CRITICO (PAC): price_data.index è ancora timezone-aware ({price_data.index.tz})!")
        return pd.DataFrame()

    sim_end_approx = pac_start_dt + relativedelta(months=duration_months)
    mask_start = price_data.index >= (pac_start_dt - timedelta(days=7))
    mask_end = price_data.index <= (sim_end_approx + timedelta(days=31))
    relevant_price_data = price_data[mask_start & mask_end].copy()

    if relevant_price_data.empty:
        print(f"INFO (PAC): Nessun dato di prezzo disponibile in relevant_price_data per il periodo PAC (da {pac_start_dt.strftime('%Y-%m-%d')} circa).")
        return pd.DataFrame()

    total_shares_owned = 0.0
    total_capital_invested = 0.0
    cash_held = 0.0
    pac_evolution = []

    for month_num in range(duration_months):
        investment_date_target = pac_start_dt + relativedelta(months=month_num)
        actual_investment_date_series = relevant_price_data.index[relevant_price_data.index >= investment_date_target]
        
        if actual_investment_date_series.empty:
            print(f"INFO (PAC): Nessuna data di trading trovata per il mese {month_num + 1} (target: {investment_date_target.strftime('%Y-%m-%d')}). Simulazione PAC potrebbe terminare prima.")
            break
        
        actual_investment_date = actual_investment_date_series[0]
        current_price = relevant_price_data.loc[actual_investment_date, 'Adj Close']

        if pd.isna(current_price) or current_price <= 0:
            print(f"ATTENZIONE (PAC): Prezzo non valido ({current_price}) il {actual_investment_date.strftime('%Y-%m-%d')}. Salto investimento per questo mese.")
            total_capital_invested += monthly_investment
            last_known_price_for_value = relevant_price_data.loc[actual_investment_date, 'Adj Close']
            if pd.isna(last_known_price_for_value):
                temp_price_series = relevant_price_data.loc[:actual_investment_date, 'Adj Close'].ffill()
                last_known_price_for_value = temp_price_series.iloc[-1] if not temp_price_series.empty and not pd.isna(temp_price_series.iloc[-1]) else 0
            
            pac_evolution.append({
                'Date': actual_investment_date, 'InvestedCapital': total_capital_invested,
                'SharesOwned': total_shares_owned,
                'PortfolioValue': total_shares_owned * last_known_price_for_value,
                'CashHeld': cash_held + monthly_investment
            })
            continue

        shares_bought = monthly_investment / current_price
        total_shares_owned += shares_bought
        total_capital_invested += monthly_investment
        pac_evolution.append({
            'Date': actual_investment_date, 'InvestedCapital': total_capital_invested,
            'SharesOwned': total_shares_owned,
            'PortfolioValue': total_shares_owned * current_price,
            'CashHeld': cash_held
        })

    if not pac_evolution:
        print("INFO (PAC): Nessun investimento è stato effettuato durante la simulazione PAC.")
        return pd.DataFrame()
        
    pac_df = pd.DataFrame(pac_evolution)
    if 'Date' in pac_df.columns:
        pac_df['Date'] = pd.to_datetime(pac_df['Date'])
        pac_df.set_index('Date', inplace=True)

    if not pac_df.empty:
        start_equity_date = pac_df.index.min()
        # Usiamo l'indice di relevant_price_data per assicurarci di avere tutti i giorni di trading
        # fino alla fine del periodo di simulazione o fino all'ultimo dato disponibile.
        end_equity_date = relevant_price_data[relevant_price_data.index >= start_equity_date].index.max()
        
        if pd.isna(end_equity_date) or pd.isna(start_equity_date):
             print("ATTENZIONE (PAC): Date di inizio/fine per l'equity line non valide.")
             if not pac_df.empty : return pac_df.reset_index()
             return pd.DataFrame()

        # Creiamo daily_portfolio partendo da una slice di relevant_price_data
        # Questo assicura che daily_portfolio non sia una slice di una slice in modo ambiguo
        daily_portfolio = relevant_price_data.loc[start_equity_date:end_equity_date, ['Adj Close']].copy()
        daily_portfolio.rename(columns={'Adj Close': 'Price'}, inplace=True)
        
        # Unisci i dati del PAC (che sono 'a salti') con l'indice giornaliero
        # Assegnando a nuove colonne o sovrascrivendo colonne di daily_portfolio
        # Questo approccio è generalmente più sicuro rispetto a modifiche inplace su slice.
        temp_join = pac_df[['InvestedCapital', 'SharesOwned', 'CashHeld']]
        daily_portfolio = daily_portfolio.join(temp_join)

        # ***** MODIFICHE PER FUTUREWARNING *****
        daily_portfolio['InvestedCapital'] = daily_portfolio['InvestedCapital'].ffill()
        daily_portfolio['SharesOwned'] = daily_portfolio['SharesOwned'].ffill()
        daily_portfolio['CashHeld'] = daily_portfolio['CashHeld'].ffill()
        
        first_investment_date_in_pac_df = pac_df.index.min()
        # Per i giorni prima del primo investimento, impostiamo i valori a 0
        cols_to_zero_out = ['InvestedCapital', 'SharesOwned', 'CashHeld']
        for col in cols_to_zero_out:
            daily_portfolio.loc[daily_portfolio.index < first_investment_date_in_pac_df, col] = 0.0
        
        # Riempi i NaN rimanenti (potrebbero esserci se ffill non ha coperto tutto dall'inizio)
        daily_portfolio['InvestedCapital'] = daily_portfolio['InvestedCapital'].fillna(0)
        daily_portfolio['SharesOwned'] = daily_portfolio['SharesOwned'].fillna(0)
        daily_portfolio['CashHeld'] = daily_portfolio['CashHeld'].fillna(0)
        # ****************************************

        daily_portfolio['PortfolioValue'] = daily_portfolio['SharesOwned'] * daily_portfolio['Price']
        daily_portfolio.index.name = 'Date'
        return daily_portfolio.reset_index()
    else:
        return pd.DataFrame()

# --- Fine contenuto di utils/pac_engine.py ---
