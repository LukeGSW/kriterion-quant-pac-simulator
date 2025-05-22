# simulatore_pac/utils/performance.py

import pandas as pd
import numpy as np
from datetime import datetime # Per type hinting
from dateutil.relativedelta import relativedelta

# Tentativo di importare pyxirr per XIRR
try:
    from pyxirr import xirr as pyxirr_xirr
except ImportError:
    # print("ATTENZIONE (performance.py): Libreria pyxirr non trovata. XIRR non calcolabile.")
    def pyxirr_xirr(dates, values, guess=None): # Funzione placeholder
        return np.nan

# --- Funzioni Metriche di Base ---

def get_total_capital_invested(pac_df: pd.DataFrame) -> float:
    """Calcola il capitale totale investito registrato nel DataFrame del PAC."""
    if pac_df.empty or 'InvestedCapital' not in pac_df.columns:
        return 0.0
    # L'ultimo valore di InvestedCapital è il totale versato
    return pac_df['InvestedCapital'].iloc[-1]

def get_final_portfolio_value(pac_df: pd.DataFrame) -> float:
    """Calcola il valore finale del portafoglio dal DataFrame del PAC."""
    if pac_df.empty or 'PortfolioValue' not in pac_df.columns:
        return 0.0
    return pac_df['PortfolioValue'].iloc[-1]

def calculate_total_return_percentage(final_value: float, total_invested: float) -> float:
    """Calcola il rendimento totale percentuale."""
    if total_invested <= 0:
        return 0.0  # O np.nan se si preferisce non mostrare 0% per nessun investimento
    return ((final_value / total_invested) - 1) * 100

def get_duration_years(sim_df: pd.DataFrame) -> float:
    """Calcola la durata totale della simulazione in anni dal DataFrame (PAC o LS)."""
    if sim_df.empty or 'Date' not in sim_df.columns or len(sim_df.dropna(subset=['Date'])) < 2:
        return 0.0
    
    dates = pd.to_datetime(sim_df['Date']).dropna()
    if len(dates) < 2:
        return 0.0
        
    start_date = dates.iloc[0]
    end_date = dates.iloc[-1]
    
    if pd.isna(start_date) or pd.isna(end_date):
        return 0.0
        
    duration_days = (end_date - start_date).days
    
    if duration_days < 0: # In caso di date invertite o problemi
        return 0.0
    if duration_days == 0: # Se la durata è di un solo giorno o meno
        return 1 / 365.25 if len(sim_df) >= 1 else 0.0
        
    return duration_days / 365.25

def calculate_cagr(final_value: float, initial_value: float, num_years: float) -> float:
    """
    Calcola il Compound Annual Growth Rate (CAGR).
    Per il PAC, 'initial_value' sarà il 'total_invested'.
    """
    if initial_value <= 0 or num_years <= 0 or pd.isna(final_value) or pd.isna(initial_value):
        return np.nan
    if final_value < 0 and initial_value > 0: # Raro ma possibile
        return np.nan 
    if final_value == 0 and initial_value > 0:
        return -100.0

    try:
        cagr = ((final_value / initial_value) ** (1 / num_years)) - 1
    except (ValueError, OverflowError, ZeroDivisionError):
        return np.nan
    return cagr * 100

# --- Funzioni per Metriche Basate sui Rendimenti ---

def calculate_portfolio_returns(sim_df: pd.DataFrame) -> pd.Series:
    """
    Calcola i rendimenti giornalieri del portafoglio.
    Input: DataFrame con colonna 'Date' e 'PortfolioValue'.
    """
    if sim_df.empty or 'PortfolioValue' not in sim_df.columns or 'Date' not in sim_df.columns:
        return pd.Series(dtype=float, name="Daily Returns")
    
    # Usa una copia per evitare SettingWithCopyWarning
    df = sim_df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    
    if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 2:
        return pd.Series(dtype=float, name="Daily Returns")
        
    df = df.sort_index()
    returns = df['PortfolioValue'].pct_change()
    # Rimuovi inf e NaN iniziali o risultanti da divisione per zero (se PortfolioValue era 0)
    return returns.replace([np.inf, -np.inf], np.nan).dropna()

def calculate_annualized_volatility(daily_returns: pd.Series, trading_days_per_year: int = 252) -> float:
    """Calcola la volatilità annualizzata dei rendimenti giornalieri."""
    if daily_returns.empty or daily_returns.isnull().all() or len(daily_returns.dropna()) < 2:
        return np.nan
    volatility = daily_returns.std() * np.sqrt(trading_days_per_year)
    return volatility * 100

def calculate_sharpe_ratio(daily_returns: pd.Series, risk_free_rate_annual: float = 0.0, trading_days_per_year: int = 252) -> float:
    """Calcola lo Sharpe Ratio annualizzato."""
    if daily_returns.empty or daily_returns.isnull().all() or len(daily_returns.dropna()) < 2:
        return np.nan
    
    daily_risk_free_rate = (1 + risk_free_rate_annual)**(1/trading_days_per_year) - 1
    excess_returns = daily_returns - daily_risk_free_rate
    mean_excess_return = excess_returns.mean()
    std_dev_excess_return = excess_returns.std()
    
    if std_dev_excess_return == 0 or pd.isna(std_dev_excess_return) or std_dev_excess_return < 1e-9: # Aggiunto controllo per std molto piccolo
        return 0.0 if np.isclose(mean_excess_return, 0) else (np.inf if mean_excess_return > 0 else -np.inf)
        
    sharpe_ratio_daily = mean_excess_return / std_dev_excess_return
    sharpe_ratio_annualized = sharpe_ratio_daily * np.sqrt(trading_days_per_year)
    return sharpe_ratio_annualized

# --- Funzioni per Drawdown ---

def calculate_drawdown_series(portfolio_values_series: pd.Series) -> pd.Series:
    """Calcola la serie storica del drawdown percentuale."""
    if not isinstance(portfolio_values_series, pd.Series) or portfolio_values_series.empty:
        return pd.Series(dtype=float, name="Drawdown (%)")
    if not isinstance(portfolio_values_series.index, pd.DatetimeIndex):
        # Tentativo di conversione se l'indice è di tipo 'object' o simile ma contiene date
        try:
            temp_index = pd.to_datetime(portfolio_values_series.index)
            if isinstance(temp_index, pd.DatetimeIndex):
                portfolio_values_series = portfolio_values_series.copy()
                portfolio_values_series.index = temp_index
            else:
                # print("Avviso (Drawdown Series): Indice non convertibile a DatetimeIndex.")
                return pd.Series(dtype=float, index=portfolio_values_series.index, name="Drawdown (%)")
        except Exception:
            # print("Eccezione (Drawdown Series): Conversione indice fallita.")
            return pd.Series(dtype=float, index=portfolio_values_series.index, name="Drawdown (%)")

    if len(portfolio_values_series.dropna()) < 2:
        return pd.Series(dtype=float, index=portfolio_values_series.index, name="Drawdown (%)")

    portfolio_values_series = portfolio_values_series.sort_index()
    cumulative_max = portfolio_values_series.cummax()
    drawdown = pd.Series(index=portfolio_values_series.index, dtype=float)
    
    non_zero_peak_mask = cumulative_max != 0
    drawdown.loc[non_zero_peak_mask] = (portfolio_values_series[non_zero_peak_mask] - cumulative_max[non_zero_peak_mask]) / cumulative_max[non_zero_peak_mask]
    drawdown.loc[cumulative_max == 0] = 0.0 # Se il picco è 0, il drawdown è 0 (o NaN se anche il valore è NaN)
    drawdown.fillna(0.0, inplace=True) # Riempi eventuali NaN rimanenti con 0 (es. all'inizio)
    
    return drawdown * 100

def calculate_max_drawdown(sim_df: pd.DataFrame) -> float:
    """Calcola il Max Drawdown (MDD) usando la serie di drawdown."""
    if sim_df.empty or 'PortfolioValue' not in sim_df.columns or 'Date' not in sim_df.columns:
        return np.nan
        
    df_for_drawdown = sim_df.copy()
    df_for_drawdown['Date'] = pd.to_datetime(df_for_drawdown['Date'])
    df_for_drawdown = df_for_drawdown.set_index('Date')
    
    if not isinstance(df_for_drawdown.index, pd.DatetimeIndex) or len(df_for_drawdown) < 2:
        return np.nan

    portfolio_values_series = df_for_drawdown['PortfolioValue']
    if portfolio_values_series.isnull().all() or len(portfolio_values_series.dropna()) < 2:
        return np.nan

    drawdown_series = calculate_drawdown_series(portfolio_values_series)
    
    if drawdown_series.empty or drawdown_series.isnull().all():
        return np.nan
        
    max_drawdown_value = drawdown_series.min()
    return max_drawdown_value if pd.notna(max_drawdown_value) else np.nan

# --- Funzioni per XIRR (usando pyxirr) ---

def generate_cash_flows_for_xirr(
    sim_df: pd.DataFrame, 
    pac_start_date_str: str, # Data inizio contributi PAC (stringa YYYY-MM-DD)
    duration_months_contributions: int, 
    monthly_investment: float, 
    final_portfolio_value: float
) -> tuple[list, list]:
    """
    Prepara le liste di date e valori dei flussi di cassa per XIRR.
    I flussi di cassa sono: -(investimento mensile) per ogni mese di contribuzione,
    e +(valore finale portafoglio) all'ultima data della simulazione.
    """
    dates_cf = []
    values_cf = []

    if sim_df.empty or 'Date' not in sim_df.columns:
        return [], []

    try:
        contribution_start_dt = pd.to_datetime(pac_start_date_str)
        if contribution_start_dt.tzinfo is not None:
            contribution_start_dt = contribution_start_dt.tz_localize(None)
    except Exception:
        # print("Errore conversione start_date_pac_str in generate_cash_flows_for_xirr")
        return [], [] # Impossibile procedere senza una data di inizio valida

    for i in range(duration_months_contributions):
        # Data del versamento (approssimata come inizio del mese target rispetto a pac_start_date_str)
        investment_date = contribution_start_dt + relativedelta(months=i)
        dates_cf.append(investment_date.date()) # Usa oggetto date
        values_cf.append(-monthly_investment)

    # Flusso finale in entrata (valore del portafoglio)
    final_sim_date = pd.to_datetime(sim_df['Date'].iloc[-1])
    if final_sim_date.tzinfo is not None:
        final_sim_date = final_sim_date.tz_localize(None)
        
    dates_cf.append(final_sim_date.date()) # Usa oggetto date
    values_cf.append(final_portfolio_value)
    
    return dates_cf, values_cf

def calculate_xirr_metric(dates: list, values: list) -> float:
    """Calcola l'XIRR usando la libreria pyxirr."""
    if 'pyxirr_xirr' not in globals() or not callable(globals()['pyxirr_xirr']):
        # print("INFO (performance.py): Funzione pyxirr_xirr non disponibile.")
        return np.nan
    if not dates or not values or len(dates) != len(values) or len(dates) < 2 : # XIRR ha bisogno di almeno 2 flussi
        # print("INFO (performance.py): Dati insufficienti per XIRR.")
        return np.nan

    has_positive = any(v > 0 for v in values)
    has_negative = any(v < 0 for v in values)
    if not (has_positive and has_negative):
        # print("INFO (performance.py): XIRR non calcolabile, mancano flussi positivi o negativi.")
        return np.nan
    
    # Assicurati che date e valori siano ordinati per data (pyxirr potrebbe richiederlo o gestirlo)
    # Per sicurezza, anche se pyxirr è robusto:
    try:
        # pyxirr si aspetta che le date siano in ordine cronologico
        # Crea un DataFrame temporaneo per ordinare
        temp_df = pd.DataFrame({'dates': pd.to_datetime(dates), 'values': values})
        temp_df = temp_df.sort_values(by='dates')
        
        # pyxirr può accettare datetime.date o stringhe YYYY-MM-DD o timestamp
        rate = pyxirr_xirr(temp_df['dates'].tolist(), temp_df['values'].tolist())
        return rate * 100 if rate is not None and pd.notna(rate) else np.nan
    except Exception as e:
        # print(f"Errore nel calcolo XIRR con pyxirr: {e}")
        return np.nan

# --- Funzioni per Dettagli per Asset (WAP, Quote Finali) ---
# Queste funzioni prenderanno asset_details_history_df come input

def get_final_shares_per_asset(asset_details_df: pd.DataFrame, tickers: list[str]) -> dict:
    """Estrae le quote finali per ogni asset."""
    final_shares = {}
    if asset_details_df.empty or 'Date' not in asset_details_df.columns:
        return {ticker: 0.0 for ticker in tickers}
        
    last_day_details = asset_details_df.iloc[-1]
    for ticker in tickers:
        shares_col_name = f'{ticker}_shares'
        final_shares[ticker] = last_day_details.get(shares_col_name, 0.0)
    return final_shares

def calculate_wap_per_asset(asset_details_df: pd.DataFrame, tickers: list[str]) -> dict:
    """Calcola il Prezzo Medio di Carico (WAP) per ogni asset."""
    waps = {}
    if asset_details_df.empty or 'Date' not in asset_details_df.columns:
        return {ticker: np.nan for ticker in tickers}

    last_day_details = asset_details_df.iloc[-1]
    for ticker in tickers:
        shares_col_name = f'{ticker}_shares'
        capital_col_name = f'{ticker}_capital_invested'
        
        final_shares = last_day_details.get(shares_col_name, 0.0)
        total_capital_for_asset = last_day_details.get(capital_col_name, 0.0)
        
        if final_shares > 1e-6 and total_capital_for_asset > 1e-6:
            waps[ticker] = total_capital_for_asset / final_shares
        elif final_shares > 1e-6 and total_capital_for_asset <= 1e-6: # Quote senza capitale (es. solo dividendi)
            waps[ticker] = 0.0 
        else:
            waps[ticker] = np.nan
    return waps
# Inserisci questa funzione in utils/performance.py
# Assicurati che sia allo stesso livello di indentazione delle altre definizioni di funzione (non dentro un'altra funzione)

def calculate_annual_returns(portfolio_values_series: pd.Series) -> pd.Series:
    """
    Calcola i rendimenti per ogni anno civile coperto dalla serie di valori di portafoglio.
    L'input portfolio_values_series deve avere un DatetimeIndex.
    """
    if not isinstance(portfolio_values_series, pd.Series) or portfolio_values_series.empty or \
       not isinstance(portfolio_values_series.index, pd.DatetimeIndex) or len(portfolio_values_series.dropna()) < 2:
        return pd.Series(dtype=float, name="Rendimento Annuale (%)")

    portfolio_values_series = portfolio_values_series.sort_index()
    portfolio_values_series = portfolio_values_series[~portfolio_values_series.index.duplicated(keep='last')]

    years = sorted(list(set(portfolio_values_series.index.year)))
    annual_returns_list = []

    if not years:
        return pd.Series(dtype=float, name="Rendimento Annuale (%)")

    for year in years:
        start_of_year_values = portfolio_values_series[portfolio_values_series.index.year == year]
        if start_of_year_values.empty:
            continue

        if year == years[0]:
            initial_value_for_year_calc = start_of_year_values.iloc[0]
        else:
            end_of_previous_year_series = portfolio_values_series[portfolio_values_series.index.year == year - 1]
            if not end_of_previous_year_series.empty:
                initial_value_for_year_calc = end_of_previous_year_series.iloc[-1]
            else:
                 annual_returns_list.append({'Year': year, 'Return': np.nan})
                 continue

        final_value_for_year_calc = start_of_year_values.iloc[-1]

        if pd.notna(initial_value_for_year_calc) and pd.notna(final_value_for_year_calc) and initial_value_for_year_calc != 0:
            year_return = (final_value_for_year_calc / initial_value_for_year_calc) - 1
            annual_returns_list.append({'Year': year, 'Return': year_return * 100})
        else:
            annual_returns_list.append({'Year': year, 'Return': np.nan})

    if not annual_returns_list:
        return pd.Series(dtype=float, name="Rendimento Annuale (%)")

    returns_df = pd.DataFrame(annual_returns_list).set_index('Year')['Return']
    returns_df.name = "Rendimento Annuale (%)"
    return returns_df
# In utils/performance.py, aggiungi queste funzioni:

# --- FUNZIONI PER ROLLING METRICS ---

def calculate_rolling_volatility(
    daily_returns: pd.Series, 
    window_days: int, 
    trading_days_per_year: int = 252
) -> pd.Series:
    """
    Calcola la volatilità annualizzata mobile.
    window_days: la dimensione della finestra in giorni di trading.
    """
    if daily_returns.empty or len(daily_returns.dropna()) < window_days: # Usa dropna() per contare i valori non-NaN
        return pd.Series(dtype=float, name="Rolling Volatility")
    
    rolling_vol = daily_returns.rolling(window=window_days, min_periods=window_days).std() * np.sqrt(trading_days_per_year)
    return rolling_vol.dropna() * 100 # In percentuale

def _calculate_sharpe_for_window(
    window_returns: pd.Series, 
    risk_free_rate_annual: float, 
    trading_days_per_year: int
) -> float:
    """Funzione helper per calcolare Sharpe su una singola finestra."""
    if window_returns.empty or len(window_returns.dropna()) < 2: 
        return np.nan
    
    daily_risk_free_rate = (1 + risk_free_rate_annual)**(1/trading_days_per_year) - 1
    excess_returns = window_returns - daily_risk_free_rate
    mean_excess_return = excess_returns.mean()
    std_dev_excess_return = excess_returns.std()
    
    if std_dev_excess_return == 0 or pd.isna(std_dev_excess_return) or std_dev_excess_return < 1e-9:
        return 0.0 if np.isclose(mean_excess_return, 0) else (np.inf if mean_excess_return > 0 else -np.inf)
        
    sharpe_ratio_window_daily = mean_excess_return / std_dev_excess_return
    sharpe_ratio_window_annualized = sharpe_ratio_window_daily * np.sqrt(trading_days_per_year)
    return sharpe_ratio_window_annualized

def calculate_rolling_sharpe_ratio(
    daily_returns: pd.Series, 
    window_days: int, 
    risk_free_rate_annual: float = 0.0, 
    trading_days_per_year: int = 252
) -> pd.Series:
    """
    Calcola lo Sharpe Ratio annualizzato mobile.
    """
    if daily_returns.empty or len(daily_returns.dropna()) < window_days:
        return pd.Series(dtype=float, name="Rolling Sharpe Ratio")
    
    rolling_sharpe = daily_returns.rolling(window=window_days, min_periods=window_days).apply(
        lambda x: _calculate_sharpe_for_window(x, risk_free_rate_annual, trading_days_per_year),
        raw=False
    )
    return rolling_sharpe.dropna()

def _calculate_cagr_for_window(
    window_portfolio_values: pd.Series, 
    # window_days_actual: int, # Non serve più se usiamo min_periods nella rolling
    trading_days_per_year: int
) -> float:
    """Funzione helper per calcolare CAGR su una singola finestra di valori di portafoglio."""
    if window_portfolio_values.empty or len(window_portfolio_values.dropna()) < 2: # Controlla valori non-NaN
        return np.nan
    
    # Prendi il primo e l'ultimo valore non-NaN nella finestra
    valid_values = window_portfolio_values.dropna()
    if len(valid_values) < 2: # Necessari almeno due punti per un periodo
        return np.nan

    start_value = valid_values.iloc[0]
    end_value = valid_values.iloc[-1]
    
    if start_value <= 0 or pd.isna(start_value) or pd.isna(end_value):
        return np.nan
    if end_value < 0 and start_value > 0: 
         return np.nan 
    if end_value == 0 and start_value > 0:
        return -100.0

    # Il numero di anni è basato sulla lunghezza effettiva della finestra di valori validi
    # non sulla dimensione nominale della finestra, per gestire i bordi.
    # Tuttavia, .rolling(window, min_periods=window) dovrebbe passare finestre piene.
    num_periods_in_window = len(valid_values) # Numero di giorni di trading effettivi nella finestra
    if num_periods_in_window < 2: # Dovrebbe essere già gestito, ma per sicurezza
        return np.nan
        
    num_years_in_window = num_periods_in_window / trading_days_per_year
    if num_years_in_window <= 0:
        return np.nan

    try:
        cagr = ((end_value / start_value) ** (1 / num_years_in_window)) - 1
    except (ValueError, OverflowError, ZeroDivisionError):
        return np.nan
    return cagr * 100

def calculate_rolling_cagr(
    portfolio_values: pd.Series, 
    window_days: int, 
    trading_days_per_year: int = 252
) -> pd.Series:
    """
    Calcola il CAGR mobile.
    portfolio_values: Serie Pandas con i valori del portafoglio e DatetimeIndex.
    """
    if not isinstance(portfolio_values, pd.Series) or portfolio_values.empty or \
       not isinstance(portfolio_values.index, pd.DatetimeIndex) or len(portfolio_values.dropna()) < window_days:
        return pd.Series(dtype=float, name="Rolling CAGR")

    rolling_cagr_series = portfolio_values.rolling(window=window_days, min_periods=window_days).apply( 
        lambda x: _calculate_cagr_for_window(x, trading_days_per_year), # Passa solo x e trading_days
        raw=False 
    )
    return rolling_cagr_series.dropna()
