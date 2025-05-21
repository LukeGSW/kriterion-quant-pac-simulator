# simulatore_pac/utils/performance.py

import pandas as pd
import numpy as np
from datetime import datetime # Aggiunto per type hinting se necessario
from dateutil.relativedelta import relativedelta

# Tentativo di importare librerie opzionali per metriche avanzate
try:
    from pyxirr import xirr as pyxirr_xirr # Rinomina per evitare conflitti se definiamo una nostra xirr
except ImportError:
    print("ATTENZIONE: Libreria pyxirr non trovata. Il calcolo XIRR non sarà disponibile.")
    def pyxirr_xirr(dates, values, guess=None): # Funzione placeholder
        return np.nan

try:
    import empyrical
except ImportError:
    print("ATTENZIONE: Libreria empyrical non trovata. Sortino Ratio e altre metriche da empyrical non saranno disponibili.")
    empyrical = None # Placeholder


def get_total_capital_invested(pac_df: pd.DataFrame) -> float:
    """
    Calcola il capitale totale investito alla fine del periodo del PAC.
    """
    if pac_df.empty or 'InvestedCapital' not in pac_df.columns:
        return 0.0
    return pac_df['InvestedCapital'].iloc[-1]

def get_final_portfolio_value(pac_df: pd.DataFrame) -> float:
    """
    Calcola il valore finale del portafoglio alla fine del periodo del PAC.
    """
    if pac_df.empty or 'PortfolioValue' not in pac_df.columns:
        return 0.0
    return pac_df['PortfolioValue'].iloc[-1]

def calculate_total_return_percentage(final_value: float, total_invested: float) -> float:
    """
    Calcola il rendimento totale percentuale.
    """
    if total_invested <= 0:
        return 0.0
    return ((final_value / total_invested) - 1) * 100

def get_duration_years(pac_df: pd.DataFrame) -> float:
    """
    Calcola la durata totale della simulazione PAC in anni.
    """
    if pac_df.empty or 'Date' not in pac_df.columns or len(pac_df) < 2:
        return 0.0
    
    # Assicura che 'Date' sia datetime
    if not pd.api.types.is_datetime64_any_dtype(pac_df['Date']):
        try:
            dates = pd.to_datetime(pac_df['Date'])
        except Exception:
            print("Errore: Impossibile convertire la colonna 'Date' in datetime per get_duration_years.")
            return 0.0
    else:
        dates = pac_df['Date']

    start_date = dates.iloc[0]
    end_date = dates.iloc[-1]
    
    if pd.isna(start_date) or pd.isna(end_date):
        print("Errore: Date di inizio o fine mancanti in pac_df per get_duration_years.")
        return 0.0
        
    duration_days = (end_date - start_date).days
    
    if duration_days <= 0:
        # Se la durata è 0 o negativa (es. solo 1 riga di dati), considera una frazione d'anno
        return 1/365.25 if len(pac_df) >=1 else 0.0
        
    return duration_days / 365.25

def calculate_cagr(final_value: float, total_invested: float, num_years: float) -> float:
    """
    Calcola il Compound Annual Growth Rate (CAGR).
    Nota: Per un PAC, l'IRR/XIRR è una misura di rendimento più appropriata.
    """
    if total_invested <= 0 or num_years <= 0:
        return np.nan
    # Per evitare errori con (valore negativo)**(potenza frazionaria) se final_value < 0
    if final_value < 0 and total_invested > 0 : # Raro per PAC ma possibile con forti perdite
        return np.nan # o una gestione specifica per CAGR negativi
    if final_value == 0 and total_invested > 0: # Perdita del 100%
        return -100.0

    try:
        cagr = ((final_value / total_invested) ** (1 / num_years)) - 1
    except (ValueError, OverflowError, ZeroDivisionError): # Gestisce errori matematici
        return np.nan
    return cagr * 100

def calculate_portfolio_returns(pac_df: pd.DataFrame) -> pd.Series:
    """
    Calcola i rendimenti giornalieri del portafoglio.
    Assicura che l'indice sia 'Date' e che sia DatetimeIndex.
    """
    if pac_df.empty or 'PortfolioValue' not in pac_df.columns or len(pac_df) < 2:
        return pd.Series(dtype=float)

    # Usa una copia per evitare SettingWithCopyWarning se pac_df è una slice
    df = pac_df.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
    
    if not isinstance(df.index, pd.DatetimeIndex):
        print("Errore: L'indice del DataFrame non è DatetimeIndex per calculate_portfolio_returns.")
        return pd.Series(dtype=float)
        
    df = df.sort_index()
    returns = df['PortfolioValue'].pct_change().dropna()
    return returns.replace([np.inf, -np.inf], np.nan).dropna() # Rimuovi inf e NaN risultanti

def calculate_annualized_volatility(daily_returns: pd.Series, trading_days_per_year: int = 252) -> float:
    """
    Calcola la volatilità annualizzata dei rendimenti giornalieri.
    """
    if daily_returns.empty or len(daily_returns) < 2 or daily_returns.isnull().all():
        return np.nan
    volatility = daily_returns.std() * np.sqrt(trading_days_per_year)
    return volatility * 100

def calculate_sharpe_ratio(daily_returns: pd.Series, risk_free_rate_annual: float = 0.0, trading_days_per_year: int = 252) -> float:
    """
    Calcola lo Sharpe Ratio annualizzato.
    """
    if daily_returns.empty or len(daily_returns) < 2 or daily_returns.isnull().all():
        return np.nan
    
    # Calcolo del rendimento giornaliero risk-free
    daily_risk_free_rate = (1 + risk_free_rate_annual)**(1/trading_days_per_year) - 1
    
    excess_returns = daily_returns - daily_risk_free_rate # Rendimenti giornalieri in eccesso
    mean_excess_return = excess_returns.mean()
    std_dev_excess_return = excess_returns.std()
    
    if std_dev_excess_return == 0 or pd.isna(std_dev_excess_return):
        # Se non c'è volatilità, lo Sharpe Ratio non è ben definito o è infinito.
        # Restituiamo NaN, o un valore molto grande se il rendimento medio in eccesso è positivo.
        return np.nan if mean_excess_return == 0 else (np.inf if mean_excess_return > 0 else -np.inf)
        
    sharpe_ratio_daily = mean_excess_return / std_dev_excess_return
    sharpe_ratio_annualized = sharpe_ratio_daily * np.sqrt(trading_days_per_year)
    return sharpe_ratio_annualized

def calculate_max_drawdown(pac_df: pd.DataFrame) -> float:
    """
    Calcola il Max Drawdown (MDD) del valore del portafoglio.
    Restituisce il MDD come percentuale negativa (es. -25.0 per -25%).
    """
    if pac_df.empty or 'PortfolioValue' not in pac_df.columns or len(pac_df) < 2:
        return np.nan
    
    # Usa una copia e imposta l'indice
    df = pac_df.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        
    if not isinstance(df.index, pd.DatetimeIndex):
        print("Errore: L'indice non è DatetimeIndex per calculate_max_drawdown.")
        return np.nan
        
    df = df.sort_index()
    portfolio_values = df['PortfolioValue']
    
    if portfolio_values.isnull().all() or len(portfolio_values.dropna()) < 2:
        return np.nan

    cumulative_max = portfolio_values.cummax()
    drawdown = (portfolio_values - cumulative_max) / cumulative_max
    # Sostituisci -inf con NaN se cumulative_max è 0 (capita se il portafoglio parte da 0)
    drawdown.replace(-np.inf, np.nan, inplace=True) 
    max_drawdown_value = drawdown.min()
    
    return max_drawdown_value * 100 if pd.notna(max_drawdown_value) else np.nan

# --- Funzioni per XIRR (usando pyxirr) ---

def generate_cash_flows_for_xirr(
    pac_df: pd.DataFrame, 
    start_date_pac_str: str, # Data inizio PAC originale per il primo flusso
    duration_months: int, 
    monthly_investment: float, 
    final_portfolio_value: float
) -> tuple[list, list]:
    """
    Prepara le liste di date e valori dei flussi di cassa per il calcolo XIRR.
    Semplificazione: assume versamenti all'inizio di ogni mese target PAC,
    e valore finale all'ultima data del pac_df.
    """
    dates_outflows = []
    values_outflows = []

    if pac_df.empty or 'Date' not in pac_df.columns:
        return [], []

    try:
        actual_pac_start_date = pd.to_datetime(start_date_pac_str)
        if actual_pac_start_date.tzinfo is not None: # Assicura che sia naive
            actual_pac_start_date = actual_pac_start_date.tz_localize(None)
    except Exception as e:
        print(f"Errore nella conversione di start_date_pac_str: {e}")
        # Fallback alla prima data del DataFrame se la conversione fallisce
        actual_pac_start_date = pd.to_datetime(pac_df['Date'].iloc[0])


    for i in range(duration_months):
        # Data del versamento (semplificata come inizio del mese target)
        investment_date = actual_pac_start_date + relativedelta(months=i)
        dates_outflows.append(investment_date)
        values_outflows.append(-monthly_investment) # Flusso in uscita

    # Flusso finale in entrata (valore del portafoglio)
    final_date = pd.to_datetime(pac_df['Date'].iloc[-1])
    if final_date.tzinfo is not None: # Assicura che sia naive
        final_date = final_date.tz_localize(None)
        
    dates_outflows.append(final_date)
    values_outflows.append(final_portfolio_value) # Flusso in entrata
    
    return dates_outflows, values_outflows

def calculate_xirr_metric(dates: list, values: list) -> float:
    """
    Calcola l'XIRR usando la libreria pyxirr.
    """
    if 'pyxirr_xirr' not in globals() or not callable(globals()['pyxirr_xirr']):
        print("INFO: Funzione pyxirr_xirr non disponibile (libreria pyxirr non caricata).")
        return np.nan
    if not dates or not values or len(dates) != len(values):
        print("INFO: Date o valori mancanti o di lunghezza diversa per XIRR.")
        return np.nan

    # pyxirr si aspetta che ci sia almeno un flusso positivo e uno negativo.
    has_positive = any(v > 0 for v in values)
    has_negative = any(v < 0 for v in values)

    if not (has_positive and has_negative):
        print("INFO: XIRR non calcolabile, mancano flussi positivi o negativi.")
        return np.nan
    
    # Converti date in oggetti date di Python se sono Timestamp di Pandas per pyxirr
    python_dates = [d.date() if isinstance(d, pd.Timestamp) else d for d in dates]

    try:
        # pyxirr potrebbe sollevare eccezioni se non converge o per input errati
        rate = pyxirr_xirr(python_dates, values) # guess opzionale
        return rate * 100 if rate is not None and pd.notna(rate) else np.nan
    except Exception as e:
        print(f"Errore nel calcolo XIRR con pyxirr: {e}")
        return np.nan

# --- Funzione per Sortino Ratio (usando empyrical) ---

def calculate_sortino_ratio_empyrical(daily_returns: pd.Series, required_return_annual: float = 0.0) -> float:
    """
    Calcola il Sortino Ratio annualizzato usando empyrical.
    required_return_annual è il tasso di rendimento minimo accettabile (MAR) annuale.
    Empyrical annualizza automaticamente.
    """
    if empyrical is None:
        print("INFO: Libreria empyrical non disponibile. Sortino Ratio non calcolabile.")
        return np.nan
    if daily_returns.empty or len(daily_returns) < 2 or daily_returns.isnull().all():
        return np.nan
    
    # Empyrical si aspetta che required_return sia il rendimento medio giornaliero del target
    # Se required_return_annual è 0.02 (2%), il giornaliero è (1+0.02)**(1/252)-1
    daily_required_return = (1 + required_return_annual)**(1/252) - 1

    try:
        sortino = empyrical.sortino_ratio(daily_returns, required_return=daily_required_return)
        return sortino if pd.notna(sortino) else np.nan
    except Exception as e:
        print(f"Errore nel calcolo del Sortino Ratio con empyrical: {e}")
        return np.nan
