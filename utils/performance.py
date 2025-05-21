# simulatore_pac/utils/performance.py

import pandas as pd
import numpy as np
# Potrebbe essere necessario installare numpy_financial: pip install numpy-financial
# Se non è già una dipendenza di pandas/numpy che hai.
# Per XIRR più preciso, si potrebbero usare altre librerie o implementazioni custom.
# Per ora, potremmo calcolare un IRR sui flussi di cassa totali se li strutturiamo.
# Oppure, un CAGR è già presente come approssimazione.

def get_total_capital_invested(pac_df: pd.DataFrame) -> float:
    if pac_df.empty or 'InvestedCapital' not in pac_df.columns:
        return 0.0
    return pac_df['InvestedCapital'].iloc[-1]

def get_final_portfolio_value(pac_df: pd.DataFrame) -> float:
    if pac_df.empty or 'PortfolioValue' not in pac_df.columns:
        return 0.0
    return pac_df['PortfolioValue'].iloc[-1]

def calculate_total_return_percentage(final_value: float, total_invested: float) -> float:
    if total_invested <= 0:
        return 0.0
    return ((final_value / total_invested) - 1) * 100

def get_duration_years(pac_df: pd.DataFrame) -> float:
    if pac_df.empty or 'Date' not in pac_df.columns or len(pac_df) < 2:
        return 0.0
    
    dates = pd.to_datetime(pac_df['Date'])
    start_date = dates.iloc[0]
    end_date = dates.iloc[-1]
    duration_days = (end_date - start_date).days
    
    if duration_days <= 0:
        return 1/365.25 if len(pac_df) >=1 else 0.0
    return duration_days / 365.25

def calculate_cagr(final_value: float, total_invested: float, num_years: float) -> float:
    # Nota: Per un PAC, l'IRR è una misura di rendimento più appropriata del CAGR
    # calcolato in questo modo semplice sul totale investito.
    # Questo CAGR misura il tasso di crescita come se l'intero 'total_invested'
    # fosse stato messo all'inizio.
    if total_invested <= 0 or num_years <= 0:
        return np.nan
    cagr = ((final_value / total_invested) ** (1 / num_years)) - 1
    return cagr * 100

# --- Nuove Metriche Avanzate ---

def calculate_portfolio_returns(pac_df: pd.DataFrame) -> pd.Series:
    """Calcola i rendimenti giornalieri del portafoglio."""
    if pac_df.empty or 'PortfolioValue' not in pac_df.columns or len(pac_df) < 2:
        return pd.Series(dtype=float)
    # Assicura che l'indice sia 'Date' e che sia ordinato
    df = pac_df.set_index('Date') if 'Date' in pac_df.columns else pac_df
    df = df.sort_index()
    return df['PortfolioValue'].pct_change().dropna()

def calculate_annualized_volatility(daily_returns: pd.Series, trading_days_per_year: int = 252) -> float:
    """Calcola la volatilità annualizzata dei rendimenti giornalieri."""
    if daily_returns.empty or len(daily_returns) < 2:
        return np.nan
    volatility = daily_returns.std() * np.sqrt(trading_days_per_year)
    return volatility * 100 # In percentuale

def calculate_sharpe_ratio(daily_returns: pd.Series, risk_free_rate_annual: float = 0.0, trading_days_per_year: int = 252) -> float:
    """
    Calcola lo Sharpe Ratio annualizzato.
    risk_free_rate_annual è il tasso risk-free annuale (es. 0.01 per 1%).
    """
    if daily_returns.empty or len(daily_returns) < 2:
        return np.nan
    
    excess_returns = daily_returns - (risk_free_rate_annual / trading_days_per_year)
    mean_excess_return = excess_returns.mean()
    std_dev_excess_return = excess_returns.std()
    
    if std_dev_excess_return == 0: # Evita divisione per zero se non c'è volatilità
        return np.nan if mean_excess_return == 0 else np.inf * np.sign(mean_excess_return)
        
    sharpe_ratio_daily = mean_excess_return / std_dev_excess_return
    sharpe_ratio_annualized = sharpe_ratio_daily * np.sqrt(trading_days_per_year)
    return sharpe_ratio_annualized

def calculate_max_drawdown(pac_df: pd.DataFrame) -> float:
    """
    Calcola il Max Drawdown (MDD) del valore del portafoglio.
    Restituisce il MDD come percentuale negativa (es. -0.25 per -25%).
    """
    if pac_df.empty or 'PortfolioValue' not in pac_df.columns or len(pac_df) < 2:
        return np.nan
        
    df = pac_df.set_index('Date') if 'Date' in pac_df.columns else pac_df
    df = df.sort_index()
    
    portfolio_values = df['PortfolioValue']
    cumulative_max = portfolio_values.cummax()
    drawdown = (portfolio_values - cumulative_max) / cumulative_max
    max_drawdown_value = drawdown.min() # Il valore più negativo
    
    return max_drawdown_value * 100 if pd.notna(max_drawdown_value) else np.nan


# --- Placeholder per IRR/XIRR e Sortino Ratio ---
# L'IRR per un PAC richiede la serie dei flussi di cassa (investimenti come negativi, valore finale come positivo)
# e le relative date. Questo è un po' più complesso da strutturare qui direttamente dal pac_df aggregato.
# Potremmo aver bisogno di modificare pac_engine per restituire anche i flussi di cassa.

# Per XIRR, si può usare una libreria come `numpy_financial.irr` (per periodi regolari)
# o implementazioni custom/librerie terze per periodi irregolari.

# def calculate_irr_pac(pac_df: pd.DataFrame, monthly_investment: float, final_value: float) -> float:
#     """
#     Calcola un IRR approssimato per il PAC.
#     Questa è una semplificazione e potrebbe non essere accurata come un XIRR.
#     Assume che gli investimenti siano regolari.
#     """
#     # ... logica per costruire i flussi di cassa ...
#     # Esempio: flussi = [-monthly_investment] * num_investimenti
#     # flussi[-1] += final_value # Aggiungi valore finale all'ultimo periodo o come flusso finale
#     # import numpy_financial as npf
#     # try:
#     #     irr = npf.irr(flussi) * 12 * 100 # Annualizza e metti in %
#     #     return irr
#     # except Exception:
#     #     return np.nan
#     return np.nan # Da implementare

# def calculate_sortino_ratio(daily_returns: pd.Series, risk_free_rate_annual: float = 0.0, target_return_annual: float = 0.0, trading_days_per_year: int = 252) -> float:
#     """
#     Calcola il Sortino Ratio annualizzato.
#     Considera solo la downside deviation (volatilità dei rendimenti negativi).
#     """
#     # ... logica per Sortino Ratio ...
#     # daily_target_return = target_return_annual / trading_days_per_year
#     # daily_risk_free = risk_free_rate_annual / trading_days_per_year
#     # excess_returns = daily_returns - daily_risk_free
    
#     # negative_excess_returns = excess_returns[excess_returns < (daily_target_return - daily_risk_free)] # o semplicemente < 0
#     # downside_deviation = np.sqrt(np.sum(negative_excess_returns**2) / len(daily_returns)) * np.sqrt(trading_days_per_year)
    
#     # if downside_deviation == 0:
#     #     return np.nan
#     # mean_excess_return_annualized = excess_returns.mean() * trading_days_per_year
#     # return mean_excess_return_annualized / downside_deviation
#     return np.nan # Da implementare
