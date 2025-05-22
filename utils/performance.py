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

def get_total_capital_invested(sim_df: pd.DataFrame) -> float:
    """Calcola il capitale totale investito registrato nel DataFrame."""
    if sim_df.empty or 'InvestedCapital' not in sim_df.columns:
        return 0.0
    return sim_df['InvestedCapital'].iloc[-1]

def get_final_portfolio_value(sim_df: pd.DataFrame) -> float:
    """Calcola il valore finale del portafoglio."""
    if sim_df.empty or 'PortfolioValue' not in sim_df.columns:
        return 0.0
    return sim_df['PortfolioValue'].iloc[-1]

def calculate_total_return_percentage(final_value: float, total_invested: float) -> float:
    """Calcola il rendimento totale percentuale."""
    if total_invested <= 0:
        return 0.0
    return ((final_value / total_invested) - 1) * 100

def get_duration_years(sim_df: pd.DataFrame) -> float:
    """Calcola la durata totale della simulazione in anni."""
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
    
    if duration_days < 0: return 0.0
    if duration_days == 0: return 1 / 365.25 if len(sim_df) >= 1 else 0.0
        
    return duration_days / 365.25

def calculate_cagr(final_value: float, initial_value: float, num_years: float) -> float:
    """Calcola il Compound Annual Growth Rate (CAGR)."""
    if initial_value <= 0 or num_years <= 0 or pd.isna(final_value) or pd.isna(initial_value):
        return np.nan
    if final_value < 0 and initial_value > 0: return np.nan 
    if final_value == 0 and initial_value > 0: return -100.0

    try:
        cagr = ((final_value / initial_value) ** (1 / num_years)) - 1
    except (ValueError, OverflowError, ZeroDivisionError):
        return np.nan
    return cagr * 100

# --- Funzioni per Metriche Basate sui Rendimenti ---

def calculate_portfolio_returns(sim_df: pd.DataFrame) -> pd.Series:
    """Calcola i rendimenti giornalieri del portafoglio."""
    if sim_df.empty or 'PortfolioValue' not in sim_df.columns or 'Date' not in sim_df.columns:
        return pd.Series(dtype=float, name="Daily Returns")
    
    df = sim_df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    
    if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 2:
        return pd.Series(dtype=float, name="Daily Returns")
        
    df = df.sort_index()
    returns = df['PortfolioValue'].pct_change()
    return returns.replace([np.inf, -np.inf], np.nan).dropna()

def calculate_annualized_volatility(daily_returns: pd.Series, trading_days_per_year: int = 252) -> float:
    """Calcola la volatilità annualizzata."""
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
    
    if std_dev_excess_return == 0 or pd.isna(std_dev_excess_return) or std_dev_excess_return < 1e-9:
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
        try:
            temp_index = pd.to_datetime(portfolio_values_series.index)
            if isinstance(temp_index, pd.DatetimeIndex):
                portfolio_values_series = portfolio_values_series.copy()
                portfolio_values_series.index = temp_index
            else: return pd.Series(dtype=float, index=portfolio_values_series.index, name="Drawdown (%)")
        except Exception: return pd.Series(dtype=float, index=portfolio_values_series.index, name="Drawdown (%)")

    if len(portfolio_values_series.dropna()) < 2:
        return pd.Series(dtype=float, index=portfolio_values_series.index, name="Drawdown (%)")

    portfolio_values_series = portfolio_values_series.sort_index()
    cumulative_max = portfolio_values_series.cummax()
    drawdown = pd.Series(index=portfolio_values_series.index, dtype=float)
    non_zero_peak_mask = cumulative_max != 0
    drawdown.loc[non_zero_peak_mask] = (portfolio_values_series[non_zero_peak_mask] - cumulative_max[non_zero_peak_mask]) / cumulative_max[non_zero_peak_mask]
    drawdown.loc[cumulative_max == 0] = 0.0
    drawdown.fillna(0.0, inplace=True)
    return drawdown * 100

def calculate_max_drawdown(sim_df: pd.DataFrame) -> float:
    """Calcola il Max Drawdown (MDD) usando la serie di drawdown."""
    if sim_df.empty or 'PortfolioValue' not in sim_df.columns or 'Date' not in sim_df.columns:
        return np.nan
    df_for_drawdown = sim_df.copy()
    if 'Date' in df_for_drawdown.columns: # Assicurati che Date sia colonna prima di convertirla
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
    pac_contribution_start_str: str, # Data inizio contributi PAC
    duration_months_contributions: int, 
    monthly_investment: float, 
    final_portfolio_value: float
) -> tuple[list, list]:
    dates_cf = []
    values_cf = []
    if sim_df.empty or 'Date' not in sim_df.columns: return [], []
    try:
        contribution_start_dt = pd.to_datetime(pac_contribution_start_str)
        if contribution_start_dt.tzinfo is not None:
            contribution_start_dt = contribution_start_dt.tz_localize(None)
    except Exception: return [], []

    for i in range(duration_months_contributions):
        investment_date = contribution_start_dt + relativedelta(months=i)
        dates_cf.append(investment_date.date())
        values_cf.append(-monthly_investment)
    
    final_sim_date = pd.to_datetime(sim_df['Date'].iloc[-1])
    if final_sim_date.tzinfo is not None:
        final_sim_date = final_sim_date.tz_localize(None)
    dates_cf.append(final_sim_date.date())
    values_cf.append(final_portfolio_value)
    return dates_cf, values_cf

def calculate_xirr_metric(dates: list, values: list) -> float:
    if 'pyxirr_xirr' not in globals() or not callable(globals()['pyxirr_xirr']): return np.nan
    if not dates or not values or len(dates) != len(values) or len(dates) < 2 : return np.nan
    has_positive = any(v > 0 for v in values); has_negative = any(v < 0 for v in values)
    if not (has_positive and has_negative): return np.nan
    try:
        temp_df = pd.DataFrame({'dates': pd.to_datetime(dates), 'values': values}).sort_values(by='dates')
        rate = pyxirr_xirr(temp_df['dates'].tolist(), temp_df['values'].tolist())
        return rate * 100 if rate is not None and pd.notna(rate) else np.nan
    except Exception: return np.nan

# --- Funzioni per Dettagli per Asset (WAP, Quote Finali) ---

def get_final_asset_details(asset_details_history_df: pd.DataFrame, tickers: list[str]) -> dict:
    """
    Estrae le quote finali e il capitale investito finale per ogni asset.
    Restituisce un dizionario: {ticker: {'shares': float, 'capital_invested': float}}
    """
    final_details = {}
    if asset_details_history_df.empty or 'Date' not in asset_details_history_df.columns:
        return {ticker: {'shares': 0.0, 'capital_invested': 0.0} for ticker in tickers}
        
    last_day_data = asset_details_history_df.iloc[-1]
    for ticker in tickers:
        shares = last_day_data.get(f'{ticker}_shares', 0.0)
        capital = last_day_data.get(f'{ticker}_capital_invested', 0.0)
        final_details[ticker] = {'shares': shares, 'capital_invested': capital}
    return final_details

def calculate_wap_for_assets(final_asset_details: dict) -> dict:
    """
    Calcola il Prezzo Medio di Carico (WAP) per ogni asset.
    Input: dizionario da get_final_asset_details.
    Restituisce un dizionario: {ticker: WAP_float}
    """
    waps = {}
    for ticker, details in final_asset_details.items():
        final_shares = details['shares']
        total_capital_for_asset = details['capital_invested']
        
        if final_shares > 1e-6 and total_capital_for_asset > 1e-6:
            waps[ticker] = total_capital_for_asset / final_shares
        elif final_shares > 1e-6 and total_capital_for_asset <= 1e-6:
            waps[ticker] = 0.0 
        else:
            waps[ticker] = np.nan
    return waps
