# simulatore_pac/utils/performance.py

import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

try:
    from pyxirr import xirr as pyxirr_xirr
except ImportError:
    def pyxirr_xirr(dates, values, guess=None):
        return np.nan

def get_total_capital_invested(sim_df: pd.DataFrame) -> float:
    if sim_df.empty or 'InvestedCapital' not in sim_df.columns:
        return 0.0
    return sim_df['InvestedCapital'].iloc[-1]

def get_final_portfolio_value(sim_df: pd.DataFrame) -> float:
    if sim_df.empty or 'PortfolioValue' not in sim_df.columns:
        return 0.0
    return sim_df['PortfolioValue'].iloc[-1]

def calculate_total_return_percentage(final_value: float, total_invested: float) -> float:
    if total_invested <= 0:
        return 0.0
    return ((final_value / total_invested) - 1) * 100

def get_duration_years(sim_df: pd.DataFrame) -> float:
    if sim_df.empty or 'Date' not in sim_df.columns or len(sim_df.dropna(subset=['Date'])) < 2:
        return 0.0
    dates = pd.to_datetime(sim_df['Date']).dropna()
    if len(dates) < 2: return 0.0
    start_date, end_date = dates.iloc[0], dates.iloc[-1]
    if pd.isna(start_date) or pd.isna(end_date): return 0.0
    duration_days = (end_date - start_date).days
    if duration_days < 0: return 0.0
    return (1 / 365.25) if duration_days == 0 and len(sim_df) >= 1 else (duration_days / 365.25)

def calculate_cagr(final_value: float, initial_value: float, num_years: float) -> float:
    if initial_value <= 0 or num_years <= 0 or pd.isna(final_value) or pd.isna(initial_value): return np.nan
    if final_value < 0 and initial_value > 0: return np.nan 
    if final_value == 0 and initial_value > 0: return -100.0
    try:
        cagr = ((final_value / initial_value) ** (1 / num_years)) - 1
    except (ValueError, OverflowError, ZeroDivisionError): return np.nan
    return cagr * 100

def calculate_portfolio_returns(sim_df: pd.DataFrame) -> pd.Series:
    if sim_df.empty or 'PortfolioValue' not in sim_df.columns or 'Date' not in sim_df.columns:
        return pd.Series(dtype=float, name="Daily Returns")
    df = sim_df.copy(); df['Date'] = pd.to_datetime(df['Date']); df = df.set_index('Date')
    if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 2: return pd.Series(dtype=float, name="Daily Returns")
    returns = df['PortfolioValue'].pct_change()
    return returns.replace([np.inf, -np.inf], np.nan).dropna()

def calculate_annualized_volatility(daily_returns: pd.Series, trading_days_per_year: int = 252) -> float:
    if daily_returns.empty or daily_returns.isnull().all() or len(daily_returns.dropna()) < 2: return np.nan
    return daily_returns.std() * np.sqrt(trading_days_per_year) * 100

def calculate_sharpe_ratio(daily_returns: pd.Series, risk_free_rate_annual: float = 0.0, trading_days_per_year: int = 252) -> float:
    if daily_returns.empty or daily_returns.isnull().all() or len(daily_returns.dropna()) < 2: return np.nan
    daily_rf = (1 + risk_free_rate_annual)**(1/trading_days_per_year) - 1
    excess_returns = daily_returns - daily_rf
    mean_er = excess_returns.mean(); std_er = excess_returns.std()
    if std_er == 0 or pd.isna(std_er) or std_er < 1e-9: return 0.0 if np.isclose(mean_er, 0) else (np.inf if mean_er > 0 else -np.inf)
    return (mean_er / std_er) * np.sqrt(trading_days_per_year)

def calculate_drawdown_series(portfolio_values_series: pd.Series) -> pd.Series:
    name = "Drawdown (%)"
    if not isinstance(portfolio_values_series, pd.Series) or portfolio_values_series.empty: return pd.Series(dtype=float, name=name)
    if not isinstance(portfolio_values_series.index, pd.DatetimeIndex):
        try:
            temp_index = pd.to_datetime(portfolio_values_series.index)
            if isinstance(temp_index, pd.DatetimeIndex): portfolio_values_series = portfolio_values_series.copy(); portfolio_values_series.index = temp_index
            else: return pd.Series(dtype=float, index=portfolio_values_series.index, name=name)
        except Exception: return pd.Series(dtype=float, index=portfolio_values_series.index, name=name)
    if len(portfolio_values_series.dropna()) < 2: return pd.Series(dtype=float, index=portfolio_values_series.index, name=name)
    pv_sorted = portfolio_values_series.sort_index(); cumulative_max = pv_sorted.cummax()
    drawdown = pd.Series(index=pv_sorted.index, dtype=float)
    mask = cumulative_max != 0
    drawdown.loc[mask] = (pv_sorted[mask] - cumulative_max[mask]) / cumulative_max[mask]
    drawdown.loc[cumulative_max == 0] = 0.0; drawdown.fillna(0.0, inplace=True)
    return drawdown * 100

def calculate_max_drawdown(sim_df: pd.DataFrame) -> float:
    if sim_df.empty or 'PortfolioValue' not in sim_df.columns or 'Date' not in sim_df.columns: return np.nan
    df_dd = sim_df.copy(); df_dd['Date'] = pd.to_datetime(df_dd['Date']); df_dd = df_dd.set_index('Date')
    if not isinstance(df_dd.index, pd.DatetimeIndex) or len(df_dd) < 2: return np.nan
    pv_series = df_dd['PortfolioValue']
    if pv_series.isnull().all() or len(pv_series.dropna()) < 2: return np.nan
    dd_series = calculate_drawdown_series(pv_series)
    if dd_series.empty or dd_series.isnull().all(): return np.nan
    mdd_val = dd_series.min()
    return mdd_val if pd.notna(mdd_val) else np.nan

def generate_cash_flows_for_xirr(sim_df: pd.DataFrame, pac_contrib_start_str: str, duration_months_contrib: int, monthly_inv: float, final_pv: float) -> tuple[list, list]:
    dates, values = [], []
    if sim_df.empty or 'Date' not in sim_df.columns: return [], []
    try:
        contrib_start_dt = pd.to_datetime(pac_contrib_start_str)
        if contrib_start_dt.tzinfo is not None: contrib_start_dt = contrib_start_dt.tz_localize(None)
    except Exception: return [], []
    for i in range(duration_months_contrib):
        dates.append((contrib_start_dt + relativedelta(months=i)).date()); values.append(-monthly_inv)
    final_sim_dt = pd.to_datetime(sim_df['Date'].iloc[-1])
    if final_sim_dt.tzinfo is not None: final_sim_dt = final_sim_dt.tz_localize(None)
    dates.append(final_sim_dt.date()); values.append(final_pv)
    return dates, values

def calculate_xirr_metric(dates: list, values: list) -> float:
    if 'pyxirr_xirr' not in globals() or not callable(globals()['pyxirr_xirr']): return np.nan
    if not (dates and values and len(dates) == len(values) and len(dates) >= 2): return np.nan
    if not (any(v > 0 for v in values) and any(v < 0 for v in values)): return np.nan
    try:
        temp_df = pd.DataFrame({'dates': pd.to_datetime(dates), 'values': values}).sort_values(by='dates')
        rate = pyxirr_xirr(temp_df['dates'].tolist(), temp_df['values'].tolist())
        return rate * 100 if rate is not None and pd.notna(rate) else np.nan
    except Exception: return np.nan

def get_final_asset_details(asset_details_hist_df: pd.DataFrame, tickers: list[str]) -> dict:
    final_details = {}
    if asset_details_hist_df.empty or 'Date' not in asset_details_hist_df.columns:
        return {ticker: {'shares': 0.0, 'capital_invested': 0.0} for ticker in tickers}
    last_day = asset_details_hist_df.iloc[-1]
    for ticker in tickers:
        final_details[ticker] = {'shares': last_day.get(f'{ticker}_shares',0.0), 'capital_invested': last_day.get(f'{ticker}_capital_invested',0.0)}
    return final_details

def calculate_wap_for_assets(final_asset_details: dict) -> dict:
    waps = {}
    for ticker, details in final_asset_details.items():
        s, c = details['shares'], details['capital_invested']
        if s > 1e-6 and c > 1e-6: waps[ticker] = c / s
        elif s > 1e-6 and c <= 1e-6: waps[ticker] = 0.0 
        else: waps[ticker] = np.nan
    return waps
# In utils/performance.py
# ... (importazioni e funzioni esistenti) ...

def calculate_tracking_error(portfolio_daily_returns: pd.Series, 
                             benchmark_daily_returns: pd.Series, 
                             trading_days_per_year: int = 252) -> float:
    """
    Calcola il Tracking Error annualizzato tra i rendimenti di un portafoglio e un benchmark.
    Assicura che le serie di rendimenti siano allineate per data.
    """
    if portfolio_daily_returns.empty or benchmark_daily_returns.empty:
        return np.nan
    
    # Allinea le serie di rendimenti sull'indice comune (date)
    # Questo è fondamentale per un calcolo corretto
    aligned_df = pd.DataFrame({
        'portfolio': portfolio_daily_returns,
        'benchmark': benchmark_daily_returns
    }).dropna() # Rimuovi date dove uno dei due non ha rendimenti

    if len(aligned_df) < 2: # Necessari almeno due punti per la deviazione standard
        return np.nan

    difference_in_returns = aligned_df['portfolio'] - aligned_df['benchmark']
    tracking_error_annualized = difference_in_returns.std() * np.sqrt(trading_days_per_year)
    
    return tracking_error_annualized * 100 # In percentuale
# Le funzioni per Rolling Metrics e calculate_annual_returns sono state rimosse
# per questa versione semplificata.
