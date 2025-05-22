# utils/performance.py
import pandas as pd
import numpy as np
from datetime import datetime # Aggiunto per type hinting se necessario
from dateutil.relativedelta import relativedelta

# ... (try-except per pyxirr e empyrical come prima) ...
try:
    from pyxirr import xirr as pyxirr_xirr 
except ImportError:
    print("ATTENZIONE: Libreria pyxirr non trovata. Il calcolo XIRR non sarà disponibile.")
    def pyxirr_xirr(dates, values, guess=None):
        return np.nan

try:
    import empyrical
except ImportError:
    print("ATTENZIONE: Libreria empyrical non trovata. Sortino Ratio e altre metriche da empyrical non saranno disponibili.")
    empyrical = None


# >>> INIZIO FUNZIONI CHE POTREBBERO ESSERE MANCANTI O COMMENTATE <<<
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
        return 1/365.25 if len(pac_df) >=1 else 0.0
        
    return duration_days / 365.25

def calculate_cagr(final_value: float, total_invested: float, num_years: float) -> float:
    """
    Calcola il Compound Annual Growth Rate (CAGR).
    """
    if total_invested <= 0 or num_years <= 0:
        return np.nan
    if final_value < 0 and total_invested > 0 :
        return np.nan 
    if final_value == 0 and total_invested > 0:
        return -100.0

    try:
        cagr = ((final_value / total_invested) ** (1 / num_years)) - 1
    except (ValueError, OverflowError, ZeroDivisionError):
        return np.nan
    return cagr * 100

def calculate_portfolio_returns(pac_df: pd.DataFrame) -> pd.Series:
    """
    Calcola i rendimenti giornalieri del portafoglio.
    """
    if pac_df.empty or 'PortfolioValue' not in pac_df.columns or len(pac_df) < 2:
        return pd.Series(dtype=float)

    df = pac_df.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
    
    if not isinstance(df.index, pd.DatetimeIndex):
        print("Errore: L'indice del DataFrame non è DatetimeIndex per calculate_portfolio_returns.")
        return pd.Series(dtype=float)
        
    df = df.sort_index()
    returns = df['PortfolioValue'].pct_change().dropna()
    return returns.replace([np.inf, -np.inf], np.nan).dropna()
# >>> FINE FUNZIONI CHE POTREBBERO ESSERE MANCANTI O COMMENTATE <<<


# --- FUNZIONI PER METRICHE AVANZATE (Volatilità, Sharpe, MDD, XIRR, Sortino) ---
# Assicurati che queste siano presenti e corrette come da nostra ultima versione:

def calculate_annualized_volatility(daily_returns: pd.Series, trading_days_per_year: int = 252) -> float:
    # ... (codice come definito precedentemente) ...
    if daily_returns.empty or len(daily_returns) < 2 or daily_returns.isnull().all():
        return np.nan
    volatility = daily_returns.std() * np.sqrt(trading_days_per_year)
    return volatility * 100

def calculate_sharpe_ratio(daily_returns: pd.Series, risk_free_rate_annual: float = 0.0, trading_days_per_year: int = 252) -> float:
    # ... (codice come definito precedentemente) ...
    if daily_returns.empty or len(daily_returns) < 2 or daily_returns.isnull().all():
        return np.nan
    daily_risk_free_rate = (1 + risk_free_rate_annual)**(1/trading_days_per_year) - 1
    excess_returns = daily_returns - daily_risk_free_rate
    mean_excess_return = excess_returns.mean()
    std_dev_excess_return = excess_returns.std()
    if std_dev_excess_return == 0 or pd.isna(std_dev_excess_return):
        return np.nan if mean_excess_return == 0 else (np.inf if mean_excess_return > 0 else -np.inf)
    sharpe_ratio_daily = mean_excess_return / std_dev_excess_return
    sharpe_ratio_annualized = sharpe_ratio_daily * np.sqrt(trading_days_per_year)
    return sharpe_ratio_annualized

def calculate_drawdown_series(portfolio_values: pd.Series) -> pd.Series:
    # ... (codice come definito precedentemente) ...
    if not isinstance(portfolio_values, pd.Series) or portfolio_values.empty or len(portfolio_values) < 2:
        return pd.Series(dtype=float)
    if not isinstance(portfolio_values.index, pd.DatetimeIndex):
        # Tenta conversione se l'indice è 'Date'
        if hasattr(portfolio_values.index, 'name') and (portfolio_values.index.name == 'Date' or portfolio_values.index.name is None and isinstance(portfolio_values.index, pd.RangeIndex) == False):
             try:
                 # Se portfolio_values è una Series il cui indice non è DatetimeIndex ma potrebbe essere date
                 # Questo caso è meno probabile se upstream è corretto, ma per sicurezza
                 if isinstance(portfolio_values.index, pd.RangeIndex) and 'Date' in portfolio_values.name: # Questo non ha senso
                     pass # L'indice di una serie non ha 'Date' nel nome così
                 portfolio_values_temp = portfolio_values.copy()
                 portfolio_values_temp.index = pd.to_datetime(portfolio_values_temp.index)
                 if isinstance(portfolio_values_temp.index, pd.DatetimeIndex):
                     portfolio_values = portfolio_values_temp
                 else:
                    print("Errore (Drawdown Series): Conversione indice a DatetimeIndex fallita.")
                    return pd.Series(dtype=float, index=portfolio_values.index)
             except Exception as e:
                print(f"Errore (Drawdown Series): Eccezione durante conversione indice: {e}")
                return pd.Series(dtype=float, index=portfolio_values.index)
        else:
            print("Errore (Drawdown Series): L'indice della serie dei valori del portafoglio deve essere DatetimeIndex.")
            return pd.Series(dtype=float, index=portfolio_values.index)

    portfolio_values = portfolio_values.sort_index()
    if portfolio_values.isnull().all() or len(portfolio_values.dropna()) < 2:
        return pd.Series(dtype=float, index=portfolio_values.index)
    cumulative_max = portfolio_values.cummax()
    drawdown = pd.Series(index=portfolio_values.index, dtype=float)
    non_zero_peak_mask = cumulative_max != 0
    drawdown[non_zero_peak_mask] = (portfolio_values[non_zero_peak_mask] - cumulative_max[non_zero_peak_mask]) / cumulative_max[non_zero_peak_mask]
    drawdown[cumulative_max == 0] = 0.0
    return drawdown * 100


def calculate_max_drawdown(pac_df: pd.DataFrame) -> float:
    # ... (codice come definito precedentemente, che ora usa calculate_drawdown_series) ...
    if pac_df.empty or 'PortfolioValue' not in pac_df.columns or len(pac_df) < 2:
        return np.nan
    df_for_drawdown = pac_df.copy()
    if 'Date' in df_for_drawdown.columns:
        df_for_drawdown['Date'] = pd.to_datetime(df_for_drawdown['Date'])
        df_for_drawdown = df_for_drawdown.set_index('Date')
    if not isinstance(df_for_drawdown.index, pd.DatetimeIndex):
        print("Errore (Max Drawdown): L'indice non è DatetimeIndex.")
        return np.nan
    portfolio_values_series = df_for_drawdown['PortfolioValue']
    if portfolio_values_series.isnull().all() or len(portfolio_values_series.dropna()) < 2:
        return np.nan
    drawdown_series = calculate_drawdown_series(portfolio_values_series)
    if drawdown_series.empty or drawdown_series.isnull().all():
        return np.nan
    max_drawdown_value = drawdown_series.min()
    return max_drawdown_value if pd.notna(max_drawdown_value) else np.nan


def generate_cash_flows_for_xirr(
    pac_df: pd.DataFrame, 
    start_date_pac_str: str,
    duration_months: int, 
    monthly_investment: float, 
    final_portfolio_value: float
) -> tuple[list, list]:
    # ... (codice come definito precedentemente) ...
    dates_outflows = []
    values_outflows = []
    if pac_df.empty or 'Date' not in pac_df.columns:
        return [], []
    try:
        actual_pac_start_date = pd.to_datetime(start_date_pac_str)
        if actual_pac_start_date.tzinfo is not None:
            actual_pac_start_date = actual_pac_start_date.tz_localize(None)
    except Exception as e:
        print(f"Errore nella conversione di start_date_pac_str: {e}")
        actual_pac_start_date = pd.to_datetime(pac_df['Date'].iloc[0])
    for i in range(duration_months):
        investment_date = actual_pac_start_date + relativedelta(months=i)
        dates_outflows.append(investment_date)
        values_outflows.append(-monthly_investment)
    final_date = pd.to_datetime(pac_df['Date'].iloc[-1])
    if final_date.tzinfo is not None:
        final_date = final_date.tz_localize(None)
    dates_outflows.append(final_date)
    values_outflows.append(final_portfolio_value)
    return dates_outflows, values_outflows

def calculate_xirr_metric(dates: list, values: list) -> float:
    # ... (codice come definito precedentemente) ...
    if 'pyxirr_xirr' not in globals() or not callable(globals()['pyxirr_xirr']):
        print("INFO: Funzione pyxirr_xirr non disponibile.")
        return np.nan
    if not dates or not values or len(dates) != len(values):
        print("INFO: Date o valori mancanti o di lunghezza diversa per XIRR.")
        return np.nan
    has_positive = any(v > 0 for v in values)
    has_negative = any(v < 0 for v in values)
    if not (has_positive and has_negative):
        print("INFO: XIRR non calcolabile, mancano flussi positivi o negativi.")
        return np.nan
    python_dates = [d.date() if isinstance(d, pd.Timestamp) else d for d in dates]
    try:
        rate = pyxirr_xirr(python_dates, values)
        return rate * 100 if rate is not None and pd.notna(rate) else np.nan
    except Exception as e:
        print(f"Errore nel calcolo XIRR con pyxirr: {e}")
        return np.nan
# utils/performance.py
# ... (altre importazioni e funzioni) ...

# in utils/performance.py

def calculate_annual_returns(portfolio_values_series: pd.Series) -> pd.Series:
    """
    Calcola i rendimenti per ogni anno civile coperto dalla serie di valori di portafoglio.
    L'input portfolio_values_series deve avere un DatetimeIndex.
    """
    if not isinstance(portfolio_values_series, pd.Series) or portfolio_values_series.empty or \
       not isinstance(portfolio_values_series.index, pd.DatetimeIndex) or len(portfolio_values_series.dropna()) < 2:
        return pd.Series(dtype=float, name="Rendimento Annuale (%)")

    # Assicura che la serie sia ordinata e senza duplicati nell'indice che potrebbero causare problemi
    portfolio_values_series = portfolio_values_series.sort_index()
    portfolio_values_series = portfolio_values_series[~portfolio_values_series.index.duplicated(keep='last')]


    # Trova tutti gli anni unici presenti nei dati
    years = sorted(list(set(portfolio_values_series.index.year)))
    annual_returns_list = []

    if not years:
        return pd.Series(dtype=float, name="Rendimento Annuale (%)")

    for year in years:
        # Valore all'inizio dell'anno (o il primo valore disponibile nell'anno)
        start_of_year_values = portfolio_values_series[portfolio_values_series.index.year == year]
        if start_of_year_values.empty:
            continue # Salta l'anno se non ci sono dati

        # Se è il primo anno della serie, il "valore iniziale" è il primo valore della serie in quell'anno
        # Altrimenti, è il valore di fine dell'anno precedente.
        
        # Valore di fine anno precedente (per calcolare il rendimento dell'anno corrente)
        # Se è il primo anno in assoluto della serie, il valore iniziale è il primo valore della serie.
        if year == years[0]: # Primo anno della serie
            # Troviamo il primo valore assoluto della serie originale per iniziare
            # Se l'anno inizia a metà, il rendimento è da quel punto a fine anno.
            first_value_in_series_for_year = start_of_year_values.iloc[0]
            
            # Se l'anno di inizio non è completo, potremmo voler calcolare il rendimento
            # dal primo giorno di investimento fino alla fine di quell'anno.
            # Oppure, se vogliamo rendimenti per anni civili completi, potremmo saltare il primo anno parziale.
            # Per ora, calcoliamo il rendimento dal primo valore dell'anno alla fine dell'anno.
            # Il "valore iniziale" per il rendimento di questo primo anno è il primo valore in quest'anno.
            initial_value_for_year_calc = first_value_in_series_for_year
        else:
            # Valore di fine anno precedente
            end_of_previous_year_series = portfolio_values_series[portfolio_values_series.index.year == year - 1]
            if not end_of_previous_year_series.empty:
                initial_value_for_year_calc = end_of_previous_year_series.iloc[-1]
            else:
                # Non ci sono dati per l'anno precedente, non possiamo calcolare il rendimento per questo anno
                # a meno che non sia il primo anno assoluto (già gestito)
                # Oppure potremmo prendere il primo valore di quest'anno se è un nuovo inizio.
                # Per ora, se manca l'anno precedente (e non è il primo anno), lo saltiamo.
                # Questo approccio è per rendimenti anno su anno.
                 annual_returns_list.append({'Year': year, 'Return': np.nan})
                 continue


        # Valore alla fine dell'anno corrente (o l'ultimo valore disponibile nell'anno)
        final_value_for_year_calc = start_of_year_values.iloc[-1]

        if pd.notna(initial_value_for_year_calc) and pd.notna(final_value_for_year_calc) and initial_value_for_year_calc != 0:
            # Se è il primo anno e inizia a metà, questo è il rendimento per il periodo rimanente dell'anno.
            # Se vogliamo annualizzarlo, dovremmo considerare la frazione d'anno.
            # Per un istogramma, di solito si mostrano i rendimenti effettivi del periodo (anno civile o parziale).
            # La funzione CAGR/XIRR gestisce meglio l'annualizzazione su periodi diversi.
            # Qui calcoliamo il rendimento semplice per l'anno (o la porzione di esso).
            
            # Se è il primo anno in assoluto e non inizia il 1/1, il rendimento sarà "dal lancio a fine anno".
            # Se è un anno intermedio, sarà il rendimento dell'anno civile.
            # Se è l'ultimo anno e non finisce il 31/12, sarà "da inizio anno alla data finale".
            
            # Per un istogramma, i rendimenti non annualizzati per ogni periodo (anno) sono spesso usati.
            year_return = (final_value_for_year_calc / initial_value_for_year_calc) - 1
            annual_returns_list.append({'Year': year, 'Return': year_return * 100})
        else:
            annual_returns_list.append({'Year': year, 'Return': np.nan}) # Non calcolabile

    if not annual_returns_list:
        return pd.Series(dtype=float, name="Rendimento Annuale (%)")

    returns_df = pd.DataFrame(annual_returns_list).set_index('Year')['Return']
    returns_df.name = "Rendimento Annuale (%)"
    return returns_df

# simulatore_pac/utils/performance.py
# ... (tutte le importazioni e le funzioni esistenti rimangono invariate) ...

# --- NUOVE FUNZIONI PER ROLLING METRICS ---

def calculate_rolling_volatility(
    daily_returns: pd.Series, 
    window_days: int, 
    trading_days_per_year: int = 252
) -> pd.Series:
    """
    Calcola la volatilità annualizzata mobile.
    window_days: la dimensione della finestra in giorni di trading.
    """
    if daily_returns.empty or len(daily_returns) < window_days:
        return pd.Series(dtype=float, name="Rolling Volatility")
    
    rolling_vol = daily_returns.rolling(window=window_days).std() * np.sqrt(trading_days_per_year)
    return rolling_vol.dropna() * 100 # In percentuale

def _calculate_sharpe_for_window(
    window_returns: pd.Series, 
    risk_free_rate_annual: float, 
    trading_days_per_year: int
) -> float:
    """Funzione helper per calcolare Sharpe su una singola finestra."""
    if window_returns.empty or len(window_returns) < 2: # Richiede almeno 2 punti per std
        return np.nan
    
    daily_risk_free_rate = (1 + risk_free_rate_annual)**(1/trading_days_per_year) - 1
    excess_returns = window_returns - daily_risk_free_rate
    mean_excess_return = excess_returns.mean()
    std_dev_excess_return = excess_returns.std()
    
    if std_dev_excess_return == 0 or pd.isna(std_dev_excess_return):
        return np.nan if mean_excess_return == 0 else (np.inf if mean_excess_return > 0 else -np.inf)
        
    sharpe_ratio_window_daily = mean_excess_return / std_dev_excess_return
    # Annualizza lo Sharpe Ratio della finestra
    # Dato che stiamo già lavorando su una finestra, l'annualizzazione avviene moltiplicando per sqrt(trading_days_per_year)
    # Tuttavia, se la media e std sono già "per finestra", alcuni annualizzano solo lo std per portarlo a scala annuale
    # o la media. La formula più comune è (Mean(Ret_excess_daily) / Std(Ret_excess_daily)) * sqrt(trading_days_per_year)
    # In questo caso, stiamo calcolando lo Sharpe per la finestra e poi lo annualizziamo.
    # Se la finestra è di un anno, lo Sharpe è già annualizzato. Se è meno, va scalato.
    # Se la finestra è `window_days`, i rendimenti sono `window_days` rendimenti giornalieri.
    # Lo Sharpe giornaliero calcolato sulla finestra viene poi annualizzato.
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
    if daily_returns.empty or len(daily_returns) < window_days:
        return pd.Series(dtype=float, name="Rolling Sharpe Ratio")
    
    rolling_sharpe = daily_returns.rolling(window=window_days).apply(
        lambda x: _calculate_sharpe_for_window(x, risk_free_rate_annual, trading_days_per_year),
        raw=False # raw=False perché la nostra funzione si aspetta una Serie Pandas
    )
    return rolling_sharpe.dropna()

def _calculate_cagr_for_window(
    window_portfolio_values: pd.Series, 
    window_days_actual: int, # Lunghezza effettiva della finestra (può essere < window_days all'inizio)
    trading_days_per_year: int
) -> float:
    """Funzione helper per calcolare CAGR su una singola finestra di valori di portafoglio."""
    if window_portfolio_values.empty or len(window_portfolio_values) < 2 or window_days_actual < 2:
        return np.nan
    
    start_value = window_portfolio_values.iloc[0]
    end_value = window_portfolio_values.iloc[-1]
    
    if start_value <= 0 or pd.isna(start_value) or pd.isna(end_value):
        return np.nan
    if end_value < 0 and start_value > 0: # Perdita totale o più
         return np.nan # CAGR per valori finali negativi è complesso
    if end_value == 0 and start_value > 0:
        return -100.0

    num_years_in_window = window_days_actual / trading_days_per_year
    if num_years_in_window <= 0:
        return np.nan

    try:
        cagr = ((end_value / start_value) ** (1 / num_years_in_window)) - 1
    except (ValueError, OverflowError, ZeroDivisionError):
        return np.nan
    return cagr * 100 # In percentuale

def calculate_rolling_cagr(
    portfolio_values: pd.Series, # Richiede la serie dei valori del portafoglio, non i rendimenti
    window_days: int, 
    trading_days_per_year: int = 252
) -> pd.Series:
    """
    Calcola il CAGR mobile.
    portfolio_values: Serie Pandas con i valori del portafoglio e DatetimeIndex.
    """
    if not isinstance(portfolio_values, pd.Series) or portfolio_values.empty or \
       not isinstance(portfolio_values.index, pd.DatetimeIndex) or len(portfolio_values) < window_days :
        return pd.Series(dtype=float, name="Rolling CAGR")

    # raw=False perché _calculate_cagr_for_window si aspetta una Serie Pandas
    # min_periods=window_days assicura che la funzione venga applicata solo a finestre complete
    rolling_cagr = portfolio_values.rolling(window=window_days, min_periods=window_days).apply( 
        lambda x: _calculate_cagr_for_window(x, window_days, trading_days_per_year),
        raw=False 
    )
    return rolling_cagr.dropna()
# Commentato per ora, dato che empyrical dava problemi di installazione
# def calculate_sortino_ratio_empyrical(daily_returns: pd.Series, required_return_annual: float = 0.0) -> float:
#     # ... (codice come definito precedentemente) ...
#     if empyrical is None:
#         print("INFO: Libreria empyrical non disponibile. Sortino Ratio non calcolabile.")
#         return np.nan
#     # ... (resto della funzione)
#     return np.nan # Placeholder se empyrical non è disponibile o commentato
