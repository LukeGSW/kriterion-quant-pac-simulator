# simulatore_pac/utils/performance.py

import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

# ... (codice esistente per pyxirr e empyrical try-except blocks) ...
try:
    from pyxirr import xirr as pyxirr_xirr
except ImportError:
    print("ATTENZIONE: Libreria pyxirr non trovata. Il calcolo XIRR non sarà disponibile.")
    def pyxirr_xirr(dates, values, guess=None): # Funzione placeholder
        return np.nan

try:
    import empyrical
except ImportError:
    print("ATTENZIONE: Libreria empyrical non trovata. Sortino Ratio e altre metriche da empyrical non saranno disponibili.")
    empyrical = None # Placeholder

# ... (tutte le funzioni esistenti come get_total_capital_invested, ..., calculate_xirr_metric, calculate_sortino_ratio_empyrical RESTANO INVARIATE) ...

# NUOVA FUNZIONE PER LA SERIE STORICA DEL DRAWDOWN
def calculate_drawdown_series(portfolio_values: pd.Series) -> pd.Series:
    """
    Calcola la serie storica del drawdown percentuale.

    Args:
        portfolio_values (pd.Series): Serie Pandas con i valori del portafoglio,
                                      con un DatetimeIndex.

    Returns:
        pd.Series: Serie Pandas con i valori del drawdown percentuale (es. -0.1 per -10%).
                   Restituisce una serie vuota se l'input non è valido.
    """
    if not isinstance(portfolio_values, pd.Series) or portfolio_values.empty or len(portfolio_values) < 2:
        return pd.Series(dtype=float)
    
    if not isinstance(portfolio_values.index, pd.DatetimeIndex):
        print("Errore (Drawdown Series): L'indice della serie dei valori del portafoglio deve essere DatetimeIndex.")
        # Tenta la conversione se l'indice è 'Date'
        if 'Date' in portfolio_values.index.name.title() or (hasattr(portfolio_values.index, 'name') and portfolio_values.index.name == 'Date'):
             try:
                 portfolio_values.index = pd.to_datetime(portfolio_values.index)
             except:
                 return pd.Series(dtype=float) # Conversione fallita
        else: # se l'indice non è data e non è convertibile facilmente
             return pd.Series(dtype=float)


    # Assicura che la serie sia ordinata per data
    portfolio_values = portfolio_values.sort_index()

    if portfolio_values.isnull().all() or len(portfolio_values.dropna()) < 2:
        return pd.Series(dtype=float, index=portfolio_values.index) # Restituisce serie di NaN con stesso indice

    cumulative_max = portfolio_values.cummax()
    # Calcola il drawdown. Se cumulative_max è 0 (all'inizio se il portafoglio parte da 0),
    # il risultato della divisione potrebbe essere -inf o NaN.
    # Sostituisci cumulative_max con NaN dove è 0 per evitare divisione per zero che dia inf.
    # Oppure, gestisci il caso in cui il portafoglio inizia da 0.
    # Se il valore del portafoglio è sempre >= 0, cumulative_max sarà >= 0.
    # Il problema sorge se cumulative_max è 0.
    
    drawdown = pd.Series(index=portfolio_values.index, dtype=float) # Inizializza con NaN
    
    # Evita divisione per zero se cumulative_max è 0
    # Se cumulative_max è 0, il drawdown è 0 (nessun picco da cui calare se il valore è sempre stato 0)
    # o NaN se il valore del portafoglio è anch'esso 0 in quel punto.
    non_zero_peak_mask = cumulative_max != 0
    drawdown[non_zero_peak_mask] = (portfolio_values[non_zero_peak_mask] - cumulative_max[non_zero_peak_mask]) / cumulative_max[non_zero_peak_mask]
    
    # Se il cumulative_max è 0 e il portfolio_value è 0, il drawdown è 0.
    # Se il cumulative_max è 0 e il portfolio_value è > 0 (impossibile per cummax),
    # o < 0 (anche questo non dovrebbe accadere con cummax), la situazione è anomala.
    # Per i casi in cui cumulative_max è 0 (tipicamente all'inizio se il portafoglio parte da 0),
    # il drawdown dovrebbe essere 0.
    drawdown[cumulative_max == 0] = 0.0
    
    return drawdown * 100 # In percentuale

# La funzione calculate_max_drawdown esistente può essere mantenuta,
# oppure potrebbe chiamare calculate_drawdown_series(pac_df['PortfolioValue']).min()
# Per coerenza, modifichiamo calculate_max_drawdown per usare la nuova funzione:

def calculate_max_drawdown(pac_df: pd.DataFrame) -> float:
    """
    Calcola il Max Drawdown (MDD) del valore del portafoglio.
    Restituisce il MDD come percentuale negativa (es. -25.0 per -25%).
    """
    if pac_df.empty or 'PortfolioValue' not in pac_df.columns or len(pac_df) < 2:
        return np.nan
        
    # Assicurati che 'Date' sia l'indice e sia DatetimeIndex
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

    drawdown_series = calculate_drawdown_series(portfolio_values_series) # Usa la nuova funzione
    
    if drawdown_series.empty or drawdown_series.isnull().all():
        return np.nan
        
    max_drawdown_value = drawdown_series.min()
    
    return max_drawdown_value if pd.notna(max_drawdown_value) else np.nan
