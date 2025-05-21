# simulatore_pac/utils/performance.py

import pandas as pd
import numpy as np

def get_total_capital_invested(pac_df: pd.DataFrame) -> float:
    """
    Calcola il capitale totale investito alla fine del periodo del PAC.

    Args:
        pac_df (pd.DataFrame): DataFrame con l'evoluzione del PAC,
                               deve contenere la colonna 'InvestedCapital'.

    Returns:
        float: Il capitale totale investito. 0.0 se il DataFrame è vuoto o la colonna manca.
    """
    if pac_df.empty or 'InvestedCapital' not in pac_df.columns:
        return 0.0
    return pac_df['InvestedCapital'].iloc[-1]

def get_final_portfolio_value(pac_df: pd.DataFrame) -> float:
    """
    Calcola il valore finale del portafoglio alla fine del periodo del PAC.

    Args:
        pac_df (pd.DataFrame): DataFrame con l'evoluzione del PAC,
                               deve contenere la colonna 'PortfolioValue'.

    Returns:
        float: Il valore finale del portafoglio. 0.0 se il DataFrame è vuoto o la colonna manca.
    """
    if pac_df.empty or 'PortfolioValue' not in pac_df.columns:
        return 0.0
    return pac_df['PortfolioValue'].iloc[-1]

def calculate_total_return_percentage(final_value: float, total_invested: float) -> float:
    """
    Calcola il rendimento totale percentuale.

    Args:
        final_value (float): Valore finale del portafoglio.
        total_invested (float): Capitale totale investito.

    Returns:
        float: Il rendimento totale in percentuale. 0.0 se total_invested è 0 o negativo.
    """
    if total_invested <= 0:
        return 0.0
    return ((final_value / total_invested) - 1) * 100

def calculate_cagr(final_value: float, initial_value_or_total_invested: float, num_years: float) -> float:
    """
    Calcola il Compound Annual Growth Rate (CAGR).

    Args:
        final_value (float): Valore finale del portafoglio.
        initial_value_or_total_invested (float): Valore iniziale o, per un PAC,
                                                 il capitale totale investito può essere usato come proxy
                                                 se si considera il rendimento sul capitale medio impiegato,
                                                 ma un calcolo più preciso dell'IRR è spesso migliore per PAC.
                                                 Per un CAGR semplice sull'investimento totale, si usa total_invested.
        num_years (float): Il numero di anni su cui è stato calcolato il rendimento.

    Returns:
        float: Il CAGR in percentuale. np.nan se initial_value è 0/negativo o num_years è 0.
    """
    if initial_value_or_total_invested <= 0 or num_years <= 0:
        return np.nan # o 0.0 a seconda di come si vuole gestire il caso limite
    
    cagr = ((final_value / initial_value_or_total_invested) ** (1 / num_years)) - 1
    return cagr * 100

def get_duration_years(pac_df: pd.DataFrame) -> float:
    """
    Calcola la durata totale della simulazione PAC in anni.

    Args:
        pac_df (pd.DataFrame): DataFrame con l'evoluzione del PAC,
                               deve avere una colonna 'Date' di tipo datetime.

    Returns:
        float: La durata in anni. 0.0 se il DataFrame è vuoto o ha meno di due punti.
    """
    if pac_df.empty or 'Date' not in pac_df.columns or len(pac_df) < 2:
        return 0.0
    
    # Assicura che 'Date' sia datetime
    if not pd.api.types.is_datetime64_any_dtype(pac_df['Date']):
        try:
            dates = pd.to_datetime(pac_df['Date'])
        except Exception:
            return 0.0 # Non può convertire le date
    else:
        dates = pac_df['Date']

    start_date = dates.iloc[0]
    end_date = dates.iloc[-1]
    duration_days = (end_date - start_date).days
    
    if duration_days <= 0: # Può accadere se start e end sono lo stesso giorno o ordine errato
        # Se c'è solo un investimento o la durata è inferiore a un giorno,
        # consideriamo una frazione molto piccola di anno per evitare divisione per zero nel CAGR,
        # ma un CAGR su un periodo così breve non è molto significativo.
        # Per semplicità, se i giorni sono 0, restituiamo un valore piccolo o gestiamo come caso limite nel CAGR.
        return 1/365.25 if len(pac_df) >=1 else 0.0 # Ad esempio, 1 giorno
        
    return duration_days / 365.25 # Media giorni in un anno
