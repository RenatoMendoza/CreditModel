import pandas as pd
import numpy as np

def calculate_traditional_credit_score_binary(df):
    """
    Calcula el credit score tradicional para todo un DataFrame.
    
    Args:
        df: DataFrame con las columnas:
            - avg_credit_history
            - avg_delay
            - avg_num_inquires
            - avg_outstanding_debt
            - avg_credit_mix
            - avg_inhand_income (nueva columna incluida)
         
    Returns:
        DataFrame original con columnas adicionales:
            - history_score: puntaje para historial crediticio
            - delay_score: puntaje para retrasos
            - inquiries_score: puntaje para consultas
            - debt_score: puntaje para deuda pendiente
            - mix_score: puntaje para mezcla de crédito
            - income_score: puntaje para ingresos disponibles
            - total_score: puntaje combinado (0-100)
            - fico_score: puntaje escalado (300-850)
            - credit_category: categoría (1=malo, 2=estándar, 3=bueno)
    """
    # Crear una copia para no modificar el original
    result_df = df.copy()
    
    # Calcular componente de historial crediticio (15% del FICO tradicional)
    result_df['history_score'] = result_df['avg_credit_history'].apply(
        lambda months: 20 if months < 12 else
                     40 if months < 24 else
                     60 if months < 48 else
                     80 if months < 96 else 100
    )
    
    # Calcular componente de retraso en pagos (35% del FICO tradicional)
    result_df['delay_score'] = result_df['avg_delay'].apply(
        lambda delay: 100 if delay <= 0 else
                     90 if delay < 5 else
                     70 if delay < 15 else
                     40 if delay < 30 else 10
    )
    
    # Calcular componente de número de consultas (10% del FICO tradicional)
    result_df['inquiries_score'] = result_df['avg_num_inquires'].apply(
        lambda inquiries: 100 if inquiries == 0 else
                         90 if inquiries <= 2 else
                         70 if inquiries <= 5 else
                         40 if inquiries <= 10 else 10
    )
    
    # Calcular componente de deuda pendiente (30% del FICO tradicional)
    result_df['debt_score'] = result_df['avg_outstanding_debt'].apply(
        lambda debt: 100 if debt == 0 else
                    90 if debt < 500 else
                    75 if debt < 1500 else
                    50 if debt < 3000 else 20
    )
    
    # Calcular componente de mezcla de crédito (10% del FICO tradicional)
    result_df['mix_score'] = result_df['avg_credit_mix'].apply(
        lambda mix: 30 if mix == 0 else
                   65 if mix == 1 else 100
    )
    
    # Calcular componente de ingresos disponibles (nuevo)
    result_df['income_score'] = result_df['avg_inhand_income'].apply(
        lambda income: 20 if income < 1000 else
                      40 if income < 2000 else
                      60 if income < 4000 else
                      80 if income < 10000 else 100
    )
    
    # Calcular puntaje total incluyendo el nuevo componente de ingresos
    # Ajustamos los pesos para incluir el nuevo factor
    result_df['total_score'] = (
        result_df['history_score'] * 0.10 +     # 10% Historial
        result_df['delay_score'] * 0.30 +       # 30% Retrasos
        result_df['inquiries_score'] * 0.10 +   # 10% Consultas
        result_df['debt_score'] * 0.15 +        # 20% Deuda
        result_df['mix_score'] * 0.20 +         # 15% Mezcla
        result_df['income_score'] * 0.15        # 15% Ingresos
    )
    
    # Convertir a escala típica de FICO (300-850)
    result_df['fico_score'] = 300 + (result_df['total_score'] / 100) * 550
    
    # Clasificar en categorías (1=malo, 2=estándar, 3=bueno)
    result_df['credit_category'] = pd.cut(
        result_df['fico_score'], 
        bins=[0, 580, 850],
        labels=[0, 1]
    )
    
    return result_df