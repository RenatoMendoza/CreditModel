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
         
    Returns:
        DataFrame original con columnas adicionales:
            - history_score: puntaje para historial crediticio
            - delay_score: puntaje para retrasos
            - inquiries_score: puntaje para consultas
            - debt_score: puntaje para deuda pendiente
            - mix_score: puntaje para mezcla de crédito
            - total_score: puntaje combinado (0-100)
            - fico_score: puntaje escalado (300-850)
            - credit_category: categoría (0=malo, 1=bueno)
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

    
    # Calcular puntaje total incluyendo el nuevo componente de ingresos
    # Ajustamos los pesos para incluir el nuevo factor
    result_df['total_score'] = (
        result_df['history_score'] * 0.15 +     # 15% Historial
        result_df['delay_score'] * 0.35 +       # 35% Retrasos
        result_df['inquiries_score'] * 0.15 +   # 15% Consultas
        result_df['debt_score'] * 0.15 +        # 15% Deuda
        result_df['mix_score'] * 0.20           # 20% Mezcla
    )
    
    # Convertir a escala típica de FICO (300-850)
    result_df['fico_score'] = 300 + (result_df['total_score'] / 100) * 550
    
    # Clasificar en categorías (1=malo, 2=estándar, 3=bueno)
    result_df['credit_category'] = pd.cut(
        result_df['fico_score'], 
        bins=[0, 590, 850],
        labels=[0, 1]
    )
    
    return result_df

def clean_data_trad(file_name):
    df = pd.read_csv(file_name)

    credit_score_mapping = {
    'Bad': 0,
    'Standard': 1,
    'Good': 2
    }

    # Convert categorical credit_score to numerical
    df['credit_mix'] = df['credit_mix'].map(credit_score_mapping)

    avg_credit_history = df.groupby('customer_id')['credit_history_age'].mean().round().astype(int).reset_index()['credit_history_age'].replace(2, 1)
    avg_delay = df.groupby('customer_id')['delay_from_due_date'].mean().round().astype(int).reset_index()['delay_from_due_date']
    avg_num_inquires = df.groupby('customer_id')['num_credit_inquiries'].mean().round().astype(int).reset_index()['num_credit_inquiries']
    avg_credit_score = df.groupby('customer_id')['credit_score'].mean().round().astype(int).reset_index()['credit_score'].replace(2, 1)
    avg_credit_mix = df.groupby('customer_id')['credit_mix'].first().reset_index()['credit_mix']
    avg_outstanding_debt = df.groupby('customer_id')['outstanding_debt'].mean().round().astype(int).reset_index()['outstanding_debt']

    df_unique = df.drop_duplicates(subset=['customer_id'])

    unique_id = df_unique['customer_id'].sort_values().reset_index(drop=True)

    clean_data = pd.DataFrame({
    'customer_id': unique_id,
    'avg_credit_history': avg_credit_history,
    'avg_delay': avg_delay,
    'avg_num_inquires': avg_num_inquires,
    'avg_outstanding_debt': avg_outstanding_debt,
    'avg_credit_mix': avg_credit_mix,
    'avg_credit_score': avg_credit_score
    })

    return clean_data

def evaluate_credit_model(y_true, y_pred, class_names=['Bad', 'Good']):
    """
    Evalúa el rendimiento de un modelo de clasificación crediticia.
    
    Parámetros:
    -----------
    y_true : array-like
        Valores reales de las clases.
    y_pred : array-like
        Valores predichos por el modelo.
    class_names : list, opcional (default=['Bad', 'Good'])
        Nombres de las clases para mostrar en los resultados.
        
    Retorna:
    --------
    dict
        Diccionario con todas las métricas de evaluación.
    """
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    from sklearn.metrics import precision_score, recall_score, f1_score, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    
    # Crear y mostrar matriz de confusión
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap='Blues', ax=ax)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
    
    # Calcular métricas básicas
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Calcular métricas para cada clase
    precision_per_class = precision_score(y_true, y_pred, average=None, 
                                         labels=list(range(len(class_names))), 
                                         zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, 
                                   labels=list(range(len(class_names))), 
                                   zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, 
                           labels=list(range(len(class_names))), 
                           zero_division=0)
    
    # Imprimir resultados
    print(f"Accuracy: {accuracy:.4f}")
    print("\nMacro-averaged metrics:")
    print(f"Precision: {precision_macro:.4f}")
    print(f"Recall: {recall_macro:.4f}")
    print(f"F1-score: {f1_macro:.4f}")
    
    print("\nPer-class metrics:")
    for i, class_name in enumerate(class_names):
        print(f"\n{class_name} Credit Score:")
        print(f"  Precision: {precision_per_class[i]:.4f}")
        print(f"  Recall: {recall_per_class[i]:.4f}")
        print(f"  F1-score: {f1_per_class[i]:.4f}")
    
    # Reporte detallado de clasificación
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Organizar todas las métricas en un diccionario para retornar
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_per_class': {class_name: precision_per_class[i] for i, class_name in enumerate(class_names)},
        'recall_per_class': {class_name: recall_per_class[i] for i, class_name in enumerate(class_names)},
        'f1_per_class': {class_name: f1_per_class[i] for i, class_name in enumerate(class_names)},
        'confusion_matrix': cm,
        'classification_report': classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    }
    
    return 