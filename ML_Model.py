import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_xgboost_credit_score_model(data, target_column='avg_credit_score', test_size=0.3, random_state=42, 
                                   save_model_path=None, verbose=True):
    """
    Entrena y evalúa un modelo XGBoost para clasificación de puntaje crediticio.
    
    Parámetros:
    -----------
    data : DataFrame
        DataFrame de pandas que contiene los datos.
    target_column : str, opcional (default='avg_credit_score')
        Nombre de la columna objetivo a predecir.
    test_size : float, opcional (default=0.3)
        Proporción de datos para el conjunto de prueba.
    random_state : int, opcional (default=42)
        Semilla para reproducibilidad.
    save_model_path : str, opcional (default=None)
        Ruta donde guardar el modelo. Si es None, no se guarda.
    verbose : bool, opcional (default=True)
        Si es True, muestra información detallada del proceso.
        
    Retorna:
    --------
    dict
        Diccionario con el modelo optimizado, métricas, y otra información relevante.
    """
   
    
    # Separar características (X) y variable objetivo (y)
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Verificar la distribución de clases
    if verbose:
        print("Distribución de clases en los datos de entrenamiento:")
        print(y_train.value_counts(normalize=True))
    
    # Paso 1: Entrenar un modelo XGBoost básico
    if verbose:
        print("\n--- Entrenando modelo XGBoost básico ---")
    
    model = xgb.XGBClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    
    # Evaluar el modelo en datos de prueba
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    if verbose:
        print(f"Precisión del modelo básico: {accuracy:.2f}")
        print("\nInforme de clasificación:")
        print(classification_report(y_test, y_pred))
    
        # Visualizar la matriz de confusión
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Clase 0', 'Clase 1'], 
                    yticklabels=['Clase 0', 'Clase 1'])
        plt.xlabel('Predicción')
        plt.ylabel('Valor Real')
        plt.title('Matriz de Confusión - Modelo Básico')
        plt.tight_layout()
        plt.show()
    
    # Paso 2: Optimizar hiperparámetros con búsqueda en cuadrícula
    if verbose:
        print("\n--- Optimizando hiperparámetros ---")
    
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    # Para ahorrar tiempo, se puede usar una búsqueda menos exhaustiva
    grid_search = GridSearchCV(
        estimator=xgb.XGBClassifier(random_state=random_state),
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        verbose=1 if verbose else 0
    )
    
    grid_search.fit(X_train, y_train)
    
    # Mejores parámetros encontrados
    if verbose:
        print(f"\nMejores parámetros: {grid_search.best_params_}")
        print(f"Mejor puntuación CV: {grid_search.best_score_:.2f}")
    
    # Paso 3: Entrenar el modelo con los mejores parámetros
    best_model = grid_search.best_estimator_
    y_pred_best = best_model.predict(X_test)
    accuracy_best = accuracy_score(y_test, y_pred_best)
    
    classification_rep = classification_report(y_test, y_pred_best, output_dict=True)
    
    if verbose:
        print(f"\nPrecisión del modelo optimizado: {accuracy_best:.2f}")
        print("\nInforme de clasificación del modelo optimizado:")
        print(classification_report(y_test, y_pred_best))
    
        # Visualizar la matriz de confusión para el modelo optimizado
        plt.figure(figsize=(8, 6))
        cm_best = confusion_matrix(y_test, y_pred_best)
        sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Clase 0', 'Clase 1'], 
                    yticklabels=['Clase 0', 'Clase 1'])
        plt.xlabel('Predicción')
        plt.ylabel('Valor Real')
        plt.title('Matriz de Confusión - Modelo Optimizado')
        plt.tight_layout()
        plt.show()
    
    # Paso 4: Analizar la importancia de las características
    if verbose:
        plt.figure(figsize=(12, 6))
        xgb.plot_importance(best_model, max_num_features=10)
        plt.title('Top 10 Características Más Importantes')
        plt.tight_layout()
        plt.show()
    
    # Obtener los valores de importancia
    feature_importance = best_model.get_booster().get_score(importance_type='weight')
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    if verbose:
        print("\nImportancia de características (peso):")
        for feature, importance in sorted_importance:
            print(f"{feature}: {importance}")
    
    # Paso 5: Guardar el modelo si se especifica una ruta
    if save_model_path:
        best_model.save_model(save_model_path)
        if verbose:
            print(f"\nModelo guardado como '{save_model_path}'")
    
    # Preparar el resultado a devolver
    results = {
        'model': best_model,
        'best_params': grid_search.best_params_,
        'accuracy': accuracy_best,
        'classification_report': classification_rep,
        'feature_importance': dict(sorted_importance),
        'confusion_matrix': confusion_matrix(y_test, y_pred_best).tolist(),
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred_best
    }
    
    return results


def clean_data_ml(file_name):
    df = pd.read_csv(file_name)

    credit_score_mapping = {
    'Bad': 0,
    'Standard': 1,
    'Good': 2
    }

    # Convert categorical credit_score to numerical
    df['credit_mix'] = df['credit_mix'].map(credit_score_mapping)

    avg_annual_income = df.groupby('customer_id')['annual_income'].mean().round().astype(int).reset_index()['annual_income']
    avg_monthly_inhand_salary = df.groupby('customer_id')['monthly_inhand_salary'].mean().round().astype(int).reset_index()['monthly_inhand_salary']
    avg_total_emi_per_month = df.groupby('customer_id')['total_emi_per_month'].mean().round().astype(int).reset_index()['total_emi_per_month']
    avg_num_bank_accounts = df.groupby('customer_id')['num_bank_accounts'].mean().round().astype(int).reset_index()['num_bank_accounts']
    avg_num_credit_card = df.groupby('customer_id')['num_credit_card'].mean().round().astype(int).reset_index()['num_credit_card']
    avg_num_loans = df.groupby('customer_id')['num_of_loan'].mean().round().astype(int).reset_index()['num_of_loan']
    avg_monthly_balance = df.groupby('customer_id')['monthly_balance'].mean().round().astype(int).reset_index()['monthly_balance']
    avg_credit_history = df.groupby('customer_id')['credit_history_age'].mean().round().astype(int).reset_index()['credit_history_age']
    avg_delay = df.groupby('customer_id')['delay_from_due_date'].mean().round().astype(int).reset_index()['delay_from_due_date']
    avg_num_inquires = df.groupby('customer_id')['num_credit_inquiries'].mean().round().astype(int).reset_index()['num_credit_inquiries']
    avg_credit_score = df.groupby('customer_id')['credit_score'].mean().round().astype(int).reset_index()['credit_score']
    avg_credit_score = avg_credit_score.replace(2, 1)
    
    avg_credit_mix = df.groupby('customer_id')['credit_mix'].first().reset_index()['credit_mix']
    avg_outstanding_debt = df.groupby('customer_id')['outstanding_debt'].mean().round().astype(int).reset_index()['outstanding_debt']
    
    df_unique = df.drop_duplicates(subset=['customer_id'])

    unique_id = df_unique['customer_id'].sort_values().reset_index(drop=True)

    clean_data = pd.DataFrame({
    'aver_annual_income': avg_annual_income,
    'avg_monthly_inhand_salary': avg_monthly_inhand_salary,
    'avg_total_emi_per_month': avg_total_emi_per_month,
    'avg_num_bank_accounts': avg_num_bank_accounts,
    'avg_num_credit_card': avg_num_credit_card,
    'avg_num_loans': avg_num_loans,
    'avg_monthly_balance': avg_monthly_balance,
    'avg_credit_history': avg_credit_history,
    'avg_delay': avg_delay,
    'avg_num_inquires': avg_num_inquires,
    'avg_outstanding_debt': avg_outstanding_debt,
    'avg_credit_mix': avg_credit_mix,
    'avg_credit_score': avg_credit_score

    })

    return clean_data

