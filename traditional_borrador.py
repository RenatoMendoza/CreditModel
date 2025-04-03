import pandas as pd
import numpy as np

def cargar_datos_csv(file_path=r"C:\Users\israe\OneDrive\Documents\8vo semestre\Modelos de crédito\proyecto\CreditModel\credit_score.csv"):
    """
    Carga los datos de crédito desde un archivo CSV.

    Parámetros:
    - file_path: Ruta del archivo CSV (por defecto '/mnt/data/credit_score.csv').

    Retorna:
    - Un DataFrame de pandas con los datos de crédito o None en caso de error.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Datos cargados exitosamente desde {file_path}")
        return df
    except Exception as e:
        print(f"Error al cargar el archivo CSV: {e}")
        return None

class FICOScoreModel:
    """
    Una implementación simplificada de un modelo de puntaje de crédito tipo FICO.
    Este modelo evalúa cinco componentes: historial de pagos, montos adeudados, 
    antigüedad del historial, mezcla de crédito y nuevo crédito.
    """
    
    def __init__(self):
        # Definir los pesos de cada componente según la metodología FICO
        self.weights = {
            'payment_history': 0.35,
            'amounts_owed': 0.30,
            'length_of_history': 0.15,
            'credit_mix': 0.10,
            'new_credit': 0.10
        }
        # Rango de puntaje FICO: 300-850
        self.min_score = 300
        self.max_score = 850
    
    def calculate_payment_history_score(self, delay_from_due_date, months_on_file=0):
        """
        Calcula el componente del historial de pagos utilizando la columna 'delay_from_due_date'.

        Parámetros:
        - delay_from_due_date: Valor único o una serie de valores que indican el retraso 
                               (en días) respecto a la fecha de vencimiento.
        - months_on_file: Total de meses en el historial crediticio.

        Retorna:
        - Un puntaje entre 0 y 100 basado en el historial de pagos.
        """
        # Si se recibe una lista, array o una columna de DataFrame, convertir a array
        if isinstance(delay_from_due_date, (list, np.ndarray)) or hasattr(delay_from_due_date, 'to_numpy'):
            delays = np.array(delay_from_due_date)
            late_payments_30_days = np.sum((delays > 0) & (delays <= 30))
            late_payments_60_days = np.sum((delays > 30) & (delays <= 60))
            late_payments_90_days = np.sum((delays > 60) & (delays <= 90))
        else:
            # Si se recibe un valor único
            delay = delay_from_due_date
            late_payments_30_days = 1 if 0 < delay <= 30 else 0
            late_payments_60_days = 1 if 30 < delay <= 60 else 0
            late_payments_90_days = 1 if 60 < delay <= 90 else 0

        base_score = 100  # Puntaje base para un historial perfecto

        # Aplicar deducciones según la gravedad de los retrasos
        deductions = (late_payments_30_days * 10 +
                      late_payments_60_days * 20 +
                      late_payments_90_days * 40)

        raw_score = max(0, base_score - deductions)

        # Ajustar puntaje si el historial es muy corto
        if months_on_file < 12:
            raw_score *= (0.5 + (months_on_file / 24))

        return min(raw_score, 100)  # Se asegura que el puntaje no supere 100

    def calculate_credit_utilization_score(self, credit_utilization_ratio):
        if credit_utilization_ratio <= 0.10:
            utilization_score = 100
        elif credit_utilization_ratio <= 0.20:
            utilization_score = 90
        elif credit_utilization_ratio <= 0.30:
            utilization_score = 80
        elif credit_utilization_ratio <= 0.40:
            utilization_score = 70
        elif credit_utilization_ratio <= 0.50:
            utilization_score = 60
        elif credit_utilization_ratio <= 0.60:
            utilization_score = 50
        elif credit_utilization_ratio <= 0.70:
            utilization_score = 40
        elif credit_utilization_ratio <= 0.80:
            utilization_score = 30
        elif credit_utilization_ratio <= 0.90:
            utilization_score = 20
        else:
            utilization_score = 10
        return min(utilization_score, 100)

    def calculate_length_of_history_score(self, credit_history_age_months=0):
        if credit_history_age_months >= 240:
            oldest_account_score = 100
        elif credit_history_age_months >= 180:
            oldest_account_score = 95
        elif credit_history_age_months >= 120:
            oldest_account_score = 90
        elif credit_history_age_months >= 60:
            oldest_account_score = 80
        elif credit_history_age_months >= 24:
            oldest_account_score = 65
        elif credit_history_age_months >= 12:
            oldest_account_score = 50
        else:
            oldest_account_score = 30
        return min(oldest_account_score, 100)

    def calculate_credit_mix_score(self, num_of_loan=0, type_of_loan=None):
        """
        Calcula el componente de mezcla de crédito utilizando la información
        de las columnas "num_of_loan" y "type_of_loan" del CSV.

        Parámetros:
        - num_of_loan: Número total de créditos (cuenta total de préstamos).
        - type_of_loan: Cadena o lista que indica los tipos de créditos, separados por comas si es cadena.

        Retorna:
        - Un puntaje de mezcla de crédito entre 0 y 100.
        """
        # Si no hay créditos, asignamos un puntaje bajo.
        if num_of_loan == 0:
            return 30

        # Procesar 'type_of_loan': si es cadena, la separamos; si es lista, la usamos directamente.
        if isinstance(type_of_loan, str):
            # Dividir la cadena por comas y limpiar espacios.
            types = [t.strip().lower() for t in type_of_loan.split(',')]
        elif isinstance(type_of_loan, list):
            types = [str(t).strip().lower() for t in type_of_loan]
        else:
            types = []

        # Contar los tipos únicos de crédito.
        unique_types = set(types)
        account_types = len(unique_types)

        # Se asume que hay 4 tipos posibles (similar al modelo original) para calcular el ratio.
        diversity_ratio = account_types / 4
        if diversity_ratio >= 0.75:
            diversity_score = 100
        elif diversity_ratio >= 0.50:
            diversity_score = 80
        elif diversity_ratio >= 0.25:
            diversity_score = 60
        else:
            diversity_score = 40

        # Definir criterios para una "buena mezcla".
        # Se considera que una buena mezcla existe si hay al menos un crédito revolving
        # (ej.: "card" o "retail") y al menos un crédito installment
        # (ej.: "installment" o "mortgage").
        revolving_keywords = ['card', 'retail']
        installment_keywords = ['installment', 'mortgage']

        has_revolving = any(any(keyword in loan_type for keyword in revolving_keywords) 
                            for loan_type in unique_types)
        has_installment = any(any(keyword in loan_type for keyword in installment_keywords) 
                            for loan_type in unique_types)

        has_good_mix = has_revolving and has_installment
        mix_score = 100 if has_good_mix else 70

        # Calcular el puntaje ponderado para la mezcla de crédito.
        raw_score = diversity_score * 0.60 + mix_score * 0.40
        return min(raw_score, 100)

    def calculate_new_credit_score(self, inquiries_last_12_months=0):
        if inquiries_last_12_months == 0:
            recent_inquiries_score = 100
        elif inquiries_last_12_months == 1:
            recent_inquiries_score = 90
        elif inquiries_last_12_months == 2:
            recent_inquiries_score = 75
        elif inquiries_last_12_months <= 4:
            recent_inquiries_score = 60
        else:
            recent_inquiries_score = 40
        return min(recent_inquiries_score, 100)

    def calculate_fico_score(self, credit_profile):
        """
        Calcula el puntaje global FICO basado en un perfil crediticio.
        Se actualiza la llamada a calculate_payment_history_score para usar 'delay_from_due_date'
        y la de credit mix para usar 'num_of_loan' y 'type_of_loan'.
        """
        payment_history_score = self.calculate_payment_history_score(
            delay_from_due_date=credit_profile.get('delay_from_due_date', []),
            months_on_file=credit_profile.get('credit_history_age', 0)
        )
        credit_utilization_score = self.calculate_credit_utilization_score(
            credit_utilization_ratio=credit_profile.get('credit_utilization_ratio', 0.0)
        )
        length_of_history_score = self.calculate_length_of_history_score(
            credit_history_age_months=credit_profile.get('credit_history_age', 0)
        )
        # Se reemplaza la antigua llamada a credit_mix_score por la nueva que usa 'num_of_loan' y 'type_of_loan'
        credit_mix_score = self.calculate_credit_mix_score(
            num_of_loan=credit_profile.get('num_of_loan', 0),
            type_of_loan=credit_profile.get('type_of_loan', None)
        )
        new_credit_score = self.calculate_new_credit_score(
            inquiries_last_12_months=credit_profile.get('num_credit_inquiries', 0)
        )
        weighted_scores = {
            'payment_history': payment_history_score * self.weights['payment_history'],
            'amounts_owed': credit_utilization_score * self.weights['amounts_owed'],
            'length_of_history': length_of_history_score * self.weights['length_of_history'],
            'credit_mix': credit_mix_score * self.weights['credit_mix'],
            'new_credit': new_credit_score * self.weights['new_credit']
        }
        base_score = sum(weighted_scores.values())
        fico_score = int(self.min_score + (base_score / 100) * (self.max_score - self.min_score))
        return {
            'fico_score': fico_score,
            'component_scores': {
                'payment_history': payment_history_score,
                'amounts_owed': credit_utilization_score,
                'length_of_history': length_of_history_score,
                'credit_mix': credit_mix_score,
                'new_credit': new_credit_score
            },
            'weighted_scores': weighted_scores
        }
if __name__ == "__main__":
    # Cargar datos del CSV
    datos_credito = cargar_datos_csv()
    
    if datos_credito is not None:
        # Ejemplo: Filtrar datos de un cliente específico
        cliente_id = 'CUS_0xd40'
        datos_cliente = datos_credito[datos_credito['customer_id'] == cliente_id]
        
        # Convertir a diccionario usando el primer registro (o definir cómo agrupar si hay múltiples registros)
        sample_credit_profile = datos_cliente.iloc[0].to_dict()
        
        # Inicializar el modelo FICO
        fico_model = FICOScoreModel()
        
        result = fico_model.calculate_fico_score(sample_credit_profile)
        
        print(f"Estimated FICO Score for {cliente_id}: {result['fico_score']}")
        print("\nComponent Scores (0-100):")
        for component, score in result['component_scores'].items():
            print(f"- {component}: {score:.1f}")
        
        print("\nWeighted Scores:")
        for component, score in result['weighted_scores'].items():
            print(f"- {component}: {score:.1f}")
