import pandas as pd
import numpy as np

def cargar_datos_csv(file_path='credit_score.csv'):
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
    
    def _init_(self):
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
    
    def calculate_payment_history_score(self, delay_from_due_date, num_of_delayed_payment=0):
        """
        Calcula el componente del historial de pagos utilizando columnas de retrasos.

        Parámetros:
        - delay_from_due_date: Array o columna de DataFrame que indica el retraso 
                               (en días) respecto a la fecha de vencimiento.
        - num_of_delayed_payment: Array o columna de DataFrame con el número de pagos retrasados.

        Retorna:
        - Un array de puntajes entre 0 y 100 basado en el historial de pagos.
        """
        # Convert inputs to numpy arrays if they aren't already
        delays = np.asarray(delay_from_due_date)
        delayed_payments = np.asarray(num_of_delayed_payment)
        
        # Initialize arrays for counting late payments by severity
        late_payments_30_days = np.zeros_like(delays, dtype=int)
        late_payments_60_days = np.zeros_like(delays, dtype=int)
        late_payments_90_days = np.zeros_like(delays, dtype=int)
        
        # Count late payments for each record based on delay days
        late_payments_30_days = np.where((delays > 0) & (delays <= 30), 1, 0)
        late_payments_60_days = np.where((delays > 30) & (delays <= 60), 1, 0)
        late_payments_90_days = np.where((delays > 60) & (delays <= 90), 1, 0)

        # Puntaje base para un historial perfecto
        base_score = np.full_like(delays, 100, dtype=float)

        # Aplicar deducciones según la gravedad de los retrasos
        deductions = (late_payments_30_days * 10 +
                      late_payments_60_days * 20 +
                      late_payments_90_days * 40)
        
        # Additional deductions based on number of delayed payments
        additional_deductions = np.minimum(delayed_payments * 5, 30)  # Cap at 30 points
        
        # Calcular puntaje crudo
        raw_score = np.maximum(0, base_score - deductions - additional_deductions)

        return np.minimum(raw_score, 100)  # Se asegura que el puntaje no supere 100
    
    def calculate_credit_utilization_score(self, credit_utilization_ratio):
        """
        Calculate credit utilization score component.
        
        Parameters:
        credit_utilization_ratio: Array or DataFrame column with utilization ratios
        
        Returns:
        Array of utilization scores
        """
        # Convert input to numpy array if it isn't already
        utilization = np.asarray(credit_utilization_ratio)
        
        # Create result array
        utilization_score = np.zeros_like(utilization, dtype=float)
        
        # Apply score rules using numpy's where function
        utilization_score = np.where(utilization <= 0.10, 100, utilization_score)
        utilization_score = np.where((utilization > 0.10) & (utilization <= 0.20), 90, utilization_score)
        utilization_score = np.where((utilization > 0.20) & (utilization <= 0.30), 80, utilization_score)
        utilization_score = np.where((utilization > 0.30) & (utilization <= 0.40), 70, utilization_score)
        utilization_score = np.where((utilization > 0.40) & (utilization <= 0.50), 60, utilization_score)
        utilization_score = np.where((utilization > 0.50) & (utilization <= 0.60), 50, utilization_score)
        utilization_score = np.where((utilization > 0.60) & (utilization <= 0.70), 40, utilization_score)
        utilization_score = np.where((utilization > 0.70) & (utilization <= 0.80), 30, utilization_score)
        utilization_score = np.where((utilization > 0.80) & (utilization <= 0.90), 20, utilization_score)
        utilization_score = np.where(utilization > 0.90, 10, utilization_score)
                    
        return np.minimum(utilization_score, 100)  # Cap at 100
    
    def calculate_length_of_history_score(self, credit_history_age=0):
        """
        Calculate score component for length of credit history.
        
        Parameters:
        credit_history_age: Array or DataFrame column with age of credit history in months
        
        Returns:
        Array of history length scores
        """
        # Convert input to numpy array if it isn't already
        age_months = np.asarray(credit_history_age)
        
        # Create result array
        oldest_account_score = np.zeros_like(age_months, dtype=float)
        
        # Apply score rules using numpy's where function
        oldest_account_score = np.where(age_months >= 240, 100, oldest_account_score)  # 20+ years
        oldest_account_score = np.where((age_months >= 180) & (age_months < 240), 95, oldest_account_score)  # 15+ years
        oldest_account_score = np.where((age_months >= 120) & (age_months < 180), 90, oldest_account_score)  # 10+ years
        oldest_account_score = np.where((age_months >= 60) & (age_months < 120), 80, oldest_account_score)  # 5+ years
        oldest_account_score = np.where((age_months >= 24) & (age_months < 60), 65, oldest_account_score)  # 2+ years
        oldest_account_score = np.where((age_months >= 12) & (age_months < 24), 50, oldest_account_score)  # 1+ years
        oldest_account_score = np.where(age_months < 12, 30, oldest_account_score)  # < 1 year
        
        return np.minimum(oldest_account_score, 100)  # Cap at 100
    
    def calculate_credit_mix_score(self, credit_mix=None):
        """
        Calculate score component for credit mix based on a classification string.
        
        Parameters:
        credit_mix: Array or DataFrame column with credit mix classifications ('Bad', 'Std', 'Good')
        
        Returns:
        Array of credit mix scores
        """
        # Convert input to numpy array if it isn't already
        mix_values = np.asarray(credit_mix)
        
        # Create result array - default score for unknown values
        mix_score = np.full(len(mix_values), 60, dtype=float)
        
        # Map classification strings to scores (case-insensitive)
        for i in range(len(mix_values)):
            value_lower = str(mix_values[i]).lower()
            if 'good' in value_lower:
                mix_score[i] = 100
            elif 'std' in value_lower or 'standard' in value_lower:
                mix_score[i] = 70
            elif 'bad' in value_lower or 'poor' in value_lower:
                mix_score[i] = 40
        
        return np.minimum(mix_score, 100)  # Cap at 100
    
    def calculate_new_credit_score(self, num_credit_inquiries=0):
        """
        Calculate score component for new credit.
        
        Parameters:
        num_credit_inquiries: Array or DataFrame column with number of inquiries
        
        Returns:
        Array of new credit scores
        """
        # Convert input to numpy array if it isn't already
        inquiries = np.asarray(num_credit_inquiries)
        
        # Create result array
        recent_inquiries_score = np.zeros_like(inquiries, dtype=float)
        
        # Apply score rules using numpy's where function
        recent_inquiries_score = np.where(inquiries == 0, 100, recent_inquiries_score)
        recent_inquiries_score = np.where(inquiries == 1, 90, recent_inquiries_score)
        recent_inquiries_score = np.where(inquiries == 2, 75, recent_inquiries_score)
        recent_inquiries_score = np.where((inquiries > 2) & (inquiries <= 4), 60, recent_inquiries_score)
        recent_inquiries_score = np.where(inquiries > 4, 40, recent_inquiries_score)
                    
        return np.minimum(recent_inquiries_score, 100)  # Cap at 100
    
    def calculate_fico_score_batch(self, df):
        """
        Calculate FICO-like scores for a batch of credit profiles in a DataFrame.
        
        Parameters:
        - df: DataFrame containing credit information columns
        
        Returns:
        - DataFrame with original data plus FICO scores and component scores
        """
        # Create a copy of the input DataFrame to add results
        result_df = df.copy()
        
        # Calculate each component score (0-100 scale)
        # Map DataFrame columns to the appropriate functions using exact column names
        payment_history_scores = self.calculate_payment_history_score(
            df['delay_from_due_date'],
            df['num_of_delayed_payment']
        )
        
        credit_utilization_scores = self.calculate_credit_utilization_score(
            df['credit_utilization_ratio']
        )
        
        length_of_history_scores = self.calculate_length_of_history_score(
            df['credit_history_age']
        )
        
        credit_mix_scores = self.calculate_credit_mix_score(
            df['credit_mix']
        )
        
        new_credit_scores = self.calculate_new_credit_score(
            df['num_credit_inquiries']
        )
        
        # Add component scores to result DataFrame
        result_df['payment_history_score'] = payment_history_scores
        result_df['amounts_owed_score'] = credit_utilization_scores
        result_df['length_of_history_score'] = length_of_history_scores
        result_df['credit_mix_score'] = credit_mix_scores
        result_df['new_credit_score'] = new_credit_scores
        
        # Calculate weighted scores
        result_df['weighted_payment_history'] = payment_history_scores * self.weights['payment_history']
        result_df['weighted_amounts_owed'] = credit_utilization_scores * self.weights['amounts_owed']
        result_df['weighted_length_of_history'] = length_of_history_scores * self.weights['length_of_history']
        result_df['weighted_credit_mix'] = credit_mix_scores * self.weights['credit_mix']
        result_df['weighted_new_credit'] = new_credit_scores * self.weights['new_credit']
        
        # Calculate base score (0-100 scale)
        result_df['base_score'] = (
            result_df['weighted_payment_history'] +
            result_df['weighted_amounts_owed'] +
            result_df['weighted_length_of_history'] +
            result_df['weighted_credit_mix'] +
            result_df['weighted_new_credit']
        )
        
        # Convert to FICO range (300-850)
        result_df['fico_score'] = self.min_score + (result_df['base_score'] / 100) * (self.max_score - self.min_score)
        result_df['fico_score'] = result_df['fico_score'].astype(int)
        
        # Return array of scores if needed
        return result_df
    
    def get_scores_array(self, df):
        """
        Calculate FICO scores and return them as a numpy array.
        
        Parameters:
        - df: DataFrame containing credit information columns
        
        Returns:
        - Numpy array containing only the calculated FICO scores
        """
        result_df = self.calculate_fico_score_batch(df)
        return result_df['fico_score'].to_numpy()

# Function to analyze what steps could improve a credit score
def analyze_credit_improvement(component_scores):
    """
    Analyze component scores and provide recommendations for improvement.
    
    Parameters:
    - component_scores: Dictionary or Series containing component scores
    
    Returns:
    - List of recommendations for improving credit score
    """
    recommendations = []
    
    # Check payment history
    if component_scores['payment_history_score'] < 90:
        recommendations.append("Focus on making all payments on time. This is the most important factor.")
        
    # Check amounts owed
    if component_scores['amounts_owed_score'] < 90:
        recommendations.append("Work on reducing credit card balances to below 10% of your credit limits.")
        recommendations.append("Pay down existing debt, particularly revolving credit balances.")
        
    # Check new credit
    if component_scores['new_credit_score'] < 90:
        recommendations.append("Limit applications for new credit in the next 6-12 months.")
        
    # Check credit mix
    if component_scores['credit_mix_score'] < 80:
        recommendations.append("Consider diversifying your credit portfolio with different types of credit.")
        
    # Check length of credit history
    if component_scores['length_of_history_score'] < 80:
        recommendations.append("Keep older accounts open to maintain a longer credit history.")
        
    return recommendations

# Example usage with a DataFrame using the exact column names
def example_usage():
    # Create a sample DataFrame with your exact column names
    data = {
        'id': [1, 2, 3, 4],
        'customer_id': ['C001', 'C002', 'C003', 'C004'],
        'month': ['Jan', 'Feb', 'Mar', 'Apr'],
        'name': ['John Doe', 'Jane Smith', 'Alex Johnson', 'Maria Garcia'],
        'credit_history_age': [120, 60, 24, 12],
        'delay_from_due_date': [0, 15, 35, 70],
        'num_of_delayed_payment': [0, 1, 3, 5],
        'num_credit_inquiries': [0, 1, 2, 5],
        'credit_mix': ['Good', 'Standard', 'Bad', 'Good'],
        'outstanding_debt': [5000, 15000, 35000, 60000],
        'credit_utilization_ratio': [0.05, 0.15, 0.35, 0.60],
        'credit_score': [750, 680, 620, 580]  # Original credit scores for comparison
    }
    
    df = pd.DataFrame(data)
    
    # Initialize the model and calculate scores
    fico_model = FICOScoreModel()
    results_df = fico_model.calculate_fico_score_batch(df)
    
    # Print results
    print("FICO Score Results:")
    print(results_df[['customer_id', 'name', 'credit_score', 'fico_score', 
                      'payment_history_score', 'amounts_owed_score', 
                      'length_of_history_score', 'credit_mix_score', 'new_credit_score']])
    
    # Example of getting recommendations for the first customer
    print("\nRecommendations for first customer:")
    first_customer_scores = results_df.iloc[0]
    recommendations = analyze_credit_improvement(first_customer_scores)
    for rec in recommendations:
        print(f"- {rec}")
    
    # Return only FICO scores as numpy array
    fico_scores = fico_model.get_scores_array(df)
    print("\nFICO scores as numpy array:")
    print(fico_scores)
    
    return fico_scores

if __name__ == "_main_":
    # Run the example
    scores_array = example_usage()