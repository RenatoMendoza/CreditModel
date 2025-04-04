import pandas as pd
import numpy as np

class FICOScoreModel:
    def calculate_payment_history_score(self, delay_from_due_date, months_on_file=0):
        print("delay_from_due_date:", delay_from_due_date)
        print("months_on_file:", months_on_file)
        values_array = []
        values_array.append(delay_from_due_date)
        values_array.append(months_on_file)
        
        if isinstance(delay_from_due_date, (list, np.ndarray)) or hasattr(delay_from_due_date, 'to_numpy'):
            delays = np.array(delay_from_due_date)
            late_payments_30_days = np.sum((delays > 0) & (delays <= 30))
            late_payments_60_days = np.sum((delays > 30) & (delays <= 60))
            late_payments_90_days = np.sum((delays > 60) & (delays <= 90))
        else:
            delay = delay_from_due_date
            late_payments_30_days = 1 if 0 < delay <= 30 else 0
            late_payments_60_days = 1 if 30 < delay <= 60 else 0
            late_payments_90_days = 1 if 60 < delay <= 90 else 0

        values_array.append(late_payments_30_days)
        values_array.append(late_payments_60_days)
        values_array.append(late_payments_90_days)

        base_score = 100
        values_array.append(base_score)

        deductions = (late_payments_30_days * 10 +
                      late_payments_60_days * 20 +
                      late_payments_90_days * 40)
        values_array.append(deductions)

        raw_score = max(0, base_score - deductions)
        values_array.append(raw_score)

        if months_on_file < 12:
            raw_score *= (0.5 + (months_on_file / 24))
            values_array.append(raw_score)

        final_score = min(raw_score, 100)
        values_array.append(final_score)

        print("Payment History Data:", values_array)
        return final_score, values_array

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
        if num_of_loan == 0:
            return 30

        if isinstance(type_of_loan, str):
            types = [t.strip().lower() for t in type_of_loan.split(',')]
        elif isinstance(type_of_loan, list):
            types = [str(t).strip().lower() for t in type_of_loan]
        else:
            types = []
        
        unique_types = set(types)
        account_types = len(unique_types)
        diversity_ratio = account_types / 4
        if diversity_ratio >= 0.75:
            diversity_score = 100
        elif diversity_ratio >= 0.50:
            diversity_score = 80
        elif diversity_ratio >= 0.25:
            diversity_score = 60
        else:
            diversity_score = 40

        revolving_keywords = ['card', 'retail']
        installment_keywords = ['installment', 'mortgage']

        has_revolving = any(any(keyword in loan_type for keyword in revolving_keywords)
                            for loan_type in unique_types)
        has_installment = any(any(keyword in loan_type for keyword in installment_keywords)
                              for loan_type in unique_types)
        has_good_mix = has_revolving and has_installment
        mix_score = 100 if has_good_mix else 70

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
        payment_history_score, ph_values = self.calculate_payment_history_score(
            delay_from_due_date=credit_profile.get('delay_from_due_date', []),
            months_on_file=credit_profile.get('credit_history_age', 0)
        )
        credit_utilization_score = self.calculate_credit_utilization_score(
            credit_utilization_ratio=credit_profile.get('credit_utilization_ratio', 0.0)
        )
        length_of_history_score = self.calculate_length_of_history_score(
            credit_history_age_months=credit_profile.get('credit_history_age', 0)
        )
        credit_mix_score = self.calculate_credit_mix_score(
            num_of_loan=credit_profile.get('num_of_loan', 0),
            type_of_loan=credit_profile.get('type_of_loan', None)
        )
        new_credit_score = self.calculate_new_credit_score(
            inquiries_last_12_months=credit_profile.get('num_credit_inquiries', 0)
        )
        weighted_scores = {
            'payment_history': payment_history_score * 0.35,
            'amounts_owed': credit_utilization_score * 0.30,
            'length_of_history': length_of_history_score * 0.15,
            'credit_mix': credit_mix_score * 0.10,
            'new_credit': new_credit_score * 0.10
        }
        base_score = sum(weighted_scores.values())
        fico_score = int(300 + (base_score / 100) * (850 - 300))
        
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
    
    # New method: process the entire DataFrame input
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        # Group by customer_id and take the first record for each customer
        grouped = data.groupby('customer_id').first()
        for _, row in grouped.iterrows():
            profile = row.to_dict()
            score_result = self.calculate_fico_score(profile)
            results.append({
                "User Name": profile.get('name', 'Sin Nombre'),
                "Customer ID": profile.get('customer_id'),
                "Delay From Due Date": profile.get('delay_from_due_date'),
                "Months on File": profile.get('credit_history_age'),
                "FICO Score": score_result.get('fico_score'),
                "Payment History Score": score_result.get('component_scores', {}).get('payment_history'),
                "Amounts Owed Score": score_result.get('component_scores', {}).get('amounts_owed'),
                "Length of History Score": score_result.get('component_scores', {}).get('length_of_history'),
                "Credit Mix Score": score_result.get('component_scores', {}).get('credit_mix'),
                "New Credit Score": score_result.get('component_scores', {}).get('new_credit')
            })
        return pd.DataFrame(results)
    
def run(self, data: pd.DataFrame) -> pd.DataFrame:
    results = []
    # Group by customer_id and take the first record for each customer
    grouped = data.groupby('customer_id').first()
    for _, row in grouped.iterrows():
        profile = row.to_dict()
        score_result = self.calculate_fico_score(profile)
        results.append({
            "User Name": profile.get('name', 'Sin Nombre'),
            "Customer ID": profile.get('customer_id'),
            "Delay From Due Date": profile.get('delay_from_due_date'),
            "Months on File": profile.get('credit_history_age'),
            "FICO Score": score_result.get('fico_score'),
            "Payment History Score": score_result.get('component_scores', {}).get('payment_history'),
            "Amounts Owed Score": score_result.get('component_scores', {}).get('amounts_owed'),
            "Length of History Score": score_result.get('component_scores', {}).get('length_of_history'),
            "Credit Mix Score": score_result.get('component_scores', {}).get('credit_mix'),
            "New Credit Score": score_result.get('component_scores', {}).get('new_credit')
        })
    return pd.DataFrame(results)

