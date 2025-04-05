import pandas as pd
import numpy as np

class FICOScoreModel:
    def calculate_payment_history_score(self, delay_from_due_date, months_on_file=0):
        # Uses avg_delay and avg_credit_history as inputs.
        values_array = []
        values_array.append(delay_from_due_date)
        values_array.append(months_on_file)
        
        if isinstance(delay_from_due_date, (list, np.ndarray)) or hasattr(delay_from_due_date, 'to_numpy'):
            delays = np.array(delay_from_due_date)
            late_payments_15_days = np.sum((delays > 0) & (delays <= 15))
            late_payments_30_days = np.sum((delays > 15) & (delays <= 30))
            late_payments_45_days = np.sum((delays > 30) & (delays <= 45))
            late_payments_60_days = np.sum((delays > 45))
        else:
            late_payments_15_days = 1 if 0 < delay_from_due_date <= 15 else 0
            late_payments_30_days = 1 if 15 < delay_from_due_date <= 30 else 0
            late_payments_45_days = 1 if 30 < delay_from_due_date <= 45 else 0
            late_payments_60_days = 1 if 50 < delay_from_due_date <= 60 else 0

        values_array.append(late_payments_15_days)
        values_array.append(late_payments_30_days)
        values_array.append(late_payments_45_days)
        values_array.append(late_payments_60_days)
        
        base_score = 100
        values_array.append(base_score)

        deductions = (late_payments_15_days * 10
                + late_payments_30_days * 30
                + late_payments_45_days * 50
                + late_payments_60_days * 70)
        values_array.append(deductions)

        raw_score = max(0, base_score - deductions)
        values_array.append(raw_score)

        if months_on_file < 12:
            raw_score *= (0.5 + (months_on_file / 24))
            values_array.append(raw_score)

        final_score = min(raw_score, 100)
        values_array.append(final_score)

        return final_score, values_array

    def calculate_credit_utilization_score(self, credit_utilization_ratio):
        # Assumes avg_outstanding_debt represents a utilization ratio (0-1)
        if credit_utilization_ratio <= 0.30:
            utilization_score = 100
        elif credit_utilization_ratio <= 0.40:
            utilization_score = 90
        elif credit_utilization_ratio <= 0.50:
            utilization_score = 80
        elif credit_utilization_ratio <= 0.60:
            utilization_score = 70
        elif credit_utilization_ratio <= 0.70:
            utilization_score = 60
        elif credit_utilization_ratio <= 0.80:
            utilization_score = 50
        elif credit_utilization_ratio <= 0.90:
            utilization_score = 40
        
        else:
            utilization_score = 20
        return min(utilization_score, 100)

    def calculate_length_of_history_score(self, credit_history_age_months=0):
        # Uses avg_credit_history as months on file
        if credit_history_age_months >= 120:
            oldest_account_score = 100
        elif credit_history_age_months >= 100:
            oldest_account_score = 95
        elif credit_history_age_months >= 80:
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

    def calculate_credit_mix_score(self, avg_credit_mix=0):
        # Here we assume avg_credit_mix is already a score between 0 and 100.
        if avg_credit_mix == 0:
            mix_score = 20
        elif avg_credit_mix == 1:
            mix_score = 60
        else:
            mix_score = 100
        return min(max(mix_score, 0), 100)

    def calculate_inquiries_score(self, inquiries_last_12_months=0):
        # Uses avg_num_inquires
        if inquiries_last_12_months == 0:
            recent_inquiries_score = 100
        elif inquiries_last_12_months <= 2:
            recent_inquiries_score = 90
        elif inquiries_last_12_months <= 5:
            recent_inquiries_score = 75
        elif inquiries_last_12_months >= 6:
            recent_inquiries_score = 50
        else:
            recent_inquiries_score = 40
        return min(recent_inquiries_score, 100)

    def calculate_fico_score(self, credit_profile):
        # Map the new keys from the DataFrame
        payment_history_score, _ = self.calculate_payment_history_score(
            delay_from_due_date=credit_profile.get('avg_delay'),
            months_on_file=credit_profile.get('avg_credit_history', 0)
        )
        credit_utilization_score = self.calculate_credit_utilization_score(
            credit_utilization_ratio=credit_profile.get('avg_outstanding_debt', 0.0)
        )
        length_of_history_score = self.calculate_length_of_history_score(
            credit_history_age_months=credit_profile.get('avg_credit_history', 0)
        )
        credit_mix_score = self.calculate_credit_mix_score(
            avg_credit_mix=credit_profile.get('avg_credit_mix', 0)
        )
        inquiries_score = self.calculate_inquiries_score(
            inquiries_last_12_months=credit_profile.get('avg_num_inquires', 0)
        )
        weighted_scores = {
            'payment_history': payment_history_score * 0.35,
            'amounts_owed': credit_utilization_score * 0.30,
            'length_of_history': length_of_history_score * 0.15,
            'credit_mix': credit_mix_score * 0.10,
            'inquiries': inquiries_score * 0.10
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
                'inquiries': inquiries_score
            },
            'weighted_scores': weighted_scores
        }
    
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        # Group by customer_id (assuming there could be duplicate rows per customer)
        # This will group the rows and let us work with each customer's group
        grouped = data.groupby('customer_id')
        for cust_id, group in grouped:
            # Use the first row from the group (or handle duplicates as needed)
            row = group.iloc[0]
            profile = row.to_dict()
            score_result = self.calculate_fico_score(profile)
            results.append({
                "Customer ID": cust_id,  # Using the group key directly
                "Avg Credit History": profile.get('avg_credit_history'),
                "Avg Delay": profile.get('avg_delay'),
                "Avg Num Inquires": profile.get('avg_num_inquires'),
                "Avg Outstanding Debt": profile.get('avg_outstanding_debt'),
                "Avg Credit Mix": profile.get('avg_credit_mix'),
                "Original Credit Score": profile.get('avg_credit_score'),
                "Payment History Score": score_result.get('component_scores', {}).get('payment_history'),
                "Amounts Owed Score": score_result.get('component_scores', {}).get('amounts_owed'),
                "Length of History Score": score_result.get('component_scores', {}).get('length_of_history'),
                "Credit Mix Score": score_result.get('component_scores', {}).get('credit_mix'),
                "Inquiries Score": score_result.get('component_scores', {}).get('inquiries'),
                "Calculated FICO Score": score_result.get('fico_score'),
        })
        return pd.DataFrame(results)
