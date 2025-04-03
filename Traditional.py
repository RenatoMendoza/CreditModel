import pandas as pd
import numpy as np

class FICOScoreModel:
    """
    A simplified implementation of a FICO-like credit scoring model.
    
    This model incorporates the five major components that influence FICO scores:
    1. Payment History (35%)
    2. Amounts Owed (30%)
    3. Length of Credit History (15%)
    4. Credit Mix (10%)
    5. New Credit (10%)
    """
    
    def __init__(self):
        # Define the component weights based on FICO's published methodology
        self.weights = {
            'payment_history': 0.35,
            'amounts_owed': 0.30,
            'length_of_history': 0.15,
            'credit_mix': 0.10,
            'new_credit': 0.10
        }
        
        # Initialize score ranges - FICO scores range from 300-850
        self.min_score = 300
        self.max_score = 850
    
    def calculate_payment_history_score(self, 
                                       late_payments_30_days=0,
                                       late_payments_60_days=0, 
                                       late_payments_90_days=0,
                                       months_on_file=0):
        """
        Calculate score component for payment history.
        
        Parameters:
        - late_payments_*: Count of late payments in the respective category
        - months_on_file: Total months of credit history
        """
        # Base score for perfect payment history
        base_score = 100
        
        # Calculate deductions based on negative items
        deductions = 0
        
        # Late payments have increasing impact based on severity
        deductions += late_payments_30_days * 10
        deductions += late_payments_60_days * 20
        deductions += late_payments_90_days * 40
        
        # Ensure score doesn't go below 0
        raw_score = max(0, base_score - deductions)
        
        # Normalize score if necessary based on months on file
        if months_on_file < 12:
            raw_score *= (0.5 + (months_on_file / 24))
        
        return min(raw_score, 100)  # Cap at 100
    
    def calculate_credit_utilization_score(self, 
                                   credit_utilization_ratio):
        """
        Calculate creditation utilization score component.
        
        Parameters:
        credit_utilization_ratio
        """
        # Lower utilization is better, with optimal being under 10%
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
            
        raw_score = (utilization_score)
                    
        return min(raw_score, 100)  # Cap at 100
    
    def calculate_length_of_history_score(self, 
                                        credit_history_age_months=0):
        """
        Calculate score component for length of credit history.
        
        Parameters:
        credit_history_age_months: Age of credit history in months
        """
        # Oldest account age (longer is better)
        if credit_history_age_months >= 240:  # 20+ years
            oldest_account_score = 100
        elif credit_history_age_months >= 180:  # 15+ years
            oldest_account_score = 95
        elif credit_history_age_months >= 120:  # 10+ years
            oldest_account_score = 90
        elif credit_history_age_months >= 60:  # 5+ years
            oldest_account_score = 80
        elif credit_history_age_months >= 24:  # 2+ years
            oldest_account_score = 65
        elif credit_history_age_months >= 12:  # 1+ years
            oldest_account_score = 50
        else:
            oldest_account_score = 30
            
        
        # Calculate weighted average
        raw_score = (oldest_account_score)
                    
        return min(raw_score, 100)  # Cap at 100
    
    def calculate_credit_mix_score(self, 
                                 num_credit_cards=0,
                                 num_retail_accounts=0,
                                 num_installment_loans=0,
                                 num_mortgage_loans=0,
                                 total_accounts=0):
        """
        Calculate score component for credit mix.
        
        Parameters:
        - num_*: Count of different account types
        - total_accounts: Total number of credit accounts
        """
        # No accounts results in poor score
        if total_accounts == 0:
            return 30
            
        # Calculate diversity ratio - how many different types of credit
        account_types = 0
        if num_credit_cards > 0:
            account_types += 1
        if num_retail_accounts > 0:
            account_types += 1
        if num_installment_loans > 0:
            account_types += 1
        if num_mortgage_loans > 0:
            account_types += 1
            
        diversity_ratio = account_types / 4  # Out of 4 possible types
        
        # Higher diversity ratio is better
        if diversity_ratio >= 0.75:  # 3-4 types
            diversity_score = 100
        elif diversity_ratio >= 0.50:  # 2 types
            diversity_score = 80
        elif diversity_ratio >= 0.25:  # 1 type
            diversity_score = 60
        else:
            diversity_score = 40
            
        # Account distribution score (having a good mix of different accounts)
        has_good_mix = False
        
        # A "good mix" typically includes at least one installment and one revolving account
        if (num_credit_cards >= 1 or num_retail_accounts >= 1) and (num_installment_loans >= 1 or num_mortgage_loans >= 1):
            has_good_mix = True
            
        mix_score = 100 if has_good_mix else 70
        
        # Calculate weighted score
        raw_score = (diversity_score * 0.60 + mix_score * 0.40)
        
        return min(raw_score, 100)  # Cap at 100
    
    def calculate_new_credit_score(self, 
                                 inquiries_last_12_months=0):
        """
        Calculate score component for new credit.
        
        Parameters:
        - inquiries_*: Number of credit inquiries in respective time periods
        """
        # Recent inquiries score (fewer is better)
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

            
        # Calculate weighted score
        raw_score = (recent_inquiries_score)
                    
        return min(raw_score, 100)  # Cap at 100
    
    def calculate_fico_score(self, credit_profile):
        """
        Calculate the overall FICO-like score based on a credit profile.
        
        Parameters:
        - credit_profile: Dictionary containing all relevant credit information
        
        Returns:
        - FICO-like score between 300-850
        """
        # Calculate each component score (0-100 scale)
        payment_history_score = self.calculate_payment_history_score(
            late_payments_30_days=credit_profile.get('late_payments_30_days', 0),
            late_payments_60_days=credit_profile.get('late_payments_60_days', 0),
            late_payments_90_days=credit_profile.get('late_payments_90_days', 0),
            months_on_file=credit_profile.get('months_on_file', 0)
        )
        
        credit_utilization_score = self.calculate_credit_utilization_score(
            credit_utilization_ratio=credit_profile.get('credit_utilization_ratio', 0.0)
        )
        
        length_of_history_score = self.calculate_length_of_history_score(
            credit_history_age_months=credit_profile.get('credit_history_age_month', 0)
        )
        
        credit_mix_score = self.calculate_credit_mix_score(
            num_credit_cards=credit_profile.get('num_credit_cards', 0),
            num_retail_accounts=credit_profile.get('num_retail_accounts', 0),
            num_installment_loans=credit_profile.get('num_installment_loans', 0),
            num_mortgage_loans=credit_profile.get('num_mortgage_loans', 0),
            total_accounts=credit_profile.get('total_accounts', 0)
        )
        
        new_credit_score = self.calculate_new_credit_score(
            inquiries_last_12_months=credit_profile.get('inquiries_last_12_months', 0)
            )
        
        # Calculate weighted component scores
        weighted_scores = {
            'payment_history': payment_history_score * self.weights['payment_history'],
            'amounts_owed': credit_utilization_score * self.weights['amounts_owed'],
            'length_of_history': length_of_history_score * self.weights['length_of_history'],
            'credit_mix': credit_mix_score * self.weights['credit_mix'],
            'new_credit': new_credit_score * self.weights['new_credit']
        }
        
        # Calculate base score (0-100 scale)
        base_score = sum(weighted_scores.values())
        
        # Convert to FICO range (300-850)
        fico_score = int(self.min_score + (base_score / 100) * (self.max_score - self.min_score))
        
        # Return the final score and component scores for transparency
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

# Example usage
if __name__ == "__main__":
    # Initialize the model
    fico_model = FICOScoreModel()
    
    # Create a sample credit profile
    sample_credit_profile = {
        # Payment History
        'late_payments_30_days': 0,
        'late_payments_60_days': 0,
        'late_payments_90_days': 0,
        'months_on_file': 120,
        
        # Amounts Owed
        'credit_utilization_ratio': 0.10,
        
        # Length of Credit History
        'credit_history_age_month': 60,
        
        # Credit Mix
        'num_credit_cards': 2,
        'num_retail_accounts': 1,
        'num_installment_loans': 1,
        'num_mortgage_loans': 1,
        'total_accounts': 5,
        
        # New Credit
        'inquiries_last_12_months': 2
    }
    
    # Calculate FICO score
    result = fico_model.calculate_fico_score(sample_credit_profile)
    
    # Print results
    print(f"Estimated FICO Score: {result['fico_score']}")
    print("\nComponent Scores (0-100):")
    for component, score in result['component_scores'].items():
        print(f"- {component}: {score:.1f}")
    
    print("\nWeighted Scores:")
    for component, score in result['weighted_scores'].items():
        print(f"- {component}: {score:.1f}")

# Function to analyze what steps could improve a credit score
def analyze_credit_improvement(fico_result):
    """
    Analyze a FICO score result and provide recommendations for improvement.
    
    Parameters:
    - fico_result: Dictionary containing FICO score and component scores
    
    Returns:
    - List of recommendations for improving credit score
    """
    recommendations = []
    component_scores = fico_result['component_scores']
    
    # Check payment history
    if component_scores['payment_history'] < 90:
        recommendations.append("Focus on making all payments on time. This is the most important factor.")
        
    # Check amounts owed
    if component_scores['amounts_owed'] < 90:
        recommendations.append("Work on reducing credit card balances to below 10% of your credit limits.")
        recommendations.append("Pay down existing debt, particularly revolving credit balances.")
        
    # Check new credit
    if component_scores['new_credit'] < 90:
        recommendations.append("Limit applications for new credit in the next 6-12 months.")
        
    # Check credit mix
    if component_scores['credit_mix'] < 80:
        recommendations.append("Consider diversifying your credit portfolio with different types of credit.")
        
    # Check length of credit history
    if component_scores['length_of_history'] < 80:
        recommendations.append("Keep older accounts open to maintain a longer credit history.")
        
    return recommendations