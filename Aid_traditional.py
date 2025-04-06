import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def clean_data_trad(file_name):
    df = pd.read_csv(file_name)

    credit_score_mapping = {
    'Bad': 0,
    'Standard': 1,
    'Good': 2
    }

    # Convert categorical credit_score to numerical
    df['credit_mix'] = df['credit_mix'].map(credit_score_mapping)

    avg_credit_history = df.groupby('customer_id')['credit_history_age'].mean().round().astype(int).reset_index()['credit_history_age']
    avg_delay = df.groupby('customer_id')['delay_from_due_date'].mean().round().astype(int).reset_index()['delay_from_due_date']
    avg_num_inquires = df.groupby('customer_id')['num_credit_inquiries'].mean().round().astype(int).reset_index()['num_credit_inquiries']
    avg_credit_score = df.groupby('customer_id')['credit_score'].mean().round().astype(int).reset_index()['credit_score']
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

def classify_fico(score):
    if score < 525:
        return 0  # Bad
    elif score < 710:
        return 1  # Standard
    else:
        return 2  # Good
