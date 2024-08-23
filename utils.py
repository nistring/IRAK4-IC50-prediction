import pandas as pd

def pIC50_to_IC50(pic50_values):
    """Convert pIC50 values to IC50 (nM)."""
    return 10 ** (9 - pic50_values)

file_name = "test_preds"

submit = pd.read_csv('data/sample_submission.csv')
preds = pd.read_csv(f'data/{file_name}.csv')["pred_0"].values
submit['IC50_nM'] = pIC50_to_IC50(preds)
submit.head()
submit.to_csv(f'data/{file_name}_submit.csv', index=False)