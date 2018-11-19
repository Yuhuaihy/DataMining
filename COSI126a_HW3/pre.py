import pandas as pd
def generate_cols():
    cols = ['item_a',
            'item_b',
            'support_ab',
            'confidence_a_to_b',
            'confidence_b_to_a']
    return cols

def prep_df(dummy_data, cols):
    df = pd.concat(dummy_data, axis=1)
    df.columns = cols
    df.sort_values(by='support_ab', inplace=True, ascending=False)
    df.reset_index(inplace=True, drop=True)
    return df