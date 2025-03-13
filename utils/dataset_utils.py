import pandas as pd

def parse_ctabgan_data(
    data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    This converts the dataframes to their desired format.
    - CTABGAN models generate discrete features as objects, thus we need to parse them to integer.
    """

    data_converted = synthetic_data.copy(deep=True)

    for col in data.columns:
        data_converted[col] = data_converted[col].astype(data[col].dtype)
        
    data_converted = data_converted.loc[:, data.columns]
        
    return data_converted
    

def parse_categorical_dataset(
    data: pd.DataFrame | pd.Series,
) -> pd.DataFrame | pd.Series:
    """
    This converts the dataframes to their desired format.
    - Scikit-learn is not able to parse categorical columns so convert them
    """

    if isinstance(data, pd.DataFrame):
        data_converted = data.copy(deep=True)
        data_converted = data_converted.reset_index(drop=True)
        for col in data_converted.columns:
            if data_converted[col].dtype == 'category':
                data_converted[col] = data_converted[col].astype('object')
    elif isinstance(data, pd.Series):
        data_converted = data.copy(deep=True)
        data_converted = data_converted.reset_index(drop=True)
        if data_converted.dtype == 'category':
            data_converted = data_converted.astype('object')

    else:
        raise ValueError(f'data is type: {type(data).__name__}, expected: pd.Dataframe or pd.Series')    

    return data_converted