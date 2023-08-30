"""
Implements MASE as described on the competition page
"""
import numpy as np

def mase(pred, actual, seasonality):
    """
    Calculates the MASE of the predicted values with respect to the observed values

    Parameters
    ----------
    pred : numpy array
        values predicted by a model
    actual : numpy array
        observed values
    seasonality : integer
        number of timesteps that constitute a season

    Returns
    -------
    Float. The MASE of the predicted values with respect to the observed values

    """
    horizon = pred.shape[0]
    # split actual up into historical and current observations
    current = actual[-horizon:] # take the true values for pred
    historical = actual[:-horizon]
    
    # reindex actual so that it is the seasonality-step ahead forecast
    # historical_1 = actual[:-seasonality] # forecast
    # historical_2 = actual[seasonality:] # actual
    
    numer = 0
    for i in range(current.shape[0]):
        numer += abs(pred[i] - current[i])
        
    denom = 0
    for i in range(seasonality, actual.shape[0]):
        denom += abs(actual[i] - actual[i-seasonality])
        
    denom *= horizon/(actual.shape[0] - seasonality)
    
    return numer/denom