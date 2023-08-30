"""
Various data cleaning and generating functions that are useful in most steps of 
modelling
"""

import numpy as np
import pandas as pd

def filt_vals(data, fill_na_only = True, lower = 0, upper = 3000):
    """
    Replaces missing values with the mean value always.
    If fill_na_only is False, it also applies this to values outside the 
    interval [lower, upper].
    
    If it's more advantageous, this could also be set to a normal RV with
    mean and sd equal to that of the time series

    Parameters
    ----------
    data : pd.Series
        Data to be cleaned.
    fill_na_only : bool, optional
        Does the function only filter out NAs or also 'outliers'?
        The default is True.
    lower : float or int, optional
        The lower bound of the values to keep. The default is 0.
    upper : float or int, optional
        The upper bound of the values to keep. The default is 3000.

    Returns
    -------
    A pd.Series with all specified values set to the mean.

    """
    # make a copy of data so that no side-effects occur
    tmp = data.copy()
    mean = tmp.sum()/tmp.shape[0]
    
    # filter out NaNs 
    tmp.fillna(mean, inplace=True)
        
    # filter out values outside the window
    if not fill_na_only:
        tmp = [x if lower <= x <= upper else mean for x in tmp.iloc[:,0]]
            
    
    return np.array(tmp).astype("float32")


def generate_timestamps(df, time_offset = 0, resolution = 15):
    """
    Generates timestamps for each observation in a tsf file

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame generated from a tsf file using 
        data_loader.convert_tsf_to_dataframe. As the main names in the dataframe
        are standardised, these are hard-coded.
        
    resolution : int
        The amount of time, in minutes, between observations in df. 
        This can be obtained from the header of the tsf file (I can't be 
        bothered to write a parsing algorithm to obtain this automatically,
        but wouldn't be too hard to do)
        
    time_offset : int
        The amount of time, in minutes, that the timestamps are ahead of UTC

    Returns
    -------
    timestamps : pandas.DataFrame
        A DataFrame containing, for each timeseries in df:
            * the name of the timeseries; and 
            * an array of the timestamps of each observation in the series 
    """
    timestamps = pd.DataFrame({"series_name": [], "series_timestamp":[]})
    
    for i in range(df.shape[0]):
        num_vals = len(df.series_value[i])
        times = list(range(num_vals))
        times[0] = df.start_timestamp[i] - pd.Timedelta(minutes = time_offset)
        for j in range(1, num_vals):
            times[j] = times[j-1] + pd.Timedelta(minutes = 15)
        times = np.array(times)
        timestamps.at[i, :] = [df.series_name[i], times]
    
    return timestamps


def extract_data(tsf_df, split_timestamp = None):
    """
    Reshapes data obtained from a .tsf file into a matrix-like dataframe

    Parameters
    ----------
    tsf_df : pandas.DataFrame
        Obtained from reading a .tsf plaintext file with convert_tsf_to_dataframe
        
    split_timestamp : pandas.TimeStamp
        Optional. A timestamp to split the data into train and test sets on.
        If None, the data is not split

    Returns
    -------
        If split_timestamp is None, pandas.DataFrame with the timeseries and 
        timestamps as columns. If split_time_stamp is not None, two such 
        DataFrames, the first being all occurrences with timestamp less 
        than or equal to the given one and the second being all occurrences 
        with a greater timestamp
    """
    
    tsf_df["time_stamps"] = generate_timestamps(tsf_df).series_timestamp

    # limit all time series to start and end at the same times (makes training 
    # easier to write) - we can remove this later. It may involve writing a custom
    # layer in keras as we need to keep the data going in/out as a matrix (ie 
    # fill missing timestamps with NAs for each time series)
    lower_time = pd.to_datetime("2020-04-25 14:00:00", format = "%Y-%m-%d %H:%M:%S")
    for row in range(tsf_df.shape[0]):  # loop over rows
        val_filter = tsf_df.loc[row, "time_stamps"] >= lower_time
        tsf_df.at[row, "time_stamps"] = tsf_df.loc[row, "time_stamps"][val_filter]
        tsf_df.at[row, "series_value"] = tsf_df.loc[row, "series_value"][val_filter]

    # remove any NAs
    for i in range(tsf_df.shape[0]):
        x = pd.Series(tsf_df.loc[i, "series_value"], dtype="float32")
        tsf_df.at[i, "series_value"] = filt_vals(x)
        
    if split_timestamp is None:
        cleaned = pd.DataFrame({"Building0": tsf_df.series_value[0], 
                                "Building1": tsf_df.series_value[1], 
                                "Building3": tsf_df.series_value[2], 
                                "Building4": tsf_df.series_value[3],
                                "Building5": tsf_df.series_value[4],
                                "Building6": tsf_df.series_value[5],
                                "Solar0": tsf_df.series_value[6],
                                "Solar1": tsf_df.series_value[7],
                                "Solar2": tsf_df.series_value[8],
                                "Solar3": tsf_df.series_value[9],
                                "Solar4": tsf_df.series_value[10],
                                "Solar5": tsf_df.series_value[11]})
        
        cleaned["timestamp"] = tsf_df.time_stamps[0]
        return cleaned
        
    else:
        # split into training and prediction/testing sets    
        for row in range(tsf_df.shape[0]):
            split_filt = tsf_df.loc[row, "time_stamps"] < split_timestamp
            
            train_series = tsf_df.loc[row, "series_value"][split_filt]
            train_times = tsf_df.loc[row, "time_stamps"][split_filt]
            train_LS.append([tsf_df.series_name[row], train_series, train_times])
            
            test_series = tsf_df.loc[row, "series_value"][~split_filt]
            test_times = tsf_df.loc[row, "time_stamps"][~split_filt]
            test_LS.append([tsf_df.series_name[row], test_series, test_times])
            
        train_LS = pd.DataFrame(train_LS)
        train_LS.columns = ["series_name", "series_value", "time_stamps"]
        test_LS = pd.DataFrame(test_LS)
        test_LS.columns = ["series_name", "series_value", "time_stamps"]
        
        train = pd.DataFrame({"Building0": train_LS.series_value[0], 
                              "Building1": train_LS.series_value[1], 
                              "Building3": train_LS.series_value[2], 
                              "Building4": train_LS.series_value[3],
                              "Building5": train_LS.series_value[4],
                              "Building6": train_LS.series_value[5],
                              "Solar0": train_LS.series_value[6],
                              "Solar1": train_LS.series_value[7],
                              "Solar2": train_LS.series_value[8],
                              "Solar3": train_LS.series_value[9],
                              "Solar4": train_LS.series_value[10],
                              "Solar5": train_LS.series_value[11]})

        test = pd.DataFrame({"Building0": test_LS.series_value[0], 
                             "Building1": test_LS.series_value[1], 
                             "Building3": test_LS.series_value[2], 
                             "Building4": test_LS.series_value[3],
                             "Building5": test_LS.series_value[4],
                             "Building6": test_LS.series_value[5],
                             "Solar0": test_LS.series_value[6],
                             "Solar1": test_LS.series_value[7],
                             "Solar2": test_LS.series_value[8],
                             "Solar3": test_LS.series_value[9],
                             "Solar4": test_LS.series_value[10],
                             "Solar5": test_LS.series_value[11]})
        
        train["timestamp"] = train_LS.time_stamps[0]
        test["timestamp"] = test_LS.time_stamps[0]
        
        return train, test
    
    
def concat_power_weather(power, era5):
    """
    Concat each building and solar with its corresponding weather data 
    
    Parameters
    ----------
    power : pd.DataFrame
        A DataFrame, in the same format as one processed with extract_data()
        As the names in the dataframe are standardised, these are hard-coded.
        
    era5 : pd.DataFrame
        A DataFrame, in the same format as ERA5_Weather_Data_Monash.csv

    Returns
    -------
    A timestamp-matched dataframe of the useful columns from power and era5
    with missing timestamps filled by the previous complete one
    
    """
    
    # merge power and weather by timestamp
    res = power.merge(era5, 
                      how = "left", 
                      left_on = "timestamp", 
                      right_on = "datetime (UTC)")
    
    # remove unused cols - these are straight from the csv, so it is ok to 
    # hardcode
    res = res[['timestamp', 
               'Building0', 'Building1', 'Building3', 'Building4',
               'Building5', 'Building6', 
               'Solar0', 'Solar1', 'Solar2', 'Solar3', 'Solar4',  'Solar5', 
               'temperature (degC)', 'dewpoint_temperature (degC)', 
               'wind_speed (m/s)', 'mean_sea_level_pressure (Pa)', 
               'relative_humidity ((0-1))', 'surface_solar_radiation (W/m^2)', 
               'surface_thermal_radiation (W/m^2)', 'total_cloud_cover (0-1)']]
    
    # rename to be more concise
    res.columns = ['timestamp', 
                   'Building0', 'Building1', 'Building3', 'Building4', 
                   'Building5', 'Building6', 
                   'Solar0', 'Solar1', 'Solar2', 'Solar3', 'Solar4', 'Solar5', 
                   'temperature', 'dewpoint_temperature', 'wind_speed', 
                   'mean_sea_level_pressure', 'relative_humidity', 
                   'surface_solar_radiation', 'surface_thermal_radiation', 
                   'total_cloud_cover']
    
    era5_cols = ['temperature', 'dewpoint_temperature', 'wind_speed', 
                 'mean_sea_level_pressure', 'relative_humidity', 
                 'surface_solar_radiation', 'surface_thermal_radiation', 
                 'total_cloud_cover']
    
    
    for i in range(1, res.shape[0]):  # assuming that th first row has a match
        if pd.isna(res.wind_speed[i]):  # just using an arbitrary column from era5
            res.loc[i, era5_cols] = res.loc[i-1, era5_cols]
        
    return res

        