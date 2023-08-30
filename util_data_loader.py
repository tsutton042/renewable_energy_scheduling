from datetime import datetime
from datetime import timedelta
from distutils.util import strtobool

import pandas as pd


# Converts the contents in a .tsf file into a dataframe and returns it along with other meta-data of the dataset: frequency, horizon, whether the dataset contains missing values and whether the series have equal lengths
#
# Parameters
# full_file_path_and_name - complete .tsf file path
# replace_missing_vals_with - a term to indicate the missing values in series in the returning dataframe
# value_column_name - Any name that is preferred to have as the name of the column containing series values in the returning dataframe
def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )


# Example of usage
# loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe("TSForecasting/tsf_data/sample.tsf")

# print(loaded_data)
# print(frequency)
# print(forecast_horizon)
# print(contain_missing_values)
# print(contain_equal_length)


# Converts the contents of a single ppoi problem instance file into a tuple of dataframes
def read_ppoi(file_path):

    """
        
    13Oct2022 CURRENTLY HAS NO INPUT VALIDATION (INCL. FILE PATH)
    
    Returns
    -------
    Tuple:
    1. Problem instance header <ppoi, #bld, #sol, #bat, #recurring, #once-off>
    2. Building data - building_instance
    3. Solar data - solar_instance
    4. Battery data - battery_instance
    5. Recurring activity data - recurring_activity_instance
    6. Once-off activity data - once_activity_instance

    """
    
    with open(str(file_path)) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        
    lines = [line.split() for line in lines]
    
    # the ppoi format starts with a header that contain various counts
    ppoi = lines[0]
    ppoi_counts = {
        'building': int(ppoi[1]),
        'solar': int(ppoi[2]),
        'battery': int(ppoi[3]),
        'recur_act': int(ppoi[4]),
        'once_act': int(ppoi[5])}
    
    # get building data
    building_instance = lines[1:1+ppoi_counts['building']]
    names = ['instance_type','building_id','no_small_rooms','no_large_rooms']
    building_instance = pd.DataFrame(building_instance, columns = names)
    building_instance = building_instance.astype(dtype={'instance_type':'object',
                                                        'building_id':'int64',
                                                        'no_small_rooms':'int64',
                                                        'no_large_rooms':'int64'})
    
    # get solar data
    solar_instance = lines[1+ppoi_counts['building']:1+ppoi_counts['building']+ppoi_counts['solar']]
    names = ['instance_type','solar_id','building_id']
    solar_instance = pd.DataFrame(solar_instance, columns = names)
    solar_instance = solar_instance.astype(dtype={'instance_type':'object',
                                                  'solar_id':'int64',
                                                  'building_id':'int64'})
    
    # get battery data 
    battery_instance = []
    for i in range(len(lines)) : 
        if lines[i][0] == 'c' :
            battery_instance.append(lines[i])
    
    if len(battery_instance) > 0 : 
        names = ['instance_type','battery_id','building_id','capacity_kwh','max_power_kwh','efficiency']
        battery_instance = pd.DataFrame(battery_instance, columns = names)
        battery_instance = battery_instance.astype(dtype={'instance_type':'object',
                                                          'battery_id':'int64',
                                                          'building_id':'int64',
                                                          'capacity_kwh':'float16',
                                                          'max_power_kwh':'float16',
                                                          'efficiency':'float16'})
    
    # get recurring activity data
    recurring_activity_instance = []
    
    for i in range(len(lines)) : 
        if lines[i][0] == 'r' :
            if lines[i][6] == '0':  # if there are no preceding activities
                recurring_activity_instance.append(lines[i].copy() + [[]])
            else :
                recurring_activity_instance.append(lines[i][:7].copy() + [lines[i][7:].copy()])
    
    for i in range(len(recurring_activity_instance)): 
        names = ['instance_type','activity_id','no_of_rooms','room_size','load_kwh','duration','no_of_precedence','precedence_list']
        recurring_activity_instance = pd.DataFrame(recurring_activity_instance, columns = names)
        recurring_activity_instance = recurring_activity_instance.astype(dtype={'instance_type':'object',
                                                                                'activity_id':'int64',
                                                                                'no_of_rooms':'int64',
                                                                                'room_size':'object',
                                                                                'load_kwh':'float16',
                                                                                'duration':'int64',
                                                                                'no_of_precedence':'int64',
                                                                                'precedence_list':'object',})
        # convert precedence list items to int
        recurring_activity_instance['precedence_list'] = recurring_activity_instance['precedence_list'].apply(lambda x:[int(y) for y in x])  # im a god
        
    # recurring: we want new columns to indicate how many small/large rooms
    x = recurring_activity_instance.loc[:,'room_size']
    small_room = [0]*len(x)
    large_room = [0]*len(x)
    for i in range(len(x)):
        if x[i] == "S":
            small_room[i] = recurring_activity_instance.loc[:,'no_of_rooms'][i]
        else: 
            large_room[i] = recurring_activity_instance.loc[:,'no_of_rooms'][i]
    recurring_activity_instance['no_small_rooms'] = small_room
    recurring_activity_instance['no_large_rooms'] = large_room

        
    # get once-off activity data
    once_activity_instance = []
    
    for i in range(len(lines)) : 
        if lines[i][0] == 'a' :
            if lines[i][8] == '0':
                once_activity_instance.append(lines[i].copy() + [[]])
            else : 
                once_activity_instance.append(lines[i][:9].copy() + [lines[i][9:].copy()])
    
    for i in range(len(once_activity_instance)): 
        names = ['instance_type','activity_id','no_of_rooms','room_size','load_kwh','duration','value','penalty','no_of_precedence','precedence_list']
        once_activity_instance = pd.DataFrame(once_activity_instance, columns = names)
        once_activity_instance = once_activity_instance.astype(dtype={'instance_type':'object',
                                                                      'activity_id':'int64',
                                                                      'no_of_rooms':'int64',
                                                                      'room_size':'object',
                                                                      'load_kwh':'float16',
                                                                      'duration':'int64',
                                                                      'value':'float16',
                                                                      'penalty':'float16',
                                                                      'no_of_precedence':'int64',
                                                                      'precedence_list':'object'})
        # convert precedence list items to int
        once_activity_instance['precedence_list'] = once_activity_instance['precedence_list'].apply(lambda x:[int(y) for y in x])  
    
    # once-off: we want new columns to indicate how many small/large rooms
    x = once_activity_instance.loc[:,'room_size']
    small_room = [0]*len(x)
    large_room = [0]*len(x)
    for i in range(len(x)):
        if x[i] == "S":
            small_room[i] = once_activity_instance.loc[:,'no_of_rooms'][i]
        else: 
            large_room[i] = once_activity_instance.loc[:,'no_of_rooms'][i]
    once_activity_instance['no_small_rooms'] = small_room
    once_activity_instance['no_large_rooms'] = large_room
    
    
    
    # return all of the data in a neat tuple
    ret_tuple = (ppoi_counts, building_instance, solar_instance, battery_instance, recurring_activity_instance, once_activity_instance)
    return ret_tuple



# Reads in AEMO electricity price data
def read_aemo(filepath):
    
    """
        
    15Oct2022 CURRENTLY HAS NO INPUT VALIDATION (INCL. FILE PATH)
    
    Function logic
    -------
    - Read in the AEMO csv
    - Filter to the 2 relevant columns (time and price)
    - Coerce time (string) to datetime
    - Convert time granularity from half-hourly to 15-minutely (double the rows)
    - Return new dataframe

    """

    df0 = pd.read_csv(filepath)
    df1 = df0.iloc[:, [1,3]]
    df1['SETTLEMENTDATE'] = df1['SETTLEMENTDATE'].astype('datetime64[ns]')

    timestep = []
    price = []
    for index, row in df1.iterrows():
        # add time-15m and current time
        timestep.append(row['SETTLEMENTDATE'] - timedelta(minutes=15))
        timestep.append(row['SETTLEMENTDATE'])
        # add prices of both timesteps to the electricity list
        price.append(row['RRP'])
        price.append(row['RRP'])
        
    # create dataframe from dict
    d = {'timestep':timestep,
         'electricity': price}
    df = pd.DataFrame(d)

    # sweet sweet release
    return df



