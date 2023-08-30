# %% imports and setting up
from os import chdir, getcwd, path

# setting working directory to file location  (!!NEW!!) 
abspath = path.abspath(__file__)  # path to file
dname = path.dirname(abspath)  # path to directory containing the file
chdir(dname)
print("Current working directory: ", getcwd())

import pandas as pd
import numpy as np
import util_data_loader as dl
import util_data_cleaning as clean
import optimisation.opt_basic_schedule as optimiser

print("--- BEGINNING PIPELINE ---")

# %% forecast: load in data for forecast
powr = dl.convert_tsf_to_dataframe("rawdata/nov_data.tsf")[0]
powr = clean.extract_data(powr)  # clean and reshape

# %% forecast: run the model
print("--- FORECASTING ---")
num_features = powr.shape[1] - 1 # remove one to account for the timestamp col
horizons = [1, 4, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1]
preds = np.zeros((2880, num_features))
for col in range(num_features):
    horizon = horizons[col]
    offset = pd.Timedelta(minutes = 15 * horizon)
    preds[:, col] = powr.iloc[-(2880+horizon):-horizon, col].copy() # Nov2020

# save predictions to dataframe
preds = pd.DataFrame(preds)
preds.columns = powr.columns[:-1]

# %% forecast: save data in output folder
preds_output = preds.T
preds_output.to_csv("final_output/nov_2020_predictions.csv", header=False)

# ALT: save a readable output csv (with timestamps)
# tmp = powr.timestamp[-(preds.shape[0]):].reset_index(drop = True) # need indexes to agree to add column
# preds["timestamp"] = tmp
# preds = preds[["timestamp"] + list(preds.columns[:-1])]  # reorder columns for readability
# preds.to_csv("results/nov_2020_predictions.csv")

print("--- FORECASTING COMPLETE ---")

# %% optimise: loop through problem instance files
print("--- OPTIMISING ---")
costs = []
files = ["phase2_instance_small_0.txt",
          "phase2_instance_small_1.txt",
          "phase2_instance_small_2.txt",
          "phase2_instance_small_3.txt",
          "phase2_instance_small_4.txt",
          "phase2_instance_large_0.txt",
          "phase2_instance_large_1.txt",
          "phase2_instance_large_2.txt",
          "phase2_instance_large_3.txt",
          "phase2_instance_large_4.txt"]

for file in files:
    print("-->>> Optimising "+str(file))
    cost, ppoi_list = optimiser.schedule_basic("rawdata/"+str(file),
                                               "final_output/nov_2020_predictions.csv",
                                               "rawdata/PRICE_AND_DEMAND_202011_VIC1.csv")
    
    # save cost
    costs.append(cost)
    
    # save ppoi list to a ppoi solution txt file
    ppoi_path = "final_output/"+str(file)
    f = open(ppoi_path, "x")
    for line in ppoi_list:
        f.write("%s\n" % line)
    f.close()
    print("-->>> Optimising "+str(file)+" complete and saved to a file!")

print("--- OPTIMISING COMPLETE ---")
print("--- ENDING PIPELINE ---")
