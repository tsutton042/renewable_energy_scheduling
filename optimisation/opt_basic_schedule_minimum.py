# %% imports and setting up
from os import chdir as setwd 
from os import getcwd as getcwd

# setting working directory
from config import file_dir
setwd(file_dir)
print("Current working directory: ", getcwd())

# import data science libraries and scripts
import pandas as pd
import gurobipy as gp
from util_data_loader import read_ppoi
from util_data_loader import read_aemo
import gurobipy as gp
    
# %% Define input data
# read in data from ppoi file(s)
(counts, 
 building_instance, 
 solar_instance,
 battery_instance,
 recurring_activity_instance, 
 once_activity_instance) = read_ppoi("rawdata/phase2_instance_small_1.txt")

# ppoi preprocessing: match index of buildings and solar panels
for i in range(len(solar_instance)):
    solar_instance.loc[i, 'solar_id'] = solar_instance.loc[i, 'building_id']
    
# read in data from forecasting
forecast = pd.read_csv("forecasting/saved_forecasts/forecast_naive.csv", header=None, index_col=0)

# read in AEMO wholesale electricity data
aemo = read_aemo("rawdata/PRICE_AND_DEMAND_202011_VIC1.csv")

# %% Process input data for ease of Gurobi modelling
# sets and indices
timestep = 30*24*4  # 15-minutely time periods for Nov2020
weekday_start = [96, 768, 1440, 2112, 2784]

timestep_r = []  # valid timesteps for recurring activities
for t in weekday_start:
    if t != 2784:
        # full weeks
        intermediate_t = t
        for i in range(5):
            timestep_r += list(range(intermediate_t+36,intermediate_t+68+1))
            intermediate_t += 96
    elif t == 2784:
        # the last week of nov2020 only has the monday
        timestep_r += list(range(t+36,t+68+1))

solar_building = building_instance['building_id'].tolist()  # building_id and solar_id
recurring = recurring_activity_instance['activity_id'].tolist()
room_size = ['S', 'L']

# parameters (fyi: r=recur, a=once-off)
build_load = forecast.iloc[:6,] # param
build_load.index = list(solar_instance['solar_id'])  # solar_id or building_id doesn't matter
build_load.columns = [x for x in range(timestep)]
pv_load = forecast.iloc[6:]  # param
pv_load.index = list(solar_instance['solar_id'])  # solar_id or building_id doesn't matter
pv_load.columns = [x for x in range(timestep)]
base_load = build_load - pv_load
base_load = list(base_load.sum(axis=0))

load_per_time_r = recurring_activity_instance['no_of_rooms']*recurring_activity_instance['duration']

room_small_total = building_instance.loc[:,'no_small_rooms'].sum()
room_large_total = building_instance.loc[:,'no_large_rooms'].sum()
rooms_needed_r = recurring_activity_instance.loc[:,['activity_id', 'no_of_rooms','room_size']]
rooms_needed_r['no_of_rooms_S'] = rooms_needed_r[rooms_needed_r['room_size'] == 'S'].loc[:,'no_of_rooms']
rooms_needed_r['no_of_rooms_L'] = rooms_needed_r[rooms_needed_r['room_size'] == 'L'].loc[:,'no_of_rooms']
rooms_needed_r['no_of_rooms_S'] = rooms_needed_r['no_of_rooms_S'].fillna(0)
rooms_needed_r['no_of_rooms_L'] = rooms_needed_r['no_of_rooms_L'].fillna(0)

dur_r = recurring_activity_instance.loc[:, ['activity_id','duration']]
dur_r.index = list(dur_r['activity_id'])
dur_r = dur_r['duration']
total_duration_r = recurring_activity_instance.loc[:,'duration'].sum()

# %% Model development: create model and decision variables
m = gp.Model('basic_schedule')
sched_recur = m.addVars(recurring, timestep, vtype = gp.GRB.BINARY, name = "sched_recur")

# %% Model development: define constraints
# cumulative load: the total cumulative load must always be positive (no feed-in)
const_load = m.addConstrs(gp.quicksum(base_load[t] + (load_per_time_r[r]*sched_recur[r,t]) 
                                      for r in recurring) >= 0 
                          for t in range(timestep))

# recurring activity precedence: activities rj must be scheduled before activities rk


# recurring weekly: establish relationship between the same recurring activity across weeks
const_weekly_r = m.addConstrs(sched_recur[r,t] == sched_recur[r,t%672] for r in recurring for t in range(timestep))

# recurring activity allocation: all recurring activities must be scheduled
const_allocate_all_r = m.addConstr(gp.quicksum(sched_recur[r,t] for r in recurring for t in timestep_r) == total_duration_r)

# activity duration: establish relationship between activity start time and end times
const_duration_r = m.addConstrs(gp.quicksum(sched_recur[r,t+d] for d in range(dur_r[r])) == dur_r[r] for r in recurring for t in timestep_r)

# space: cannot schedule more classes than the available space (room) for them
const_space_S = m.addConstrs(gp.quicksum(sched_recur[r,t] * rooms_needed_r.loc[r,'no_of_rooms_S'] 
                                         for r in recurring) <= room_small_total 
                             for t in range(timestep))

const_space_L = m.addConstrs(gp.quicksum(sched_recur[r,t] * rooms_needed_r.loc[r,'no_of_rooms_L'] 
                                         for r in recurring) <= room_large_total 
                             for t in range(timestep))


# %% Model development: define objective function
obj_total_cost = gp.quicksum((0.25 * base_load[t] * aemo.loc[t,'electricity'])/1000 for t in range(timestep)) 
base_load

# - objective function!

m.setObjective(obj_total_cost)

# - solve
m.optimize()





