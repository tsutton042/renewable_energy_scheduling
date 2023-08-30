# %% imports and setting up
import pandas as pd
import numpy as np
import gurobipy as gp
import csv
from util_data_loader import read_ppoi
from util_data_loader import read_aemo

# %% JK's modular optimiser - optimise one and output a Gurobi model
def schedule_basic(ppoi_filepath, forecast_filepath, aemo_filepath):
    """
    Given building and pv load predictions for Nov2020, generate a basic 
    schedule optimised for minimum cost using a Gurobi model. A basic schedule
    only considers recurring activities.
    
    Parameters
    ----------
    ppoi_problem : string
        Path to a .txt file that contains a problem instance in ppoi
        (predict-plus-optimise-instance) format.
    
    forecast_output_filepath : string
        Path to a .csv file that contains the forecasted energy consumption and 
        production for 6 buildings and 6 solar panels. There are 2880 
        time periods.
        
    aemo_filepath : string
        Path to a .csv file containing the electricity price for the 2880 time 
        periods of month November 2020.
        
    Returns
    -------
    A tuple, consisting of:
        
        cost: int
            The cost of the optimised schedule
        
        ppoi_list: Python list
            A list representation of a ppoi file, where each element in the list
            corresponds to a line of that ppoi solution file
        
    Code sections (pseudocode-ish)
    -------
    1. Read in datasets for input
    2. Define sets and parameters for models
    3. Optimise for the objective function's maxpower
    4. Optimise for the main objective function
    5. Gather results and return ppoi_list
        
    """
    
    # 1. Read in datasets for input
    # read in data from ppoi file(s)
    (counts, 
     building_instance, 
     solar_instance,
     battery_instance,
     recurring_activity_instance, 
     once_activity_instance) = read_ppoi(ppoi_filepath)
        
    # read in data from forecasting, and calculate net load i at time t
    forecast = pd.read_csv(forecast_filepath, header=None, index_col=0)

    building_consumption = [[x] for x in forecast.index[0:6]]
    solar_production = [[x] for x in forecast.index[6:12]]
    for i in range(0,6):
        building_consumption[i] += forecast.iloc[i].to_list()
        solar_production[i] += forecast.iloc[i+6].to_list()

    netload = []
    for t in range(1,2881):
        sum_building_consumption = 0
        sum_solar_production = 0
        for i in range(6):
            sum_building_consumption += float(building_consumption[i][t])
            sum_solar_production += float(solar_production[i][t])
        netload.append(sum_building_consumption-sum_solar_production)

    # read in AEMO wholesale electricity data
    aemo = read_aemo(aemo_filepath)
    price_elec = aemo['electricity'].rename('RRP')


    # 2. DEFINE SETS AND PARAMETERS FOR MODELS
    nbulding = 6                                            # no. of buildings
    nsolar = 6                                              # no. of solar panels
    nperiod = 2880                                          # no. of time steps
    nbattery = battery_instance.shape[0]                    # no. of batteries
    nreccur = recurring_activity_instance.shape[0]          # no. of recurring acts
    nonce = once_activity_instance.shape[0]                 # no. of once-off acts
    nsmallroom = sum(building_instance['no_small_rooms'])   # no. of small rooms
    nlargeroom = sum(building_instance['no_large_rooms'])   # no. of large rooms
       
    w = 672
    s = 132
    d = 96
       
    # valid_t includes all the valid time 
    # week1 valid time 
    valid_t = list(range(s,s+32)) + list(range(s+d,s+d+32)) + list(range(s+2*d,s+2*d+32)) + list(range(s+3*d,s+3*d+32)) + list(range(s+4*d,s+4*d+32))
       
    # week 2 valid time
    for i in range(7,12):
        valid_t = valid_t + list(range(s+i*d,s+i*d+32)) 
       
    # week 3 valid time
    for i in range(14,19):
        valid_t = valid_t + list(range(s+i*d,s+i*d+32)) 
       
    # week 4 valid time
    for i in range(21,26):
        valid_t = valid_t + list(range(s+i*d,s+i*d+32)) 
       
    # week 5 valid time
    i = 28
    valid_t = valid_t + list(range(s+i*d,s+i*d+32)) 
       
    # non valid time
    nonvalid_t = set(list(range(nperiod))) - set(valid_t)
    nonvalid_t = list(nonvalid_t)
       
    # there are 5 weeks in Nov 2020 that contains weekdays
    # list all the week time range 
    week1_t = list(range(d,d+w))
    week2_t = list(range(d+w,d+w*2))
    week3_t = list(range(d+w*2,d+w*3))
    week4_t = list(range(d+w*3,d+w*4))
    week5_t = list(range(d+w*4,2880))

    # create day of the week for each time stamp in November 2020
    # starts with Sunday followed by 4 weeks, and ends on Monday
    e = []
    for i in range(1,8):
        e += [i for _ in range(96)]

    dayofweek = [7 for _ in range(96)] + e*4 + [1 for _ in range(96)]

    # create time of day
    # consist of 96 t(s), start with index 0 ends with 95

    t = [i for i in range(96)]


    optimData = {'Base_load' : netload ,
                'price' : price_elec,
                'dayofweek' : dayofweek,
                'timeofday' : t*30
                }
    optimData = pd.DataFrame(optimData)
    base_load = netload

    # 3. OPTIMISE FOR OBJECTIVE FUNCTION'S MAXPOWER
    m = gp.Model("MonashScheduling")
    m.setParam('TimeLimit', 3*60)


    ## variables
    net_power = m.addVars(nperiod, vtype = gp.GRB.CONTINUOUS, lb = 0,  name = 'net power') # net power at each time t 
    act_power = m.addVars(nperiod, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'act power') # power from acts at each time t
    bat_level = m.addVars(nbattery, nperiod, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'bat level') # battery level status at each time t
    ract_begin = m.addVars(nreccur, nperiod, vtype = gp.GRB.BINARY, name = 'recurring activity begin') # 1 if activity begins 
    ract_ongoing = m.addVars(nreccur, nperiod, vtype = gp.GRB.BINARY, lb = 0, name = 'recurring activity ongoing') # 1 if activity is ongoing
    bat_charge = m.addVars(nbattery, nperiod, vtype = gp.GRB.BINARY, name = 'battery charge')
    bat_discharge = m.addVars(nbattery, nperiod, vtype = gp.GRB.BINARY, name = 'battery discharge')

    maxpower = m.addVar(name = "max power") # max power


    ## constraints
    # -- battery -- 
    for bat in range(nbattery):

        # each battery at start of the month is full
        m.addConstr(bat_level[bat,0] == battery_instance['capacity_kwh'][bat])

        # each battery has to be less than the capacity
        for t in range(nperiod):
            m.addConstr(bat_level[bat,t] <= battery_instance['capacity_kwh'][bat])

    # -- activity --
    for act in range(nreccur):

        # each activity happend only once a week 
        m.addConstr(gp.quicksum(ract_begin[act,t] for t in week1_t) == 1)
        m.addConstr(gp.quicksum(ract_begin[act,t] for t in week2_t) == 1)
        m.addConstr(gp.quicksum(ract_begin[act,t] for t in week3_t) == 1)
        m.addConstr(gp.quicksum(ract_begin[act,t] for t in week4_t) == 1)

        # precendece activity must happend before 
        if len(recurring_activity_instance['precedence_list'][act]) > 0 : 
            prec_list = recurring_activity_instance['precedence_list'][act]
            for prec_act in prec_list: 
                prec_act = int(prec_act)
                m.addConstr(gp.quicksum(ract_begin[act, t] * t for t in week1_t) >= gp.quicksum(ract_begin[prec_act, t] * t for t in week1_t))

        # same time every week 
        for num in range(768, nperiod):
            m.addConstr(ract_begin[act, num] == ract_begin[act, num - 672])
            m.addConstr(ract_ongoing[act, num] == ract_ongoing[act, num - 672])

        # activity should last as long as their period 
        for t in range(nperiod-recurring_activity_instance['duration'][act]):
            for dur in range(recurring_activity_instance['duration'][act]):
                m.addConstr(ract_begin[act,t] <= ract_ongoing[act, t + dur])

        # activity is only between monday to friday 9am-5pm
        for t in nonvalid_t:
            m.addConstr(ract_ongoing[act,t] == 0)
            m.addConstr(ract_begin[act,t] == 0)

    for t in range(nperiod):
        # no of small room use in time t for all acitivity is less than equal to number of small rooms in all buildings
        m.addConstr(gp.quicksum(ract_ongoing[act,t] * recurring_activity_instance['no_small_rooms'][act] for act in range(nreccur))
                    <= nsmallroom )

        # no of large room use in time t for all acitivity is less than equal to number of large rooms in all buildings
        m.addConstr(gp.quicksum(ract_ongoing[act,t] * recurring_activity_instance['no_large_rooms'][act] for act in range(nreccur))
                    <= nlargeroom )

    # -- max power -- 
    m.addConstr(maxpower == gp.max_(net_power[t] for t in range(nperiod)))

    # -- power calculation --
    # power from all activity that is happening at each time t 
    for t in range(nperiod):
         m.addConstr(act_power[t] == gp.quicksum(ract_ongoing[act,t] * recurring_activity_instance['load_kwh'][act] 
                                                * (recurring_activity_instance['no_small_rooms'][act] + recurring_activity_instance['no_large_rooms'][act])
                                                for act in range(nreccur)) )

    # -- battery level calculation -- 
    for t in range(1,nperiod):
        for bat in range(nbattery):
           m.addConstr(bat_level[bat,t] == bat_level[bat,t-1] + (bat_charge[bat,t] * battery_instance['max_power_kwh'][bat] * battery_instance['efficiency'][bat] ** -0.5) 
                        - (bat_discharge[bat,t] * battery_instance['max_power_kwh'][bat] * battery_instance['efficiency'][bat] ** 0.5))

    # -- net power calculation -- 
    for t in range(nperiod):
        m.addConstr(net_power[t] == base_load[t] + act_power[t] + gp.quicksum(bat_charge[bat,t] * battery_instance['max_power_kwh'][bat] * battery_instance['efficiency'][bat] ** -0.5 
                       - bat_discharge[bat,t] * battery_instance['max_power_kwh'][bat] * battery_instance['efficiency'][bat] ** 0.5 for bat in range(nbattery)))


    ## objective
    m.setObjective(maxpower)

    ## optimize 
    m = m.presolve()
    m = m.relax()
    m.optimize()

    obj = m.getObjective()
    max_limit = obj.getValue()  # jackpot. this is used for the next part in the objective function


    # 4. Optimise for the main objectve function

    m = gp.Model("MonashScheduling")
    m.setParam('TimeLimit', 3*60)

    ## variables
    net_power = m.addVars(nperiod, vtype = gp.GRB.CONTINUOUS, lb = 0,  name = 'net power') # net power at each time t 
    act_power = m.addVars(nperiod, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'act power') # power from acts at each time t
    bat_level = m.addVars(nbattery, nperiod, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'bat level') # battery level status at each time t
    ract_begin = m.addVars(nreccur, nperiod, vtype = gp.GRB.BINARY, name = 'recurring activity begin') # 1 if activity begins 
    ract_ongoing = m.addVars(nreccur, nperiod, vtype = gp.GRB.BINARY, lb = 0, name = 'recurring activity ongoing') # 1 if activity is ongoing
    bat_charge = m.addVars(nbattery, nperiod, vtype = gp.GRB.BINARY, name = 'battery charge')
    bat_discharge = m.addVars(nbattery, nperiod, vtype = gp.GRB.BINARY, name = 'battery discharge')

    maxpower = m.addVar(name = "max power") # max power


    ## constraints

    # -- battery -- 


    for bat in range(nbattery):

        # each battery at start of the month is full
        m.addConstr(bat_level[bat,0] == battery_instance['capacity_kwh'][bat])

        # each battery has to be less than the capacity
        for t in range(nperiod):
            m.addConstr(bat_level[bat,t] <= battery_instance['capacity_kwh'][bat])

    # -- activity --

    for act in range(nreccur):

        # each activity happend only once a week 
        m.addConstr(gp.quicksum(ract_begin[act,t] for t in week1_t) == 1)
        m.addConstr(gp.quicksum(ract_begin[act,t] for t in week2_t) == 1)
        m.addConstr(gp.quicksum(ract_begin[act,t] for t in week3_t) == 1)
        m.addConstr(gp.quicksum(ract_begin[act,t] for t in week4_t) == 1)

        # precedence activity must happend before 
        if len(recurring_activity_instance['precedence_list'][act]) > 0 : 
            prec_list = recurring_activity_instance['precedence_list'][act]
            for prec_act in prec_list: 
                prec_act = int(prec_act)
                m.addConstr(gp.quicksum(ract_begin[act, t] * t for t in week1_t) >= gp.quicksum(ract_begin[prec_act, t] * t for t in week1_t))

        # same time every week 
        for num in range(768, nperiod):
            m.addConstr(ract_begin[act, num] == ract_begin[act, num - 672])
            m.addConstr(ract_ongoing[act, num] == ract_ongoing[act, num - 672])

        # activity should last as long as their period 
        for t in range(nperiod-recurring_activity_instance['duration'][act]):
            for dur in range(recurring_activity_instance['duration'][act]):
                m.addConstr(ract_begin[act,t] <= ract_ongoing[act, t + dur])

        # activity is only between monday to friday 9am-5pm
        for t in nonvalid_t:
            m.addConstr(ract_ongoing[act,t] == 0)
            m.addConstr(ract_begin[act,t] == 0)


    for t in range(nperiod):
        # no of small room use in time t for all acitivity is less than equal to number of small rooms in all buildings
        m.addConstr(gp.quicksum(ract_ongoing[act,t] * recurring_activity_instance['no_small_rooms'][act] for act in range(nreccur))
                    <= nsmallroom )

        # no of large room use in time t for all acitivity is less than equal to number of large rooms in all buildings
        m.addConstr(gp.quicksum(ract_ongoing[act,t] * recurring_activity_instance['no_large_rooms'][act] for act in range(nreccur))
                    <= nlargeroom )


    # -- max power -- 
    m.addConstr(maxpower == gp.max_(net_power[t] for t in range(nperiod)))


    # -- power calculation --

    # power from all activity that is happening at each time t 
    for t in range(nperiod):
         m.addConstr(act_power[t] == gp.quicksum(ract_ongoing[act,t] * recurring_activity_instance['load_kwh'][act] 
                                                * (recurring_activity_instance['no_small_rooms'][act] + recurring_activity_instance['no_large_rooms'][act])
                                                for act in range(nreccur)) )

    # -- battery level calculation -- 
    for t in range(1,nperiod):
        for bat in range(nbattery):
           m.addConstr(bat_level[bat,t] == bat_level[bat,t-1] + (bat_charge[bat,t] * battery_instance['max_power_kwh'][bat] * battery_instance['efficiency'][bat] ** -0.5) 
                        - (bat_discharge[bat,t] * battery_instance['max_power_kwh'][bat] * battery_instance['efficiency'][bat] ** 0.5))


    # -- net power calculation -- 
    for t in range(nperiod):
        m.addConstr(net_power[t] == base_load[t] + act_power[t] + gp.quicksum(bat_charge[bat,t] * battery_instance['max_power_kwh'][bat] * battery_instance['efficiency'][bat] ** -0.5 
                       - bat_discharge[bat,t] * battery_instance['max_power_kwh'][bat] * battery_instance['efficiency'][bat] ** 0.5 for bat in range(nbattery)))


    ## objective
    m.setObjective(gp.quicksum(((0.25)*net_power[t]*price_elec[t])/1000 for t in range(nperiod)) + 0.005*max_limit**2)


    ## optimise and get minimum cost
    m.optimize()
    cost = m.getObjective().getValue()
    
    
    # 5. Gather results and return
    ppoi_list = []

    # ppoi header
    line_header = "ppoi" \
                   + " " + str(counts['building']) \
                   + " " + str(counts['solar']) \
                   + " " + str(counts['battery']) \
                   + " " + str(counts['recur_act']) \
                   + " " + str(counts['once_act'])
        
    ppoi_list.append(line_header)

    # sched line
    line_sched = "sched " + str(counts['recur_act']) + " 0"  # 0 once-off activities
    ppoi_list.append(line_sched)

    # recurring activity lines
    for t in range(nperiod):
        for r in range(nreccur):
            start = ract_begin[r,t].x  # 1=recur.act. r starts at time t
            if start == 1.0:
                no_of_rooms = recurring_activity_instance.iloc[r,2]  # 'no_of_rooms'
                line = "r "+str(r)+" "+str(t)+" "+str(no_of_rooms)  # currently missing which rooms...
                ppoi_list.append(line)


    # battery lines
    for t in range(nperiod):
        for b in range(nbattery):
            charge = bat_charge[b,t].x  # 0=hold/discharge, 1=charge
            discharge = bat_discharge[b,t].x  # 0=hold/charge, 1=discharge
            
            # note that ppoi representation is different (0=charge, 2=discharge)
            # that's why we save it as a different value
            if charge == 1.0:
                line = "b "+str(b)+" "+str(t)+" 0"
                ppoi_list.append(line)
            if discharge == 1.0:
                line = "b "+str(b)+" "+str(t)+" 2"
                ppoi_list.append(line)

    return (cost, ppoi_list)
