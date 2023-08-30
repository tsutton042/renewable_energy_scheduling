# Renewable energy scheduling

------------------------------------------------------------------------

### Troy Sutton

This repo contains all code from a toy group project undertaken as part of my degree. Data and some helper scripts were obtained from <a href="https://ieee-dataport.org/competitions/ieee-cis-technical-challenge-predictoptimize-renewable-energy-scheduling"> Competition page </a>

The repository has been edited such that all code in the `forecasting` folder is written by me. `util_data_cleaning.py`, `util_MASE.py` and `pipeline.py` were predominately written by me (`util_data_loader.py` is primarily provided code and code written by one of my team mates and is widely used throughout the repo). The files in the `optimisation` folder were not written by me, but are left in with some small edits for contextual purposes. 

`forecasting` contains an LSTM training script. Despite a lot of effort to get this model working, we ended up using a naive model (ie just using the data from previous timesteps) for each time series in the final design as it performed the best. However, the trained LSTM is still included.

Some of the data is linked or extraction method is shown in the data description document. 



