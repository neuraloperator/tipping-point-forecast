# Tipping Point Forecasting

This is the official implementation of the code from the paper "Tipping Point Forecasting in Non-Stationary Dynamics on Function Spaces."

## Short file descriptions:
Training models on pre-tipping data is performed by running files with names such as `nonstationary_lorenz/rno_1d_ode.py` or `ks/rnn_ks.py`. Then, the tipping point analysis is performed in Jupyter notebooks with names `DKW_analysis_[...].ipynb`.

## Requirements:
- Julia is required to run cloud cover experiments
- [MixedLayerModel.jl](https://github.com/claresinger/MixedLayerModel.jl) is used to generate training data for the cloud cover experiments
- PyTorch
