# Tipping Point Forecasting

This is the official implementation of the code from the paper "Tipping Point Forecasting in Non-Stationary Dynamics on Function Spaces."

## Short file descriptions:
Training models on pre-tipping data is performed by running files with names such as `nonstationary_lorenz/rno_1d_ode.py` or `ks/rnn_ks.py`. Then, the tipping point analysis is performed in separate analysis Jupyter notebooks or files (e.g., `DKW_analysis_[...].ipynb`).

## Requirements:
- Julia is required to run cloud cover experiments
- Training data for non-stationary Lorenz-63 can be generated with `nonstationary_lorenz/run_ode_solver.py`.
- Training data for non-stationary KS can be generated with `ks/generate_nonstationary_ks.py`.
- [MixedLayerModel.jl](https://github.com/claresinger/MixedLayerModel.jl) is used to generate training data for the cloud cover experiments
- PyTorch
- [Neural Operator Library](https://github.com/neuraloperator/neuraloperator)
- The airfoil dataset can be found [on Zenodo](https://zenodo.org/records/18098807)
