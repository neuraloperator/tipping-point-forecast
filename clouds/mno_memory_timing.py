import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import pickle
from julia.api import Julia
from sklearn.preprocessing import MinMaxScaler
import string

from mno_clouds import *

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    sub = 1 # temporal subsampling rate
    T = 128 # input last T time steps and output next T

    experiment_name = 'PATH/TO/MODEL'

    model_in_dim = 5
    out_dim = 5

    # Benchmark different hyperparameters
    #model = FNO1d(model_in_dim, out_dim, 64, 128).to(device).float()

    n_iter = 100

    ############################################################
    #### Preprocessing
    ############################################################
    # Data is of the shape (number of trajectories, number of samples, grid size)
    data = np.load("PATH/TO/DATA.npy")
    extra_calibration_data = np.load("PATH/TO/EXTRA/CALIBRATION/DATA/IF/NEEDED.npy")

    dt = 1.0 / 365 # years
    dt_sec = dt * 3600 * 24 * 365 # seconds
    T1 = 0.0 # years
    T2 = 20.0 # years
    data = data[:,:-1] # Make shape easier to divide
    extra_calibration_data = extra_calibration_data[:,:-1]

    num_traj = data.shape[0]
    traj_len = data.shape[1]

    train_trajectories = 50
    data_test_trajectories = 50
    test_trajectories = len(data) + len(extra_calibration_data) - train_trajectories

    # Min-max scaling
    scaler = MinMaxScaler()
    scaler.fit(np.reshape(data[:train_trajectories], (train_trajectories * traj_len, in_dim))) # fit on training set
    data = scaler.transform(np.reshape(data, (num_traj * traj_len, in_dim)))
    data = np.reshape(data, (num_traj, traj_len, in_dim))

    extra_calibration_data = scaler.transform(np.reshape(extra_calibration_data, (-1, in_dim)))
    extra_calibration_data = np.reshape(extra_calibration_data, (-1, traj_len, in_dim))

    # Train-test split
    test = data[-data_test_trajectories:]
    test = np.concatenate([extra_calibration_data, test], axis=0)

    # Make testing data easier to reshape
    test_idx = round_down(test.shape[1], T)
    test = test[:,:test_idx]

    test = np.reshape(test, (test_trajectories, -1, T, in_dim))

    num_val = 145
    num_test = 5

    val = torch.tensor(test[:num_val])
    test = torch.tensor(test[-num_test:])

    ############################################################
    #### Benchmarking
    ############################################################
    # Model memory and size: https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs # in bytes

    print("Model memory footprint: " + str(mem) + " bytes")
    print()

    num_steps = test.shape[1]
    print("Bechmark number of steps:", num_steps)
    print()
    times = []
    for i in range(n_iter + 1):
        model_input = torch.randn(1, T, model_in_dim).to(device)

        # Time: https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        
        out = model(model_input)
        for _ in range(num_steps - 1):
            out = model(out)

        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()

        if i > 0: # exclude first run
            times.append(start.elapsed_time(end))

    print(f"Mean inference time over {n_iter} iterations: {sum(times) / len(times)} ms")
