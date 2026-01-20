import torch.nn.functional as F
import sys
from timeit import default_timer

from sklearn.preprocessing import MinMaxScaler

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../nonstationary_lorenz"))
from utilities_lorenz import *
import numpy as np

from tqdm import tqdm
import random

from gru_clouds import GRU_net

torch.manual_seed(0)
np.random.seed(0)

def get_index_from_time(T_min, T_max, query_time, arr_size):
    """
    Given array with data spanning from time T_min to T_max, return the index in the array
    associated with query_time.
    """
    time_range = T_max - T_min
    return round(arr_size * (query_time - T_min) / time_range)

def round_down(num, divisor): # rounds `num` down to nearest multiple of `divisor`
    return num - (num % divisor)

if __name__ == '__main__':
    ################################################################
    #  configurations
    ################################################################
    eval_steps = [1,2,4,8,16]
    
    sub = 1 # temporal subsampling rate
    T = 128 # input last T time steps and output next T
    n_intervals = 5 # number of intervals in each training example
    batch_size = 64
    
    in_dim = 5 # dimensionality of ODE + dimension of input parameter
    out_dim = 5

    ################################################################
    # read data
    ################################################################
    # Data is of the shape (number of trajectories, number of samples, grid size)
    data = np.load("PATH/TO/DATA.npy")
    extra_calibration_data = np.load("PATH/TO/EXTRA/CALIBRATION/DATA/IF/NEEDED.npy")

    tipping_point = 10.0 # time of tipping point -- data-dependent!

    dt = 1.0 / 365 # years
    dt_sec = dt * 3600 * 24 * 365 # seconds
    T1 = 0.0 # years
    T2 = 20.0 # years
    data = data[:,:-1] # Make shape easier to divide
    extra_calibration_data = extra_calibration_data[:,:-1]

    data = data[..., :-1] # remove forcing
    extra_calibration_data = extra_calibration_data[..., :-1] # remove forcing

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

    # Tipping point
    test_idx = round_down(get_index_from_time(T1, T2, tipping_point, test.shape[1]), T)
    test = test[:,:test_idx]
    test = np.reshape(test, (test_trajectories, -1, T, in_dim))
    #### END NEW ####

    x_test = []
    y_test = []

    for traj in test:
        for i in range(test.shape[1] - n_intervals - 1):
            x_test.append(traj[i:i + n_intervals])
            y_test.append(traj[i + n_intervals])
    
    x_test = np.array(x_test)
    y_test = np.array(y_test)[...,:out_dim]

    x_test = torch.tensor(x_test).float()
    y_test = torch.tensor(y_test).float()

    multi_step_x_test_list = []
    multi_step_y_test_list = []

    for n in eval_steps[1:]:
        n_xlist = []
        n_ylist = []
        for traj in test:
            for i in range(test.shape[1] - n_intervals - n):
                n_xlist.append(traj[i:i + n_intervals])
                n_ylist.append(traj[i + n_intervals + n - 1])
        multi_step_x_test_list.append(np.array(n_xlist))
        multi_step_y_test_list.append(np.array(n_ylist))

    multi_step_x_test_list = [torch.tensor(arr).float() for arr in multi_step_x_test_list]
    multi_step_y_test_list = [torch.tensor(arr).float()[...,:out_dim] for arr in multi_step_y_test_list]
    multi_step_x_test_list.insert(0, x_test)
    multi_step_y_test_list.insert(0, y_test)

    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    multi_step_test_loaders = [torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=batch_size, shuffle=False) for x,y in zip(multi_step_x_test_list, multi_step_y_test_list)]

    print("Preprocessing complete")
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)
    print()

    # model
    experiment_name = 'PATH/TO/MODEL'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(experiment_name, map_location=device)
    print("Parameters:", model.count_params())
    print("Experiment name:", experiment_name)
    print()

    ################################################################
    # evaluation
    ################################################################
    ntest = x_test.shape[0]
    ntest_list = [arr.shape[0] for arr in multi_step_x_test_list]

    myloss = LpLoss(size_average=False)

    test_l2_list = [0.0 for i in range(len(multi_step_test_loaders))]
    with torch.no_grad():
        for n in range(len(multi_step_test_loaders)):
            num_steps = eval_steps[n]
            print("Current num. steps:", num_steps)
            for x, y in multi_step_test_loaders[n]:
                x, y = x.cuda(), y.cuda()
                x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

                out = model.predict(x, num_steps)[:,-1]

                test_l2_list[n] += myloss(torch.reshape(out, (-1, T * out_dim)), torch.reshape(y, (-1, T * out_dim))).item()

    for n in range(len(multi_step_test_loaders)):
        test_l2_list[n] /= ntest_list[n]

    print("Evaluation results:")
    for n in range(len(test_l2_list)):
        print("Steps:", eval_steps[n], "    Rel. L2:", test_l2_list[n])