import torch.nn.functional as F
from timeit import default_timer
from utilities_lorenz import *
from tqdm import tqdm
import pickle
import sys
import random

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
    sub = 5 # temporal subsampling rate
    T = 64 # each interval is T time-steps long
    n_intervals = 5 # number of intervals in each training example
    dim = 3 # dimensionality of ODE
    
    batch_size = 256

    tipping_point = 0

    eval_steps = [1,2,4,8,16,32,64,128]

    ################################################################
    # read data
    ################################################################

    # Data is of the shape (number of samples, grid size)
    with open("PATH/TO/DATA.p", "rb") as file:
        data_dict = pickle.load(file)
    
    tipping_point = 0 # time of tipping point -- data-dependent!

    dt = data_dict['dt']
    T1 = data_dict['T1']
    T2 = data_dict['T2']
    data = data_dict['data'][:,::sub]
    data = data[:,:-1] # Make shape easier to divide
    
    test_trajectories = 5

    test = data[-test_trajectories:]

    test_idx = round_down(get_index_from_time(T1, T2, tipping_point, test.shape[1]), T * dim)
    test = test[:,:test_idx]

    test = np.reshape(test, (test_trajectories, -1, T, dim))

    # Generate training pairs
    x_test = []
    y_test = []

    for traj in test:
        for i in range(test.shape[1] - n_intervals - 1):
            x_test.append(traj[i:i + n_intervals])
            y_test.append(traj[i + n_intervals])

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    x_test = torch.tensor(x_test).float()
    y_test = torch.tensor(y_test).float()

    # Multi-step eval
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
    multi_step_y_test_list = [torch.tensor(arr).float() for arr in multi_step_y_test_list]
    multi_step_x_test_list.insert(0, x_test)
    multi_step_y_test_list.insert(0, y_test)

    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    multi_step_test_loaders = [torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=batch_size, shuffle=False) for x,y in zip(multi_step_x_test_list, multi_step_y_test_list)]

    print("Preprocessing complete")
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)
    print()

    # model
    from neuralop.models import RNO
    model = torch.load('PATH/TO/MODEL').cuda()
    print("Parameters:", model.count_params())
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

                x_in = x.permute(0, 1, 3, 2)
                out = model.predict(x_in, num_steps)[:,-1]
                out = out.permute(0, 2, 1)

                test_l2_list[n] += myloss(torch.reshape(out, (-1, T * dim)), torch.reshape(y, (-1, T * dim))).item()

    for n in range(len(multi_step_test_loaders)):
        test_l2_list[n] /= ntest_list[n]

    print("Evaluation results:")
    for n in range(len(test_l2_list)):
        print("Steps:", eval_steps[n], "    Rel. L2:", test_l2_list[n])