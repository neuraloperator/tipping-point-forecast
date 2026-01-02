import torch.nn.functional as F
from timeit import default_timer
from utilities_ks import *
from tqdm import tqdm
import pickle
import sys
import random

from neuralop.models import FNO
from neuralop.utils import count_model_params

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
    ntest = 40

    tipping_data_split_prop = 0.6 # proportion of trajectory at which to split the data for pre/post tipping

    eval_steps = [1,2,4,8,12,16,20]

    dim = 1

    batch_size = 64

    sub = 4 # spatial subsample
    T = 16 # input last T time steps and output next T
    n_intervals = 1

    model_path = 'PATH/TO/MODEL'

    ################################################################
    # data and preprocessing
    ################################################################

    t1 = default_timer()
    data = np.load('PATH/TO/DATA.npy')[:, ::sub]
    data = torch.tensor(data)

    n_time = data.shape[2]
    n_x = data.shape[1]

    test = data[-ntest:]

    tipping_idx = round_down(int(tipping_data_split_prop * n_time), T)
    test = test[:,:,:tipping_idx]

    test = torch.stack(torch.split(test, T, dim=-1)).permute(1,0,3,2).unsqueeze(-1)

    # Generate testing pairs
    x_test = []
    y_test = []

    for traj in test:
        for i in range(test.shape[1] - n_intervals - 1):
            x_test.append(traj[i:i + n_intervals])
            y_test.append(traj[i + n_intervals])
    
    x_test = torch.stack(x_test, dim=0).float()
    y_test = torch.stack(y_test, dim=0).float()

    multi_step_x_test_list = []
    multi_step_y_test_list = []

    for n in eval_steps[1:]:
        n_xlist = []
        n_ylist = []
        for traj in test:
            for i in range(test.shape[1] - n_intervals - n):
                n_xlist.append(traj[i:i + n_intervals])
                n_ylist.append(traj[i + n_intervals + n - 1])
        multi_step_x_test_list.append(torch.stack(n_xlist, dim=0).float())
        multi_step_y_test_list.append(torch.stack(n_ylist, dim=0).float())

    multi_step_x_test_list.insert(0, x_test)
    multi_step_y_test_list.insert(0, y_test)

    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False, drop_last=True)

    multi_step_test_loaders = [torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=batch_size, shuffle=False, drop_last=True) for x,y in zip(multi_step_x_test_list, multi_step_y_test_list)]

    print("Preprocessing complete")
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)
    print("multi-step test shapes:", [batch_size * len(loader) for loader in multi_step_test_loaders])
    device = torch.device('cuda')
    print()


    ################################################################
    # model
    ################################################################

    model = torch.load(model_path).cuda()
    print("Model:", model_path)
    print("Parameters:", count_model_params(model))
    print()

    ################################################################
    # evaluation
    ################################################################
    lploss = LpLoss(size_average=False)

    ntest_list = [arr.shape[0] for arr in multi_step_x_test_list]

    ### Evaluation
    model.eval()
    test_l2_list = [0.0 for i in range(len(multi_step_test_loaders))]
    with torch.no_grad():
        for n in range(len(multi_step_test_loaders)):
            num_steps = eval_steps[n]
            for x, y in multi_step_test_loaders[n]:
                x = x.to(device).view(batch_size, T, n_x, dim)
                y = y.to(device).view(batch_size, T, n_x, dim)

                x_perm = x.permute(0, 3, 1, 2)
                out = model(x_perm)
                out_reshaped = out.permute(0, 2, 3, 1).reshape(batch_size, T, n_x, dim)
                
                for i in range(num_steps - 1):
                    out_perm = out_reshaped.permute(0, 3, 1, 2)
                    out = model(out_perm)
                    out_reshaped = out.permute(0, 2, 3, 1).reshape(batch_size, T, n_x, dim)

                out = out_reshaped

                test_l2_list[n] += lploss(torch.reshape(out, (-1, n_x * T * dim)), torch.reshape(y, (-1, n_x * T * dim))).item()

    for n in range(len(multi_step_test_loaders)):
        test_l2_list[n] /= ntest_list[n]

    print()
    print("Evaluation results:")
    for n in range(len(test_l2_list)):
        print("Steps:", eval_steps[n], "    Rel. L2:", test_l2_list[n])