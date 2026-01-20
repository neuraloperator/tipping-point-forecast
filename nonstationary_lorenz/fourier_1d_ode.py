"""
Learns trajectory of ODE by using a 1D FNO performing FFT and IFFT in time.
"""

import torch.nn.functional as F
import numpy as np
from timeit import default_timer
from utilities_lorenz import *
from tqdm import tqdm
import pickle
import random

torch.manual_seed(0)
np.random.seed(0)

from neuralop.models import FNO

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
    T = 64 # input last T time steps and output next T
    dim = 3 # dimensionality of ODE

    batch_size = 64
    learning_rate = 0.001
    epochs = 50

    scheduler_step = 400
    scheduler_gamma = 0.5

    modes = 24
    width = 48

    train_up_to_tipping_point = True # if true, only trains model on data up to tipping point
    
    multi_step_fine_tuning = True
    fine_tuning_num_steps = [1,2,3,4,5,6,7,8,9,10,11,12]
    fine_tune_ep = 5
    fine_tune_scheduler_step = 4000
    fine_tuning_lr = 1e-4

    model_save_path = None # Update

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
    
    # Number of training/testing trajectories
    train_trajectories = 10
    test_trajectories = 5

    train = data[:train_trajectories]
    test = data[-test_trajectories:]

    if train_up_to_tipping_point:
        train_idx = round_down(get_index_from_time(T1, T2, tipping_point, train.shape[1]), T * dim)
        test_idx = round_down(get_index_from_time(T1, T2, tipping_point, test.shape[1]), T * dim)
        train = train[:,:train_idx]
        test = test[:,:test_idx]

    train = np.reshape(train, (train_trajectories, -1, T, dim))
    test = np.reshape(test, (test_trajectories, -1, T, dim))

    x_train = train[:,:-1]
    y_train = train[:,1:]
    x_test = test[:,:-1]
    y_test = test[:,1:]

    x_train = torch.tensor(np.reshape(x_train, (-1, T, dim))).float()
    y_train = torch.tensor(np.reshape(y_train, (-1, T, dim))).float()
    x_test = torch.tensor(np.reshape(x_test, (-1, T, dim))).float()
    y_test = torch.tensor(np.reshape(y_test, (-1, T, dim))).float()

    if multi_step_fine_tuning:
        multi_step_x_train_list = []
        multi_step_y_train_list = []
        multi_step_x_test_list = []
        multi_step_y_test_list = []

        for n in fine_tuning_num_steps[1:]:
            n_xlist = []
            n_ylist = []
            for traj in train:
                for i in range(train.shape[1] - 1 - n):
                    n_xlist.append(traj[i:i + 1])
                    n_ylist.append(traj[i + 1 + n - 1])
            multi_step_x_train_list.append(np.array(n_xlist))
            multi_step_y_train_list.append(np.array(n_ylist))

        for n in fine_tuning_num_steps[1:]:
            n_xlist = []
            n_ylist = []
            for traj in test:
                for i in range(test.shape[1] - 1 - n):
                    n_xlist.append(traj[i:i + 1])
                    n_ylist.append(traj[i + 1 + n - 1])
            multi_step_x_test_list.append(np.array(n_xlist))
            multi_step_y_test_list.append(np.array(n_ylist))

        multi_step_x_train_list = [torch.tensor(arr).float() for arr in multi_step_x_train_list]
        multi_step_y_train_list = [torch.tensor(arr).float() for arr in multi_step_y_train_list]
        multi_step_x_test_list = [torch.tensor(arr).float() for arr in multi_step_x_test_list]
        multi_step_y_test_list = [torch.tensor(arr).float() for arr in multi_step_y_test_list]
        multi_step_x_train_list.insert(0, x_train)
        multi_step_y_train_list.insert(0, y_train)
        multi_step_x_test_list.insert(0, x_test)
        multi_step_y_test_list.insert(0, y_test)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    if multi_step_fine_tuning:
        multi_step_train_loaders = [torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=batch_size, shuffle=True) for x,y in zip(multi_step_x_train_list, multi_step_y_train_list)]
        multi_step_test_loaders = [torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=batch_size, shuffle=False) for x,y in zip(multi_step_x_test_list, multi_step_y_test_list)]

        multi_step_list = []
        for i in range(len(multi_step_train_loaders)):
            multi_step_list += [i for j in range(len(multi_step_train_loaders[i]))]
        random.shuffle(multi_step_list)

    print("Preprocessing complete")
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)
    print()

    # model
    model = FNO(
        n_modes=(modes,),
        hidden_channels=width,
        in_channels=dim,
        out_channels=dim,
        n_layers=4,
        domain_padding=0.125
    ).cuda().float()
    print("Parameters:", count_params(model))
    print()

    ################################################################
    # training and evaluation
    ################################################################
    ntrain = x_train.shape[0]
    ntest = x_test.shape[0]
    iterations = epochs * (ntrain//batch_size)

    if multi_step_fine_tuning:
        ntrain_list = [arr.shape[0] for arr in multi_step_x_train_list]
        ntest_list = [arr.shape[0] for arr in multi_step_x_test_list]

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, scheduler_step, scheduler_gamma)

    ### Training
    myloss = LpLoss(size_average=False)
    print("Begin training:")
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0
        for x, y in tqdm(train_loader):
            x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            # Permute for FNO
            x_perm = x.permute(0, 2, 1)
            out_perm = model(x_perm)
            # Permute back
            out = out_perm.permute(0, 2, 1)

            mse = F.mse_loss(torch.reshape(out, (-1, T * dim)), torch.reshape(y, (-1, T * dim)), reduction='mean')
            l2 = myloss(torch.reshape(out, (-1, T * dim)), torch.reshape(y, (-1, T * dim)))
            l2.backward() # use the l2 relative loss

            optimizer.step()
            scheduler.step()
            train_mse += mse.item()
            train_l2 += l2.item()

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()

                # Permute for FNO
                x_perm = x.permute(0, 2, 1)
                out_perm = model(x_perm)
                out = out_perm.permute(0, 2, 1)
                test_l2 += myloss(torch.reshape(out, (-1, T * dim)), torch.reshape(y, (-1, T * dim))).item()

        train_mse /= len(train_loader)
        train_l2 /= ntrain
        test_l2 /= ntest

        t2 = default_timer()
        print("Epoch:", ep + 1, "Time:", t2-t1, "Train MSE:", train_mse, "Train L2:", train_l2, "Test L2:", test_l2)

    ### Multi-step fine tuning
    if multi_step_fine_tuning:
        print("Begin multi-step fine-tuning:")

        optimizer = torch.optim.Adam(model.parameters(), lr=fine_tuning_lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, fine_tune_scheduler_step, scheduler_gamma)
        
        for ep in range(fine_tune_ep):
            random.shuffle(multi_step_list)
            iters = [iter(loader) for loader in multi_step_train_loaders]

            model.train()
            t1 = default_timer()
            
            train_mse_list = [0.0 for i in range(len(multi_step_train_loaders))]
            train_l2_list = [0.0 for i in range(len(multi_step_train_loaders))]

            for n in tqdm(multi_step_list):
                num_steps = fine_tuning_num_steps[n]

                x, y = next(iters[n])
                x, y = x.cuda(), y.cuda()
                x = x.reshape(-1, T, dim)

                optimizer.zero_grad()
                
                # Permute for FNO
                x_perm = x.permute(0, 2, 1)
                out_perm = model(x_perm)
                out = out_perm.permute(0, 2, 1)

                for i in range(num_steps - 1):
                    out_perm = out.permute(0, 2, 1)
                    out_next_perm = model(out_perm)
                    out = out_next_perm.permute(0, 2, 1)

                mse = F.mse_loss(torch.reshape(out, (-1, T * dim)), torch.reshape(y, (-1, T * dim)), reduction='mean')
                l2 = myloss(torch.reshape(out, (-1, T * dim)), torch.reshape(y, (-1, T * dim)))
                l2.backward() # use the l2 relative loss

                optimizer.step()
                scheduler.step()
                train_mse_list[n] += mse.item()
                train_l2_list[n] += l2.item()

            model.eval()
            test_l2_list = [0.0 for i in range(len(multi_step_test_loaders))]
            with torch.no_grad():
                for n in range(len(multi_step_test_loaders)):
                    num_steps = fine_tuning_num_steps[n]
                    for x, y in multi_step_test_loaders[n]:
                        x, y = x.cuda(), y.cuda()
                        x = x.reshape(-1, T, dim)

                        # Permute for FNO
                        x_perm = x.permute(0, 2, 1)
                        out_perm = model(x_perm)
                        out = out_perm.permute(0, 2, 1)

                        for i in range(num_steps - 1):
                            out_perm = out.permute(0, 2, 1)
                            out_next_perm = model(out_perm)
                            out = out_next_perm.permute(0, 2, 1)

                        test_l2_list[n] += myloss(torch.reshape(out, (-1, T * dim)), torch.reshape(y, (-1, T * dim))).item()

            for n in range(len(multi_step_train_loaders)):
                train_mse_list[n] /= len(multi_step_train_loaders[n])
                train_l2_list[n] /= ntrain_list[n]
                test_l2_list[n] /= ntest_list[n]

            t2 = default_timer()
            print("Fine-tune epoch:", ep + 1, "Time:", t2-t1, "Train MSE:", train_mse_list, "Train L2:", train_l2_list, "Test L2:", test_l2_list)

    if model_save_path:
        torch.save(model, model_save_path)
        print("Weights saved to", model_save_path)