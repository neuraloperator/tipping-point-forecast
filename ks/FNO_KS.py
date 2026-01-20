"""
    This script implements a 2D FNO, which maps the solution from some time interval
    in the past to a time interval in the future.
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
from tqdm import tqdm

from utilities_ks import *
from neuralop.models import FNO

from timeit import default_timer
import scipy.io

torch.manual_seed(0)
np.random.seed(0)

def round_down(num, divisor): # rounds `num` down to nearest multiple of `divisor`
    return num - (num % divisor)


if __name__ == '__main__':
    ################################################################
    # parameters
    ################################################################

    ntrain = 160
    ntest = 40

    train_up_to_tipping_point = True
    tipping_data_split_prop = 0.60 # proportion of trajectory at which to split the data for pre/post tipping

    modes1 = 20
    modes2 = 20
    width = 32

    dim = 1

    batch_size = 32
    epochs = 100
    learning_rate = 0.005
    scheduler_step = 200
    scheduler_gamma = 0.5

    multi_step_fine_tuning = True
    fine_tuning_num_steps = 5
    fine_tune_ep = 50
    fine_tuning_lr = 1e-5
    fine_tune_scheduler_step = 250
    fine_tune_scheduler_gamma = 0.5

    print("Epochs:", epochs)
    print("Learning rate:", learning_rate)
    print("Scheduler step:", scheduler_step)
    print("Scheduler gamma:", scheduler_gamma)
    print()

    path_model = 'MODEL/SAVE/PATH'

    sub = 4 # spatial subsample
    T = 16 # input last T time steps and output next T

    ################################################################
    # data and preprocessing
    ################################################################

    t1 = default_timer()
    data = np.load('PATH/TO/DATA.npy')[:, ::sub]
    data = torch.tensor(data, dtype=torch.float)

    n_time = data.shape[2]
    n_x = data.shape[1]

    train = data[:ntrain]
    test = data[-ntest:]

    if train_up_to_tipping_point:
        tipping_idx = round_down(int(tipping_data_split_prop * n_time), T)
        train = train[:,:,:tipping_idx]
        test = test[:,:,:tipping_idx]
    else:
        idx = round_down(n_time, T)
        train = train[:,:,:idx]
        test = test[:,:,:idx]

    train = torch.stack(torch.split(train, T, dim=-1)).permute(1,0,3,2)
    test = torch.stack(torch.split(test, T, dim=-1)).permute(1,0,3,2)

    x_train = train[:,:-1].float()
    y_train = train[:,1:].float()
    x_test = test[:,:-1].float()
    y_test = test[:,1:].float()
    print(x_train.shape)

    if multi_step_fine_tuning:
        multi_step_x_train_list = [train[:,:-n][:,::n] for n in range(2, fine_tuning_num_steps + 1)]
        multi_step_y_train_list = [train[:,n:][:,::n] for n in range(2, fine_tuning_num_steps + 1)]
        multi_step_x_test_list = [test[:,:-n][:,::n] for n in range(2, fine_tuning_num_steps + 1)]
        multi_step_y_test_list = [test[:,n:][:,::n] for n in range(2, fine_tuning_num_steps + 1)]

    x_train = torch.reshape(x_train, (-1, T, n_x, dim)).float()
    y_train = torch.reshape(y_train, (-1, T, n_x, dim)).float()
    x_test = torch.reshape(x_test, (-1, T, n_x, dim)).float()
    y_test = torch.reshape(y_test, (-1, T, n_x, dim)).float()

    if multi_step_fine_tuning:
        multi_step_x_train_list = [torch.tensor(np.reshape(arr, (-1, T, n_x, dim))).float() for arr in multi_step_x_train_list]
        multi_step_y_train_list = [torch.tensor(np.reshape(arr, (-1, T, n_x, dim))).float() for arr in multi_step_y_train_list]
        multi_step_x_test_list = [torch.tensor(np.reshape(arr, (-1, T, n_x, dim))).float() for arr in multi_step_x_test_list]
        multi_step_y_test_list = [torch.tensor(np.reshape(arr, (-1, T, n_x, dim))).float() for arr in multi_step_y_test_list]
        multi_step_x_train_list.insert(0, x_train)
        multi_step_y_train_list.insert(0, y_train)
        multi_step_x_test_list.insert(0, x_test)
        multi_step_y_test_list.insert(0, y_test)

    assert x_train.shape == y_train.shape
    assert x_test.shape == y_test.shape

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False, drop_last=True)

    if multi_step_fine_tuning:
        multi_step_train_loaders = [torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=batch_size, shuffle=True, drop_last=True) for x,y in zip(multi_step_x_train_list, multi_step_y_train_list)]
        multi_step_test_loaders = [torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=batch_size, shuffle=False, drop_last=True) for x,y in zip(multi_step_x_test_list, multi_step_y_test_list)]

        multi_step_list = []
        for i in range(len(multi_step_train_loaders)):
            multi_step_list += [i for j in range(len(multi_step_train_loaders[i]))]
        random.shuffle(multi_step_list)

    print('preprocessing finished, time used:', default_timer() - t1)
    print("Train shape:", x_train.shape)
    print("Test shape:", x_test.shape)
    print()
    device = torch.device('cuda')

    ################################################################
    # model and optimizer
    ################################################################

    model = FNO(
        n_modes=(modes1, modes2),
        hidden_channels=width,
        in_channels=dim,
        out_channels=dim,
        n_layers=4,
        domain_padding=[0.5, 0]
    ).cuda().float()
    print("Model parameters:", count_params(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    lploss = LpLoss(size_average=False)

    ################################################################
    # training and evaluation
    ################################################################

    num_train_samples = x_train.shape[0] 
    num_test_samples = x_test.shape[0]

    if multi_step_fine_tuning:
        ntrain_list = [arr.shape[0] for arr in multi_step_x_train_list]
        ntest_list = [arr.shape[0] for arr in multi_step_x_test_list]

    ### Training 
    print("Begin training:")
    for ep in range(1, epochs + 1):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        for x, y in train_loader:
            x = x.to(device).view(batch_size, T, n_x, dim)
            y = y.to(device).view(batch_size, T, n_x, dim)

            # Permute for FNO
            x_perm = x.permute(0, 3, 1, 2)
            out_perm = model(x_perm)
            # Permute back
            out = out_perm.permute(0, 2, 3, 1).reshape(batch_size, T, n_x, dim)
            loss = lploss(out, y)
            train_l2 += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()

        test_l2 = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device).view(batch_size, T, n_x, dim)
                y = y.to(device).view(batch_size, T, n_x, dim)

                # Permute for FNO
                x_perm = x.permute(0, 3, 1, 2)
                out_perm = model(x_perm)
                out = out_perm.permute(0, 2, 3, 1).reshape(batch_size, T, n_x, dim)
                test_l2 += lploss(out, y).item()

        train_l2 /= num_train_samples
        test_l2 /= num_test_samples

        print("Epoch:", ep, "Time:", default_timer() - t1, "Train L2:", train_l2, "Test L2:", test_l2)

    ### Multi-step fine-tuning
    if multi_step_fine_tuning:
        print("Begin multi-step fine-tuning:")

        optimizer = torch.optim.Adam(model.parameters(), lr=fine_tuning_lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=fine_tune_scheduler_step, gamma=fine_tune_scheduler_gamma)
        
        for ep in range(fine_tune_ep):
            random.shuffle(multi_step_list)
            iters = [iter(loader) for loader in multi_step_train_loaders]

            model.train()
            t1 = default_timer()
            
            train_mse_list = [0.0 for i in range(len(multi_step_train_loaders))]
            train_l2_list = [0.0 for i in range(len(multi_step_train_loaders))]

            for n in tqdm(multi_step_list):
                num_steps = n + 1

                x, y = next(iters[n])
                x = x.to(device).view(batch_size, T, n_x, dim)
                y = y.to(device).view(batch_size, T, n_x, dim)

                optimizer.zero_grad()
                # Permute for FNO
                x_perm = x.permute(0, 3, 1, 2)
                out_perm = model(x_perm)
                out = out_perm.permute(0, 2, 3, 1).reshape(batch_size, T, n_x, dim)

                for i in range(num_steps - 1):
                    out_perm = out.permute(0, 3, 1, 2)
                    out_next_perm = model(out_perm)
                    out = out_next_perm.permute(0, 2, 3, 1).reshape(batch_size, T, n_x, dim)

                mse = F.mse_loss(torch.reshape(out, (-1, n_x * T * dim)), torch.reshape(y, (-1, n_x * T * dim)), reduction='mean')
                l2 = lploss(torch.reshape(out, (-1, n_x * T * dim)), torch.reshape(y, (-1, n_x * T * dim)))
                optimizer.zero_grad()
                l2.backward() # use the l2 relative loss

                optimizer.step()
                scheduler.step()
                train_mse_list[n] += mse.item()
                train_l2_list[n] += l2.item()

            model.eval()
            test_l2_list = [0.0 for i in range(len(multi_step_test_loaders))]
            with torch.no_grad():
                for n in range(len(multi_step_test_loaders)):
                    num_steps = n + 1
                    for x, y in multi_step_test_loaders[n]:
                        x = x.to(device).view(batch_size, T, n_x, dim)
                        y = y.to(device).view(batch_size, T, n_x, dim)

                        # Permute for FNO
                        x_perm = x.permute(0, 3, 1, 2)
                        out_perm = model(x_perm)
                        out = out_perm.permute(0, 2, 3, 1).reshape(batch_size, T, n_x, dim)

                        for i in range(num_steps - 1):
                            out_perm = out.permute(0, 3, 1, 2)
                            out_next_perm = model(out_perm)
                            out = out_next_perm.permute(0, 2, 3, 1).reshape(batch_size, T, n_x, dim)

                        test_l2_list[n] += lploss(torch.reshape(out, (-1, n_x * T * dim)), torch.reshape(y, (-1, n_x * T * dim))).item()

            for n in range(len(multi_step_train_loaders)):
                train_mse_list[n] /= len(multi_step_train_loaders[n])
                train_l2_list[n] /= ntrain_list[n]
                test_l2_list[n] /= ntest_list[n]

            t2 = default_timer()
            print("Fine-tune epoch:", ep + 1, "Time:", t2-t1, "Train MSE:", train_mse_list, "Train L2:", train_l2_list, "Test L2:", test_l2_list)


    torch.save(model, path_model)
    print("Weights saved to", path_model)