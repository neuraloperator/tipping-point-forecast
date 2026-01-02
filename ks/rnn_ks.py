"""
    This script applies a GRU to predict nonstationary KS in an
    autoregressive manner, where each step outputs an interval of the solution.
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import sys
import random

from utilities_ks import *
from torch.nn import GRU

from timeit import default_timer
import scipy.io
from tqdm import tqdm

torch.manual_seed(0)
np.random.seed(0)

class GRU_net(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_units, num_layers):
        super(GRU_net, self).__init__()

        self.hidden_units = hidden_units
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        self.p1 = nn.Linear(in_dim, self.hidden_units // 2)
        self.p2 = nn.Linear(self.hidden_units // 2, self.hidden_units)
        self.p3 = nn.Linear(self.hidden_units, 3 * self.hidden_units // 2)
        self.gru = GRU(input_size = 3 * self.hidden_units // 2, hidden_size = hidden_units, num_layers=num_layers, batch_first=True)
        self.q1 = nn.Linear(self.hidden_units, self.hidden_units * 2)
        self.q2 = nn.Linear(self.hidden_units * 2, out_dim)

        self.bias_h = nn.Parameter(torch.normal(torch.tensor(0.),torch.tensor(1.)))

    def forward(self, x, init_hidden_states=None): # init_hidden_states has shape (num_layers, batch_size, hidden_units)
        batch_size, seq_len = x.shape[0], x.shape[1]

        if init_hidden_states is None:
            init_hidden_states = torch.zeros(self.num_layers, batch_size, self.hidden_units).to(x.device)
            init_hidden_states += self.bias_h

        x = F.selu(self.p1(x))
        x = F.selu(self.p2(x))
        x = F.selu(self.p3(x))
        

        out, h = self.gru(x, init_hidden_states)
        pred = out[:,-1]

        pred = F.selu(self.q1(pred))
        pred = self.q2(pred)

        return pred, h

    def predict(self, x, num_steps):
        output = []
        states = None
        
        for i in range(num_steps):
            pred, states = self.forward(x, states)
            output.append(pred)
            x = pred.reshape((pred.shape[0], 1, pred.shape[-1]))

        return torch.stack(output, dim=1)

    def count_params(self):
        # Credit: Vadim Smolyakov on PyTorch forum
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return int(sum([np.prod(p.size()) for p in model_parameters]))

def round_down(num, divisor): # rounds `num` down to nearest multiple of `divisor`
    return num - (num % divisor)


if __name__ == '__main__':
    ntrain = 160
    ntest = 40

    train_up_to_tipping_point = True
    tipping_data_split_prop = 0.60 # proportion of trajectory at which to split the data for pre/post tipping

    hidden_units = 512
    num_layers = 3

    dim = 1

    batch_size = 32
    epochs = 40
    learning_rate = 1e-3
    scheduler_step = 500
    scheduler_gamma = 0.5

    multi_step_fine_tuning = True
    fine_tuning_num_steps = 3
    fine_tune_ep = 25
    fine_tuning_lr = 1e-5 # NOTE: change if necessary
    fine_tune_scheduler_step = 500
    fine_tune_scheduler_gamma = 0.5

    print("Epochs:", epochs)
    print("Learning rate:", learning_rate)
    print("Scheduler step:", scheduler_step)
    print("Scheduler gamma:", scheduler_gamma)
    print()

    path = 'MODEL/SAVE/PATH'
    path_model = 'model/'+path

    sub = 4 # spatial subsample
    T = 16 # input last T time steps and output next T
    n_intervals = 10

    ################################################################
    # data and preprocessing
    ################################################################

    t1 = default_timer()
    data = np.load('PATH/TO/DATA.npy')[:, ::sub]
    data = torch.tensor(data)

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

    train = torch.stack(torch.split(train, T, dim=-1)).permute(1,0,3,2).unsqueeze(-1)
    test = torch.stack(torch.split(test, T, dim=-1)).permute(1,0,3,2).unsqueeze(-1)

    # Generate training pairs
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for traj in train:
        for i in range(train.shape[1] - n_intervals - 1): # - 1 to make sure the y value exists
            x_train.append(traj[i:i + n_intervals])
            y_train.append(traj[i + n_intervals])

    for traj in test:
        for i in range(test.shape[1] - n_intervals - 1):
            x_test.append(traj[i:i + n_intervals])
            y_test.append(traj[i + n_intervals])
    
    x_train = torch.stack(x_train, dim=0).float()
    y_train = torch.stack(y_train, dim=0).float()
    x_test = torch.stack(x_test, dim=0).float()
    y_test = torch.stack(y_test, dim=0).float()

    if multi_step_fine_tuning:
        multi_step_x_train_list = []
        multi_step_y_train_list = []
        multi_step_x_test_list = []
        multi_step_y_test_list = []

        for n in range(2, fine_tuning_num_steps + 1):
            n_xlist = []
            n_ylist = []
            for traj in train:
                for i in range(train.shape[1] - n_intervals - n):
                    n_xlist.append(traj[i:i + n_intervals])
                    n_ylist.append(traj[i + n_intervals + n - 1])
            multi_step_x_train_list.append(torch.stack(n_xlist, dim=0).float())
            multi_step_y_train_list.append(torch.stack(n_ylist, dim=0).float())

        for n in range(2, fine_tuning_num_steps + 1):
            n_xlist = []
            n_ylist = []
            for traj in test:
                for i in range(test.shape[1] - n_intervals - n):
                    n_xlist.append(traj[i:i + n_intervals])
                    n_ylist.append(traj[i + n_intervals + n - 1])
            multi_step_x_test_list.append(torch.stack(n_xlist, dim=0).float())
            multi_step_y_test_list.append(torch.stack(n_ylist, dim=0).float())

        multi_step_x_train_list.insert(0, x_train)
        multi_step_y_train_list.insert(0, y_train)
        multi_step_x_test_list.insert(0, x_test)
        multi_step_y_test_list.insert(0, y_test)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False, drop_last=True)

    if multi_step_fine_tuning:
        multi_step_train_loaders = [torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=batch_size, shuffle=True, drop_last=True) for x,y in zip(multi_step_x_train_list, multi_step_y_train_list)]
        multi_step_test_loaders = [torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=batch_size, shuffle=False, drop_last=True) for x,y in zip(multi_step_x_test_list, multi_step_y_test_list)]

        multi_step_list = []
        for i in range(len(multi_step_train_loaders)):
            multi_step_list += [i for j in range(len(multi_step_train_loaders[i]))]
        random.shuffle(multi_step_list)

    print("Preprocessing complete")
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)
    device = torch.device('cuda')
    print()


    ################################################################
    # model and optimizer
    ################################################################

    model = GRU_net(in_dim = T*dim*n_x, out_dim = T*dim*n_x, hidden_units = hidden_units, num_layers = num_layers).cuda()
    print("Model parameters:", model.count_params())

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
        for x, y in tqdm(train_loader):
            x = x.to(device).view(batch_size, n_intervals, T, n_x, dim)
            y = y.to(device).view(batch_size, T, n_x, dim)
            x = x.reshape(batch_size, n_intervals, T * n_x * dim)

            out = model.predict(x, num_steps=1)[:,-1]
            out = out.reshape((batch_size, T, n_x, dim))
            loss = lploss(out, y)
            train_l2 += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()

        test_l2 = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device).view(batch_size, n_intervals, T, n_x, dim)
                y = y.to(device).view(batch_size, T, n_x, dim)
                x = x.reshape(batch_size, n_intervals, T * n_x * dim)

                out = model.predict(x, num_steps=1)[:,-1]
                out = out.reshape((batch_size, T, n_x, dim))
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
            
            train_l2_list = [0.0 for i in range(len(multi_step_train_loaders))]

            for n in tqdm(multi_step_list):
                num_steps = n + 1

                x, y = next(iters[n])
                x, y = x.cuda(), y.cuda()
                x = x.reshape(batch_size, n_intervals, T * n_x * dim)

                optimizer.zero_grad()
                out = model.predict(x, num_steps)[:,-1]
                out = out.reshape((batch_size, T, n_x, dim))

                l2 = lploss(torch.reshape(out, (-1, n_x * T * dim)), torch.reshape(y, (-1, n_x * T * dim)))
                optimizer.zero_grad()
                l2.backward() # use the l2 relative loss

                optimizer.step()
                scheduler.step()
                train_l2_list[n] += l2.item()

            model.eval()
            test_l2_list = [0.0 for i in range(len(multi_step_test_loaders))]
            with torch.no_grad():
                for n in range(len(multi_step_test_loaders)):
                    num_steps = n + 1
                    for x, y in multi_step_test_loaders[n]:
                        x = x.to(device).view(batch_size, n_intervals, T, n_x, dim)
                        y = y.to(device).view(batch_size, T, n_x, dim)
                        x = x.reshape(batch_size, n_intervals, T * n_x * dim)

                        out = model.predict(x, num_steps)[:,-1]
                        out = out.reshape((batch_size, T, n_x, dim))

                        test_l2_list[n] += lploss(torch.reshape(out, (-1, n_x * T * dim)), torch.reshape(y, (-1, n_x * T * dim))).item()

            for n in range(len(multi_step_train_loaders)):
                train_l2_list[n] /= ntrain_list[n]
                test_l2_list[n] /= ntest_list[n]

            t2 = default_timer()
            print("Fine-tune epoch:", ep + 1, "Time:", t2-t1, "Train L2:", train_l2_list, "Test L2:", test_l2_list)


    torch.save(model, path_model)
    print("Weights saved to", path_model)