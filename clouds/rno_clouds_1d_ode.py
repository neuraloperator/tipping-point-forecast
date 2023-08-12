import torch.nn.functional as F
from timeit import default_timer
from tqdm import tqdm
import pickle
import sys
import random
from sklearn.preprocessing import MinMaxScaler

sys.path.append("../nonstationary_lorenz")
from utilities3 import *

sys.path.append('../')
from RNO_1d import *

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

class RNO_1D_5layers(nn.Module):
    def __init__(self, in_dim, out_dim, modes, width, padding=None):
        super(RNO_1D_5layers, self).__init__()

        self.modes = modes
        self.width = width
        self.padding = padding
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.p = nn.Linear(in_dim + 1, self.width) # input channel_dim is in_dim + 1: u is in-dim, and time label t is 1 dim

        self.layer1 = RNO_layer(in_dim, out_dim, modes, width, return_sequences=True)
        self.layer2 = RNO_layer(in_dim, out_dim, modes, width, return_sequences=True)
        self.layer3 = RNO_layer(in_dim, out_dim, modes, width, return_sequences=True)
        self.layer4 = RNO_layer(in_dim, out_dim, modes, width, return_sequences=True)
        self.layer5 = RNO_layer(in_dim, out_dim, modes, width, return_sequences=False)

        self.q = nn.Linear(self.width, out_dim)
    
    def forward(self, x, init_hidden_states=[None, None, None, None, None]): # h must be padded if using padding
        batch_size, timesteps, domain_size, dim = x.shape

        h1, h2, h3, h4, h5 = init_hidden_states

        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)

        x = x.permute(0, 1, 3, 2)
        if self.padding:
            x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        hidden_seq1 = self.layer1(x, h1)
        hidden_seq2 = self.layer2(hidden_seq1, h2)
        hidden_seq3 = self.layer3(hidden_seq2, h3)
        hidden_seq4 = self.layer4(hidden_seq3, h4)
        h = self.layer5(hidden_seq4, h5) # output shape is (batch, width, domain_size)

        # Save final hidden states
        final_hidden_states = [hidden_seq1[:,-1], hidden_seq2[:,-1], hidden_seq3[:,-1], hidden_seq4[:,-1], h]

        h_unpad = h[..., :-self.padding] # pad the domain if input is non-periodic

        h_unpad = h_unpad.permute(0, 2, 1)
        
        pred = self.q(h_unpad)

        return pred, final_hidden_states

    def predict(self, x, num_steps, forcing=None): # num_steps is the number of steps ahead to predict
        # forcing is an array of length `num_steps - 1` that is appended to the end of the 
            # input dimensions of `pred` at subsequent steps ahead
            # `forcing` should have shape (batch, num_steps - 1, domain_size, forcing_dim)
        output = []
        states = [None, None, None, None, None]
        
        for i in range(num_steps):
            pred, states = self.forward(x, states)
            output.append(pred)
            x = pred.reshape((pred.shape[0], 1, pred.shape[1], pred.shape[2]))
            if forcing is not None and i < num_steps - 1:
                forcing_term = torch.unsqueeze(forcing[:, i], 1)
                x = torch.cat((x, forcing_term), dim=-1)

        return torch.stack(output, dim=1)
        
    def get_grid(self, shape, device):
        batchsize, steps, size_x = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, steps, 1, 1])
        return gridx.to(device)

    def count_params(self):
        # Credit: Vadim Smolyakov on PyTorch forum
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return int(sum([np.prod(p.size()) for p in model_parameters]))

class RNO_1D_7layers(nn.Module):
    def __init__(self, in_dim, out_dim, modes, width, padding=None):
        super(RNO_1D_7layers, self).__init__()

        self.modes = modes
        self.width = width
        self.padding = padding
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.p = nn.Linear(in_dim + 1, self.width) # input channel_dim is in_dim + 1: u is in-dim, and time label t is 1 dim

        self.layer1 = RNO_layer(in_dim, out_dim, modes, width, return_sequences=True)
        self.layer2 = RNO_layer(in_dim, out_dim, modes, width, return_sequences=True)
        self.layer3 = RNO_layer(in_dim, out_dim, modes, width, return_sequences=True)
        self.layer4 = RNO_layer(in_dim, out_dim, modes, width, return_sequences=True)
        self.layer5 = RNO_layer(in_dim, out_dim, modes, width, return_sequences=True)
        self.layer6 = RNO_layer(in_dim, out_dim, modes, width, return_sequences=True)
        self.layer7 = RNO_layer(in_dim, out_dim, modes, width, return_sequences=False)

        self.q = nn.Linear(self.width, out_dim)
    
    def forward(self, x, init_hidden_states=[None, None, None, None, None, None, None]): # h must be padded if using padding
        batch_size, timesteps, domain_size, dim = x.shape

        h1, h2, h3, h4, h5, h6, h7 = init_hidden_states

        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)

        x = x.permute(0, 1, 3, 2)
        if self.padding:
            x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        hidden_seq1 = self.layer1(x, h1)
        hidden_seq2 = self.layer2(hidden_seq1, h2)
        hidden_seq3 = self.layer3(hidden_seq2, h3)
        hidden_seq4 = self.layer4(hidden_seq3, h4)
        hidden_seq5 = self.layer3(hidden_seq4, h5)
        hidden_seq6 = self.layer4(hidden_seq5, h6)
        h = self.layer7(hidden_seq6, h7) # output shape is (batch, width, domain_size)

        # Save final hidden states
        final_hidden_states = [hidden_seq1[:,-1], hidden_seq2[:,-1], hidden_seq3[:,-1], hidden_seq4[:,-1], hidden_seq5[:,-1], hidden_seq6[:,-1], h]

        h_unpad = h[..., :-self.padding] # pad the domain if input is non-periodic

        h_unpad = h_unpad.permute(0, 2, 1)
        
        pred = self.q(h_unpad)

        return pred, final_hidden_states

    def predict(self, x, num_steps, forcing=None): # num_steps is the number of steps ahead to predict
        # forcing is an array of length `num_steps - 1` that is appended to the end of the 
            # input dimensions of `pred` at subsequent steps ahead
            # `forcing` should have shape (batch, num_steps - 1, domain_size, forcing_dim)
        output = []
        states = [None, None, None, None, None, None, None]
        
        for i in range(num_steps):
            pred, states = self.forward(x, states)
            output.append(pred)
            x = pred.reshape((pred.shape[0], 1, pred.shape[1], pred.shape[2]))
            if forcing is not None and i < num_steps - 1:
                forcing_term = torch.unsqueeze(forcing[:, i], 1)
                x = torch.cat((x, forcing_term), dim=-1)

        return torch.stack(output, dim=1)
        
    def get_grid(self, shape, device):
        batchsize, steps, size_x = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, steps, 1, 1])
        return gridx.to(device)

    def count_params(self):
        # Credit: Vadim Smolyakov on PyTorch forum
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return int(sum([np.prod(p.size()) for p in model_parameters]))

if __name__ == '__main__':
    ################################################################
    #  configurations
    ################################################################
    sub = 1 # temporal subsampling rate
    T = 128 # input last T time steps and output next T
    n_intervals = 5 # number of intervals in each training example
    
    in_dim = 5 # dimensionality of ODE + dimension of input parameter
    out_dim = 5

    batch_size = 64
    learning_rate = 1e-3
    epochs = 0

    scheduler_step = 100
    scheduler_gamma = 0.5

    modes = T // 2
    width = 128

    train_up_to_tipping_point = True # if true, only trains model on data up to tipping point
    
    multi_step_fine_tuning = True
    
    fine_tuning_num_steps = [1,2,3,4,5,6,7,8,9,10]

    fine_tune_ep = 25
    fine_tune_scheduler_step = 300
    fine_tuning_lr = 1e-3

    experiment_name = 'MODEL/PATH'

    ################################################################
    # read data
    ################################################################

    # Data is of the shape (number of trajectories, number of samples, grid size)
    data = np.load("clouds_noisy_brownian_100_traj_small_dt.npy")
    
    tipping_point = 10.0 # time of tipping point -- data-dependent!

    dt = 1.0 / 365 # years
    dt_sec = dt * 3600 * 24 * 365 # seconds
    T1 = 0.0 # years
    T2 = 20.0 # years
    data = data[:,:-1] # Make shape easier to divide

    data = data[..., :-1] # remove forcing

    num_traj = data.shape[0]
    traj_len = data.shape[1]

    train_trajectories = 50
    test_trajectories = 50

    # Min-max scaling
    scaler = MinMaxScaler()
    scaler.fit(np.reshape(data[:train_trajectories], (train_trajectories * traj_len, in_dim))) # fit on training set
    data = scaler.transform(np.reshape(data, (num_traj * traj_len, in_dim)))
    data = np.reshape(data, (num_traj, traj_len, in_dim))

    # Train-test split
    train = data[:train_trajectories]
    test = data[-test_trajectories:]

    if train_up_to_tipping_point:
        train_idx = round_down(get_index_from_time(T1, T2, tipping_point, train.shape[1]), T)
        test_idx = round_down(get_index_from_time(T1, T2, tipping_point, test.shape[1]), T)
        train = train[:,:train_idx]
        test = test[:,:test_idx]

    train = np.reshape(train, (train_trajectories, -1, T, in_dim))
    test = np.reshape(test, (test_trajectories, -1, T, in_dim))

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
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)[...,:out_dim]
    x_test = np.array(x_test)
    y_test = np.array(y_test)[...,:out_dim]

    x_train = torch.tensor(x_train).float()
    y_train = torch.tensor(y_train).float()
    x_test = torch.tensor(x_test).float()
    y_test = torch.tensor(y_test).float()


    if multi_step_fine_tuning:
        multi_step_x_train_list = []
        multi_step_y_train_list = []
        multi_step_x_test_list = []
        multi_step_y_test_list = []

        for n in fine_tuning_num_steps[1:]:
            n_xlist = []
            n_ylist = []
            for traj in train:
                for i in range(train.shape[1] - n_intervals - n):
                    n_xlist.append(traj[i:i + n_intervals])
                    n_ylist.append(traj[i + n_intervals + n - 1])
            multi_step_x_train_list.append(np.array(n_xlist))
            multi_step_y_train_list.append(np.array(n_ylist))


        for n in fine_tuning_num_steps[1:]:
            n_xlist = []
            n_ylist = []
            for traj in test:
                for i in range(test.shape[1] - n_intervals - n):
                    n_xlist.append(traj[i:i + n_intervals])
                    n_ylist.append(traj[i + n_intervals + n - 1])
            multi_step_x_test_list.append(np.array(n_xlist))
            multi_step_y_test_list.append(np.array(n_ylist))

        multi_step_x_train_list = [torch.tensor(arr).float() for arr in multi_step_x_train_list]
        multi_step_y_train_list = [torch.tensor(arr).float()[...,:out_dim] for arr in multi_step_y_train_list]
        multi_step_x_test_list = [torch.tensor(arr).float() for arr in multi_step_x_test_list]
        multi_step_y_test_list = [torch.tensor(arr).float()[...,:out_dim] for arr in multi_step_y_test_list]
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
    model = RNO_1D_5layers(in_dim, out_dim, modes, width, padding=T // 8).cuda().float()
    print("Parameters:", model.count_params())
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
            out = model.predict(x, num_steps=1)[:,-1]

            mse = F.mse_loss(torch.reshape(out, (-1, T * out_dim)), torch.reshape(y, (-1, T * out_dim)), reduction='mean')
            l2 = myloss(torch.reshape(out, (-1, T * out_dim)), torch.reshape(y, (-1, T * out_dim)))
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

                out = model.predict(x, num_steps=1)[:,-1]
                test_l2 += myloss(torch.reshape(out, (-1, T * out_dim)), torch.reshape(y, (-1, T * out_dim))).item()

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

                optimizer.zero_grad()

                out = model.predict(x, num_steps)[:,-1]

                mse = F.mse_loss(torch.reshape(out, (-1, T * out_dim)), torch.reshape(y, (-1, T * out_dim)), reduction='mean')
                l2 = myloss(torch.reshape(out, (-1, T * out_dim)), torch.reshape(y, (-1, T * out_dim)))
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
                        x, y = x.cuda(), y.cuda()

                        out = model.predict(x, num_steps)[:,-1]

                        test_l2_list[n] += myloss(torch.reshape(out, (-1, T * out_dim)), torch.reshape(y, (-1, T * out_dim))).item()

            for n in range(len(multi_step_train_loaders)):
                train_mse_list[n] /= len(multi_step_train_loaders[n])
                train_l2_list[n] /= ntrain_list[n]
                test_l2_list[n] /= ntest_list[n]

            t2 = default_timer()
            print("Fine-tune epoch:", ep + 1, "Time:", t2-t1, "Train MSE:", train_mse_list, "Train L2:", train_l2_list, "Test L2:", test_l2_list)

    torch.save(model, 'model/' + experiment_name)