import numpy as np
import random
import os
from functools import lru_cache
import torch
import math

# ---------------- Constants ----------------

wake_transition_times_1k = {
    'traj1' : 85.7, 'traj2' : 90.5, 'traj3' : 91.1, 'traj4' : 90.9, 'traj5' : 91.7,
    'traj6' : 92.4, 'traj7' : 92.1, 'traj8' : 92.3, 'traj9' : 93, 'traj10' : 93.2
}

wake_transition_times_5k = {
    'traj1_5k' : 131.5, 'traj2_5k' : 132.3, 'traj3_5k' : 132.4, 'traj4_5k' : 133,
    'traj5_5k' : 133.6, 'traj6_5k' : 133.3, 'traj7_5k' : 133, 'traj8_5k' : 133.8,
    'traj9_5k' : 133.4, 'traj10_5k' : 135.8
}

stall_times_5k = {
    'traj1_5k' : 146.9, 'traj2_5k' : 147.6, 'traj3_5k' : 147.7, 'traj4_5k' : 148.4,
    'traj5_5k' : 148.9, 'traj6_5k' : 148.6, 'traj7_5k' : 148.3, 'traj8_5k' : 149.2,
    'traj9_5k' : 148.8, 'traj10_5k' : 151.2
}

# ---------------- Helpers ----------------

class LpLoss(object):
    def __init__(self, d=1, p=2, L=2*math.pi, reduce_dims=0, reductions='sum'):
        super().__init__()

        self.d = d
        self.p = p

        if isinstance(reduce_dims, int):
            self.reduce_dims = [reduce_dims]
        else:
            self.reduce_dims = reduce_dims
        
        if self.reduce_dims is not None:
            if isinstance(reductions, str):
                assert reductions == 'sum' or reductions == 'mean'
                self.reductions = [reductions]*len(self.reduce_dims)
            else:
                for j in range(len(reductions)):
                    assert reductions[j] == 'sum' or reductions[j] == 'mean'
                self.reductions = reductions

        if isinstance(L, float):
            self.L = [L]*self.d
        else:
            self.L = L
    
    def uniform_h(self, x):
        h = [0.0]*self.d
        for j in range(self.d, 0, -1):
            h[-j] = self.L[-j]/x.size(-j)
        
        return h

    def reduce_all(self, x):
        for j in range(len(self.reduce_dims)):
            if self.reductions[j] == 'sum':
                x = torch.sum(x, dim=self.reduce_dims[j], keepdim=True)
            else:
                x = torch.mean(x, dim=self.reduce_dims[j], keepdim=True)
        
        return x

    def abs(self, x, y, h=None):
        #Assume uniform mesh
        if h is None:
            h = self.uniform_h(x)
        else:
            if isinstance(h, float):
                h = [h]*self.d
        
        const = math.prod(h)**(1.0/self.p)
        diff = const*torch.norm(torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d), \
                                              p=self.p, dim=-1, keepdim=False)

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()
            
        return diff

    def rel(self, x, y):

        diff = torch.norm(torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d), \
                          p=self.p, dim=-1, keepdim=False)
        ynorm = torch.norm(torch.flatten(y, start_dim=-self.d), p=self.p, dim=-1, keepdim=False)

        diff = diff/ynorm

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()
            
        return diff

    def __call__(self, x, y):
        return self.rel(x, y)

class UnitGaussianNormalizer:
    def __init__(self, x, eps=0.00001, reduce_dim=[0], verbose=True):
        super().__init__()
        n_samples, *shape = x.shape
        self.sample_shape = shape
        self.verbose = verbose
        self.reduce_dim = reduce_dim

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, reduce_dim, keepdim=True).squeeze(0)
        self.std = torch.std(x, reduce_dim, keepdim=True).squeeze(0)
        self.eps = eps
        
        if verbose:
            print(f'UnitGaussianNormalizer init on {n_samples}, reducing over {reduce_dim}, samples of shape {shape}.')
            print(f'   Mean and std of shape {self.mean.shape}, eps={eps}')

    def encode(self, x):
        # x = x.view(-1, *self.sample_shape)
        x -= self.mean
        x /= (self.std + self.eps)
        # x = (x.view(-1, *self.sample_shape) - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        # x = (x.view(self.sample_shape) * std) + mean
        # x = x.view(-1, *self.sample_shape)
        x *= std
        x += mean

        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()
        return self

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()
        return self
    
    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

@lru_cache(maxsize=256)
def get_grid_2d(shape, device):
    # `shape` is tuple (batch, ..., nx, ny)
    # output shape is (batch, ..., 1, nx, ny), where `1` is the channel dimension
    nx, ny = shape[-2], shape[-1]
    non_spatial_shape = tuple(shape[:-3]) # exclude channel dim
    temp_ones = tuple([1 for i in range(len(non_spatial_shape))])

    gridx = torch.tensor(np.linspace(0, 1, nx), dtype=torch.float)
    gridx = gridx.reshape(temp_ones + (nx, 1)).repeat(non_spatial_shape + (1, 1, ny))
    gridy = torch.tensor(np.linspace(0, 1, ny), dtype=torch.float)
    gridy = gridy.reshape(temp_ones + (1, ny)).repeat(non_spatial_shape + (1, nx, 1))
    return torch.cat((gridx, gridy), dim=-3).to(device)

def divergence_2d(u, domain_spacing):
    """Approximates divergence by finite difference"""
    assert len(u.shape) == 4
    batch_size, dim, nx, ny = u.shape
    assert dim == 2
    FX_x = torch.gradient(u[:,0], spacing=domain_spacing, dim=1)[0]
    FY_y = torch.gradient(u[:,1], spacing=domain_spacing, dim=2)[0]
    return FX_x + FY_y

# ---------------- Data Loaders ----------------

def load_airfoil_multi_step(num_steps, 
                            history, 
                            batch_size, 
                            ntrain,
                            nval,
                            ntest,
                            pre_tipping=True, 
                            normalize='channel-wise', 
                            normalize_using_pre_tipping_only=True,
                            normalizer_object=None,
                            re5000=False, 
                            stall=False, 
                            common_path='./airfoil_stall_2D/', 
                            domain_keep_x=[0,1],
                            domain_keep_y=[0,1],
                            return_one_step_no_loader=False,
                            use_divergence_corrected=True,
                            return_normalizer=False):
    """
    Legacy loader used by training scripts. Loads data into memory lists and then creates loaders.
    """
    if re5000 and stall:
        transition_times = stall_times_5k
    elif re5000 and not stall:
        transition_times = wake_transition_times_5k
    elif not re5000 and not stall:
        transition_times = wake_transition_times_1k
    else:
        raise RuntimeError("re1000 = True and stall = True are not compatible.")

    traj_postfix = '_5k' if re5000 else ''
    n_traj = ntrain + nval + ntest

    train_data = []
    val_data = []
    test_data = []
    normalization_data = [] 

    for i in range(1, n_traj + 1):
        traj_name = 'traj' + str(i) + traj_postfix
        traj_filename = 'divergence_corrected_traj.npy' if use_divergence_corrected else 'traj.npy'
        
        # Load trajectory
        traj_path = os.path.join(common_path, traj_name, traj_filename)
        if not os.path.exists(traj_path):
            raise FileNotFoundError(f"Data file not found: {traj_path}. Please ensure preprocessed data is available.")

        traj = torch.tensor(np.load(traj_path)).float()
        nt, _, nx, ny = traj.shape

        time = np.genfromtxt(os.path.join(common_path, traj_name, 'time.dat'))
        tipping_idx = np.argmin(np.abs(time - transition_times[traj_name]))
        cutoff_idx = tipping_idx if pre_tipping else len(traj)

        assert nt == len(time), "Length of time labels and input trajectory must be equal."

        # Truncation of the spatial domain
        x_truncate = [int(domain_keep_x[0] * nx), int(domain_keep_x[1] * nx)]
        y_truncate = [int(domain_keep_y[0] * ny), int(domain_keep_y[1] * ny)]

        traj_slice = traj[:cutoff_idx, :, x_truncate[0]:x_truncate[1], y_truncate[0]:y_truncate[1]]

        if i <= ntrain:
            train_data.append(traj_slice)
            if normalize_using_pre_tipping_only:
                normalization_data.append(traj[:tipping_idx, :, x_truncate[0]:x_truncate[1], y_truncate[0]:y_truncate[1]])
        elif i > ntrain and i <= ntrain + nval:
            val_data.append(traj_slice)
        else:
            test_data.append(traj_slice)

    # Prepare normalizer
    y_normalize = []
    source_for_norm = normalization_data if (normalize_using_pre_tipping_only and len(normalization_data) > 0) else train_data

    for traj in source_for_norm:
        traj_copy = torch.clone(traj)
        for i in range(traj_copy.shape[0] - history - 1):
            y_normalize.append(traj_copy[i + history])

    if len(y_normalize) > 0:
        y_normalize = torch.stack(y_normalize, dim=0).float()

    reduce_dims = None
    if normalize == 'channel-wise':
        reduce_dims = list(range(y_normalize.ndim)) 
    elif normalize == 'pixel-wise':
        reduce_dims = [0]
    
    if normalize and normalizer_object is None and len(y_normalize) > 0:
        normalizer = UnitGaussianNormalizer(y_normalize, reduce_dim=reduce_dims)
    elif normalize and normalizer_object:
        normalizer = normalizer_object
    else:
        normalizer = None

    # Helper to create pairs
    def create_pairs(data_list):
        x_list, y_list = [], []
        for traj in data_list:
            traj_copy = torch.clone(traj)
            if normalizer:
                traj_copy = normalizer.encode(traj_copy.reshape(-1, traj_copy.shape[-2], traj_copy.shape[-1])).reshape(*traj_copy.shape)
            x_traj, y_traj = [], []
            for i in range(traj_copy.shape[0] - history - 1):
                x_traj.append(traj_copy[i : i + history])
                y_traj.append(traj_copy[i + history])
            x_list.append(torch.stack(x_traj, dim=0))
            y_list.append(torch.stack(y_traj, dim=0))
        return x_list, y_list

    x_train, y_train = create_pairs(train_data)
    x_val, y_val = create_pairs(val_data)
    x_test, y_test = create_pairs(test_data)

    if return_one_step_no_loader:
        if return_normalizer:
            return (x_train, y_train), (x_val, y_val), (x_test, y_test), normalizer
        else:
            return (x_train, y_train), (x_val, y_val), (x_test, y_test)
    
    # Flatten for loading
    x_train = torch.cat(x_train, dim=0).float() if len(x_train) > 0 else []
    y_train = torch.cat(y_train, dim=0).float() if len(y_train) > 0 else []
    x_val = torch.cat(x_val, dim=0).float() if len(x_val) > 0 else []
    y_val = torch.cat(y_val, dim=0).float() if len(y_val) > 0 else []
    x_test = torch.cat(x_test, dim=0).float() if len(x_test) > 0 else []
    y_test = torch.cat(y_test, dim=0).float() if len(y_test) > 0 else []

    print("One-step data info:")
    print("x_train shape:", len(x_train) if isinstance(x_train, list) else x_train.shape)
    
    # Multi-step fine-tuning List generation
    multi_step_x_train_list, multi_step_y_train_list = [], []
    multi_step_x_val_list, multi_step_y_val_list = [], []
    multi_step_x_test_list, multi_step_y_test_list = [], []

    def create_multistep(data_list, n_steps, dest_x, dest_y):
        for n in range(2, n_steps + 1):
            n_xlist, n_ylist = [], []
            for traj in data_list:
                traj_copy = torch.clone(traj)
                if normalizer:
                    traj_copy = normalizer.encode(traj_copy.reshape(-1, traj_copy.shape[-2], traj_copy.shape[-1])).reshape(*traj_copy.shape)
                for i in range(traj_copy.shape[0] - history - n):
                    n_xlist.append(traj_copy[i : i + history])
                    n_ylist.append(traj_copy[i + history + n - 1])
            if len(n_xlist) > 0:
                dest_x.append(torch.stack(n_xlist, dim=0).float())
                dest_y.append(torch.stack(n_ylist, dim=0).float())

    create_multistep(train_data, num_steps, multi_step_x_train_list, multi_step_y_train_list)
    create_multistep(val_data, num_steps, multi_step_x_val_list, multi_step_y_val_list)
    create_multistep(test_data, num_steps, multi_step_x_test_list, multi_step_y_test_list)
    
    # Insert 1-step at the beginning
    multi_step_x_train_list.insert(0, x_train)
    multi_step_y_train_list.insert(0, y_train)
    multi_step_x_val_list.insert(0, x_val)
    multi_step_y_val_list.insert(0, y_val)
    multi_step_x_test_list.insert(0, x_test)
    multi_step_y_test_list.insert(0, y_test)

    # Create loaders
    multi_step_train_loaders = [torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=batch_size, shuffle=True, drop_last=True) for x,y in zip(multi_step_x_train_list, multi_step_y_train_list)] if len(x_train) > 0 else []
    multi_step_val_loaders = [torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=batch_size, shuffle=False, drop_last=True) for x,y in zip(multi_step_x_val_list, multi_step_y_val_list)] if len(x_val) > 0 else []
    multi_step_test_loaders = [torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=batch_size, shuffle=False, drop_last=True) for x,y in zip(multi_step_x_test_list, multi_step_y_test_list)] if len(x_test) > 0 else []

    # Index list used for shuffling during training
    multi_step_list = []
    for i in range(len(multi_step_train_loaders)):
        multi_step_list += [i for _ in range(len(multi_step_train_loaders[i]))]
    random.shuffle(multi_step_list)

    ntrain_list = [arr.shape[0] for arr in multi_step_x_train_list] if not isinstance(multi_step_x_train_list[0], list) else []
    nval_list = [arr.shape[0] for arr in multi_step_x_val_list] if not isinstance(multi_step_x_val_list[0], list) else []
    ntest_list = [arr.shape[0] for arr in multi_step_x_test_list] if not isinstance(multi_step_x_test_list[0], list) else []

    if return_normalizer:
        return multi_step_train_loaders, multi_step_val_loaders, multi_step_test_loaders, multi_step_list, ntrain_list, nval_list, ntest_list, normalizer
    else:
        return multi_step_train_loaders, multi_step_val_loaders, multi_step_test_loaders, multi_step_list, ntrain_list, nval_list, ntest_list


class AirfoilMultiStepDataset(torch.utils.data.Dataset):
    """
    A custom PyTorch Dataset for the Airfoil data that generates samples on-the-fly.
    Assumes the trajectories are already normalized.
    """
    def __init__(self, trajectories, history, step_ahead):
        super().__init__()
        self.trajectories = trajectories
        self.history = history
        self.step_ahead = step_ahead
        self.cumulative_samples = [0]
        
        for traj in self.trajectories:
            num_valid_samples = traj.shape[0] - self.history - self.step_ahead
            self.cumulative_samples.append(self.cumulative_samples[-1] + max(0, num_valid_samples))

    def __len__(self):
        return self.cumulative_samples[-1]

    def __getitem__(self, idx):
        traj_idx = next(i for i, total in enumerate(self.cumulative_samples) if total > idx) - 1
        time_idx = idx - self.cumulative_samples[traj_idx]
        traj = self.trajectories[traj_idx]

        x = traj[time_idx : time_idx + self.history]
        y = traj[time_idx + self.history + self.step_ahead - 1]
        return x, y

def load_airfoil_multi_step_efficient(steps, 
                                      history, 
                                      batch_size, 
                                      ntrain,
                                      nval,
                                      ntest,
                                      pre_tipping=True, 
                                      normalize='channel-wise', 
                                      normalize_using_pre_tipping_only=True,
                                      normalizer_object=None,
                                      re5000=False, 
                                      stall=False, 
                                      common_path='./airfoil_stall_2D/', 
                                      domain_keep_x=[0,1],
                                      domain_keep_y=[0,1],
                                      use_divergence_corrected=True,
                                      return_normalizer=False):
    """
    Memory-efficient loader using PyTorch Dataset.
    """
    if re5000 and stall:
        transition_times = stall_times_5k
    elif re5000 and not stall:
        transition_times = wake_transition_times_5k
    elif not re5000 and not stall:
        transition_times = wake_transition_times_1k
    else:
        raise RuntimeError("re1000 = True and stall = True are not compatible.")

    traj_postfix = '_5k' if re5000 else ''
    n_traj = ntrain + nval + ntest

    train_data, val_data, test_data, normalization_data = [], [], [], []
    for i in range(1, n_traj + 1):
        traj_name = f'traj{i}{traj_postfix}'
        traj_filename = 'divergence_corrected_traj.npy' if use_divergence_corrected else 'traj.npy'
        traj_path = os.path.join(common_path, traj_name, traj_filename)
        
        if not os.path.exists(traj_path):
             raise FileNotFoundError(f"Data file not found: {traj_path}")

        traj = torch.tensor(np.load(traj_path)).float()
        nt, _, nx, ny = traj.shape

        time = np.genfromtxt(os.path.join(common_path, traj_name, 'time.dat'))
        tipping_idx = np.argmin(np.abs(time - transition_times[traj_name]))
        cutoff_idx = tipping_idx if pre_tipping else len(traj)
        assert nt == len(time)

        x_truncate = [int(domain_keep_x[0] * nx), int(domain_keep_x[1] * nx)]
        y_truncate = [int(domain_keep_y[0] * ny), int(domain_keep_y[1] * ny)]
        truncated_traj = traj[:, :, x_truncate[0]:x_truncate[1], y_truncate[0]:y_truncate[1]]

        if i <= ntrain:
            train_data.append(truncated_traj[:cutoff_idx])
            if normalize_using_pre_tipping_only:
                normalization_data.append(traj[:tipping_idx, :, x_truncate[0]:x_truncate[1], y_truncate[0]:y_truncate[1]])
        elif i > ntrain and i <= ntrain + nval:
            val_data.append(truncated_traj[:cutoff_idx])
        else:
            test_data.append(truncated_traj[:cutoff_idx])

    normalizer = None
    if normalize:
        if normalizer_object is None:
            y_normalize = []
            source_data = normalization_data if normalize_using_pre_tipping_only and normalization_data else train_data
            for traj in source_data:
                for i in range(traj.shape[0] - history - 1):
                    y_normalize.append(traj[i + history])
            
            if y_normalize:
                y_normalize = torch.stack(y_normalize, dim=0).float()
                if normalize == 'channel-wise': reduce_dims = list(range(y_normalize.ndim))
                elif normalize == 'pixel-wise': reduce_dims = [0]
                else: raise NotImplementedError("Normalization mode not supported.")
                normalizer = UnitGaussianNormalizer(y_normalize, reduce_dim=reduce_dims)
        else:
            normalizer = normalizer_object
            
    if normalizer:
        train_data = [normalizer.encode(traj) for traj in train_data]
        val_data = [normalizer.encode(traj) for traj in val_data]
        test_data = [normalizer.encode(traj) for traj in test_data]

    # --- DataLoader Creation ---
    if isinstance(steps, int):
        steps_to_iterate = range(1, steps + 1)
    elif isinstance(steps, (list, tuple)):
        steps_to_iterate = sorted(steps)
    else:
        raise TypeError("Argument 'steps' must be an integer or a list/tuple of integers.")
        
    multi_step_train_loaders, multi_step_val_loaders, multi_step_test_loaders = [], [], []

    for n in steps_to_iterate:
        if train_data:
            train_dataset = AirfoilMultiStepDataset(train_data, history, step_ahead=n)
            multi_step_train_loaders.append(torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True))
        if val_data:
            val_dataset = AirfoilMultiStepDataset(val_data, history, step_ahead=n)
            multi_step_val_loaders.append(torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True))
        if test_data:
            test_dataset = AirfoilMultiStepDataset(test_data, history, step_ahead=n)
            multi_step_test_loaders.append(torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True))
            
    ntrain_list = [len(loader.dataset) for loader in multi_step_train_loaders]
    nval_list = [len(loader.dataset) for loader in multi_step_val_loaders]
    ntest_list = [len(loader.dataset) for loader in multi_step_test_loaders]

    multi_step_list = []
    if multi_step_train_loaders:
        for i in range(len(multi_step_train_loaders)):
            multi_step_list += [i] * len(multi_step_train_loaders[i])
        random.shuffle(multi_step_list)

    if return_normalizer:
        return multi_step_train_loaders, multi_step_val_loaders, multi_step_test_loaders, multi_step_list, ntrain_list, nval_list, ntest_list, normalizer
    else:
        return multi_step_train_loaders, multi_step_val_loaders, multi_step_test_loaders, multi_step_list, ntrain_list, nval_list, ntest_list