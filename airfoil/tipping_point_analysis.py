import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from utilities import divergence_2d, load_airfoil_multi_step_efficient, get_grid_2d, wake_transition_times_1k, wake_transition_times_5k, stall_times_5k
from neuralop.models import FNO, RNO
from neuralop.utils import count_model_params
from models import GRU_spatial_net

# ---------------- CONFIGURATION ----------------

# Select the setting and model to analyze here
# options: 're1000_wake', 're5000_wake', 're5000_stall'
SETTING = 're1000_wake' 
# options: 'MNO', 'RNO', 'GRU'
MODEL_TYPE = 'MNO' 

# Hyperparameters
ntrain = 7
nval = 2 # calibration data
ntest = 1 # unseen trajectory
batch_size = 1
history = 4
comp_steps = 10 # number of steps to compose (forecast horizon)

domain_keep_x = [0.375, 0.75]
domain_keep_y = [0.25, 0.83]
positional_encoding = True
normalize = 'channel-wise'
dim = 2
use_divergence_corrected = True
plot_legend = False

# ---------------- SETUP PATHS ----------------

if SETTING == 're1000_wake':
    re5000 = False
    stall = False
    pred_stall_pre_wake = False
    # Update these filenames to match your saved models
    if MODEL_TYPE == 'MNO':
        model_path = ''
    elif MODEL_TYPE == 'RNO':
        model_path = ''
    elif MODEL_TYPE == 'GRU':
        model_path = ''

elif SETTING == 're5000_wake':
    re5000 = True
    stall = False
    pred_stall_pre_wake = True
    if MODEL_TYPE == 'MNO':
        model_path = ''
    elif MODEL_TYPE == 'RNO':
        model_path = ''

elif SETTING == 're5000_stall':
    re5000 = True
    stall = True
    pred_stall_pre_wake = False
    if MODEL_TYPE == 'MNO':
        model_path = ''
    elif MODEL_TYPE == 'RNO':
        model_path = ''

else:
    raise ValueError("Invalid setting")

# Path checking
full_model_path = os.path.join('model_paths', model_path)
if not os.path.exists(full_model_path):
    print(f"WARNING: Model file not found at {full_model_path}")
    print("Please ensure you have trained the model or updated the filename in the script.")

common_path='./airfoil_stall_2D/'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# ---------------- FUNCTIONS ----------------

def div_loss(u, transpose_spatial_dims=True):
    """Computes L2 divergence loss assuming the last dimension of the grid is mapped onto [0,1]."""
    if transpose_spatial_dims: # Transpose last two dimensions
        u = torch.transpose(u, -1, -2)

    div = divergence_2d(u, domain_spacing=1./u.shape[-1])
    return torch.norm(div, p='fro', dim=(-1,-2)) / math.sqrt(div.shape[-1] * div.shape[-2])

def run_inference(model, x, num_steps, model_type, device):
    """Unified inference function for all model types."""
    x = x.to(device)
    
    if model_type == 'MNO':
        # MNO takes (B, C, H, W) - we take the last history step
        curr = x[:, -1]
        out = model(curr)
        for _ in range(num_steps - 1):
            out = model(out)
        return out

    elif model_type == 'RNO':
        # RNO takes (B, T, C, H, W)
        # Predict returns full sequence, we take the last one
        return model.predict(x, num_steps)[:, -1]

    elif model_type == 'GRU' or model_type == 'RNN':
        # GRU requires manual grid concatenation
        if positional_encoding:
            grid = get_grid_2d((x.shape[0], x.shape[1], 1, x.shape[-2], x.shape[-1]), device)
            x = torch.cat((x, grid), dim=2)
        # Predict
        return model.predict(x, num_steps, grid_function=get_grid_2d)[:, -1]
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# ---------------- MAIN EXECUTION ----------------

if __name__ == '__main__':
    
    # 1. Load Data
    print(f"Loading data for {SETTING}...")
    
    # We use a dummy 1-step to get the full datasets, then we'll iterate manually
    _, val_loaders, test_loaders, _, _, _, _, normalizer = load_airfoil_multi_step_efficient(
        steps=[comp_steps], # Only load the horizon we care about
        history=history,
        batch_size=batch_size,
        ntrain=ntrain,
        nval=nval,
        ntest=ntest,
        pre_tipping=False, 
        normalize=normalize,
        re5000=re5000,
        stall=stall,
        domain_keep_x=domain_keep_x,
        domain_keep_y=domain_keep_y,
        return_normalizer=True,
        use_divergence_corrected=use_divergence_corrected
    )

    # For calibration, we need pre-tipping validation data
    _, pre_tipping_val_loaders, _, _, _, _, _, _ = load_airfoil_multi_step_efficient(
        steps=[comp_steps],
        history=history,
        batch_size=batch_size,
        ntrain=ntrain,
        nval=nval,
        ntest=ntest,
        pre_tipping=True,
        normalize=normalize,
        normalizer_object=normalizer,
        re5000=re5000,
        stall=True if pred_stall_pre_wake else stall,
        domain_keep_x=domain_keep_x,
        domain_keep_y=domain_keep_y,
        return_normalizer=True,
        use_divergence_corrected=use_divergence_corrected
    )

    # 2. Load Model
    print(f"Loading model from {full_model_path}...")
    model = torch.load(full_model_path, map_location=device)
    model.eval()

    # 3. Calibration (ECDF)
    print("Calculating divergence loss on calibration set...")
    pde_loss_samples = []
    
    val_dataset = pre_tipping_val_loaders[0].dataset 
    
    with torch.no_grad():
        for i in range(len(val_dataset)):
            x_batch, _ = val_dataset[i]
            x_batch = x_batch.unsqueeze(0) # Add batch dim
            
            u = run_inference(model, x_batch, comp_steps, MODEL_TYPE, device)
            
            # Reshape for div_loss
            u = u.reshape((x_batch.shape[0], dim, x_batch.shape[-2], x_batch.shape[-1]))
            pde_loss_samples.append(div_loss(u).cpu().numpy())

    pde_loss_samples = np.array(pde_loss_samples).flatten()

    # Plot Histogram
    plt.figure(figsize=(8,6))
    plt.hist(pde_loss_samples, bins=30)
    plt.title("Histogram of divergence loss, pre-tipping")
    plt.xlabel("Divergence Loss")
    plt.ylabel("Frequency")
    plt.show()

    # ECDF Calculation
    ecdf = lambda x: np.sum(pde_loss_samples < x) / len(pde_loss_samples)
    
    plt.figure(figsize=(8,6))
    plt.plot(np.sort(pde_loss_samples), np.linspace(0, 1, len(pde_loss_samples), endpoint=False))
    plt.title("Empirical CDF of divergence Loss")
    plt.show()

    # DKW Inequality
    delta = 1e-4
    eps = np.sqrt(np.log(2 / delta) / (2 * len(pde_loss_samples)))
    print(f"DKW Epsilon: {eps}")

    alpha_list = [eps] + [round(eps, 2) + i * 0.01 for i in range(1,4)]

    # 4. Tipping Point Analysis on Test Trajectory
    print("Analyzing test trajectory...")
    traj_idx = 0
    test_dataset = test_loaders[0].dataset
    
    loss = []
    
    # We iterate through the whole test trajectory in the dataset
    with torch.no_grad():
        for i in range(len(test_dataset)):
            x_step, _ = test_dataset[i]
            x_step = x_step.unsqueeze(0)
            
            u = run_inference(model, x_step, comp_steps, MODEL_TYPE, device)
            
            u = u.reshape((x_step.shape[0], dim, x_step.shape[-2], x_step.shape[-1]))
            loss.append(div_loss(u).cpu().numpy())

    loss = np.array(loss).flatten()

    # Metadata for plotting
    traj_postfix = '_5k' if re5000 else ''
    if re5000 and stall:
        transition_times = stall_times_5k
    elif re5000 and not stall:
        if pred_stall_pre_wake:
            transition_times = stall_times_5k
        else:
            transition_times = wake_transition_times_5k
    elif not re5000 and not stall:
        transition_times = wake_transition_times_1k
    
    # Determine which trajectory name corresponds to ntrain + nval + traj_idx + 1
    traj_num = ntrain + nval + traj_idx + 1
    traj_name = f'traj{traj_num}{traj_postfix}'
    
    time_file = os.path.join(common_path, traj_name, 'time.dat')
    if os.path.exists(time_file):
        time = np.round(np.genfromtxt(time_file), decimals=1)
        dt = time[1] - time[0]
        true_tipping_time = transition_times.get(traj_name, np.nan)
    else:
        print(f"Warning: Time file not found at {time_file}. Mocking time data.")
        time = np.arange(len(loss) + 50) * 0.1
        dt = 0.1
        true_tipping_time = 0

    print("True tipping time:", true_tipping_time)

    # 5. Plotting Results
    
    # Try using LaTeX fonts, fall back if not available
    try:
        from matplotlib import rc
        rc('text', usetex=True)
        plt.rcParams['font.family'] = 'serif'
        tex_available = True
    except:
        print("LaTeX rendering not available. Using standard fonts.")
        tex_available = False

    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['lines.linewidth'] = 2.5

    fig, axes = plt.subplots(len(alpha_list), 1, figsize=(6, len(alpha_list) * 4))
    if len(alpha_list) == 1: axes = [axes] # Handle single plot case
    else: axes = axes.flatten()

    prediction_times = time[:len(loss)] + history * dt

    for i, alpha in enumerate(alpha_list):
        print(f"Alpha: {alpha}")
        tipping_time = np.inf
        
        for loss_val, t in zip(loss, prediction_times):
            if alpha >= 1 - ecdf(loss_val) + eps:
                tipping_time = t
                break

        if tipping_time != np.inf:
            print(f"Predicted tipping point at time {tipping_time:.2f}")
        else:
            print("No tipping point found")

        ax = axes[i]
        ax.plot(prediction_times, loss, color='black')
        
        # Formatting labels based on TeX availability
        if tex_available:
            label1 = r'\textbf{prediction}' if i == 0 else None
            label2 = r'\textbf{tipping point}' if i == 0 else None
            label3 = r'\textbf{forecast time}'
            title_str = r'\textbf{$\alpha$ = ' + str(round(alpha, 2)) + ' }'
            xlab = r'\textbf{Time ($tU_\infty/C$)}'
            ylab = r'\textbf{Physics loss}'
        else:
            label1 = 'prediction' if i == 0 else None
            label2 = 'tipping point' if i == 0 else None
            label3 = 'forecast time'
            title_str = f'alpha = {round(alpha, 2)}'
            xlab = 'Time'
            ylab = 'Physics loss'

        ax.axvline(x = tipping_time + comp_steps * dt, color='r', ls='-.', label = label1)
        ax.axvline(x = true_tipping_time, color='b', ls='dotted', label = label2)
        ax.axvline(x=tipping_time, color='green', ls='-.', label=label3)
        ax.set_title(title_str)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)

    if plot_legend:
        legend = axes[0].legend(loc='upper left', frameon=False)
        legend.get_frame().set_linewidth(0)    
        
    print(f"\nTitle: {SETTING} {MODEL_TYPE} tipping point forecasting, {round(comp_steps * dt, 1)} seconds ahead")
    plt.tight_layout()
    plt.show()