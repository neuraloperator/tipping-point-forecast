import torch
from neuralop.models import FNO, RNO
from neuralop.utils import count_model_params
from neuralop import LpLoss
import os

from utilities import load_airfoil_multi_step_efficient, get_grid_2d, wake_transition_times_1k, wake_transition_times_5k, stall_times_5k
from models import GRU_spatial_net 

if __name__ == '__main__':
    ntrain = 7
    nval = 0 # calibration data
    ntest = 3 # unseen trajectory

    batch_size = 1
    history = 4
    comp_steps = [2, 4, 6, 8]

    positional_encoding = True
    normalize = 'channel-wise'
    dim = 2
    use_divergence_corrected = True

    domain_keep_x = [0.375, 0.75] 
    domain_keep_y = [0.25, 0.83]

    ########### CHANGE ###########
    model_path = '' # UPDATE
    re5000 = True
    stall = True
    ########### ###### ###########

    # Load data
    traj_postfix = '_5k' if re5000 else ''
    if re5000 and stall:
        transition_times = stall_times_5k
        problem_setting = 'Re5000 stall'
    elif re5000 and not stall:
        transition_times = wake_transition_times_5k
        problem_setting = 'Re5000 wake'
    elif not re5000 and not stall:
        transition_times = wake_transition_times_1k
        problem_setting = 'Re1000 wake'
    else:
        raise RuntimeError("re1000 = True and stall = True are not compatible.")
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    _, _, multi_step_test_loaders, _, _, _, ntest_list, normalizer = load_airfoil_multi_step_efficient(comp_steps, 
                                            history, 
                                            batch_size, 
                                            ntrain=ntrain, 
                                            nval=nval,
                                            ntest=ntest,
                                            pre_tipping=True,
                                            normalize=normalize,
                                            re5000=re5000,
                                            stall=stall,
                                            domain_keep_x=domain_keep_x,
                                            domain_keep_y=domain_keep_y,
                                            use_divergence_corrected=use_divergence_corrected,
                                            return_normalizer=True)


    # Load model and loss
    model = torch.load(os.path.join('model_paths/', model_path), map_location=device)
    print("Model parameters:", count_model_params(model))

    lploss = LpLoss(d=2, p=2, reduce_dims=[0,1])

    # Evaluate
    model.eval()
    print(f"Evaluating {model_path.split('_')[0]} on {problem_setting}...")
    test_l2_list = [0.0 for i in range(len(multi_step_test_loaders))]
    with torch.no_grad():
        for i, n in enumerate(range(len(multi_step_test_loaders))):
            num_steps = comp_steps[i]
            for x, y in multi_step_test_loaders[n]:
                x, y = x.to(device), y.to(device)

                if 'MNO' in model_path:          # MNO inference
                    x = x[:,-1]
                    u = model(x)
                    for i in range(num_steps - 1):
                        u = model(u)
                
                elif 'GRU' in model_path:         # RNN inference (GRU_spatial_net)
                    if positional_encoding:
                        grid = get_grid_2d((x.shape[0], x.shape[1], 1, x.shape[-2], x.shape[-1]), device)
                        x = torch.cat((x, grid), dim=2)
                    u = model.predict(x, num_steps, grid_function=get_grid_2d)[:,-1]

                else:                             # RNO inference
                    u = model.predict(x, num_steps)[:,-1]

                test_l2_list[n] += lploss(u, y).item()

    for i, n in enumerate(comp_steps):
        test_l2_list[i] /= ntest_list[i]
        print(f"L2 loss for {n} steps: {test_l2_list[i]}")