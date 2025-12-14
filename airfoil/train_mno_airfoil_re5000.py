import torch
import torch.nn as nn
from timeit import default_timer
from tqdm import tqdm
import random
import os

from utilities import load_airfoil_multi_step
from neuralop.utils import count_model_params
from neuralop.models import FNO
from neuralop import LpLoss

if __name__ == '__main__':
    ntrain = 7
    nval = 0
    ntest = 3

    train_up_to_tipping_point = True
    stall = True
    
    domain_keep_x = [0.375, 0.75] 
    domain_keep_y = [0.25, 0.83]

    modes1 = 26
    modes2 = 26
    width = 28
    num_layers = 4
    positional_encoding = True
    normalize = 'channel-wise'
    use_divergence_corrected = True

    dim = 2
    batch_size = 5
    history = 4

    multi_step_num_steps = 3
    multi_step_ep = 50
    lr = 5e-4                                       
    multi_step_scheduler_step = 10
    multi_step_scheduler_gamma = 0.5

    print("Multi-step epochs:", multi_step_ep)
    print("Learning rate:", lr)
    print("Scheduler step:", multi_step_scheduler_step)
    print("Scheduler gamma:", multi_step_scheduler_gamma)
    print()

    tipping_point_type = 'stall' if stall else 'wake'

    path = 'MNO_airfoil_div_corrected_normalization_fixed_'+tipping_point_type+'_re5000_multi_step' + str(multi_step_num_steps) + '_N' +str(ntrain) + '_ep' + str(multi_step_ep) + '_1m' + str(modes1) + '_2m' + str(modes2) + '_w' + str(width) + 'domain_trunc' + \
            'x' + str(domain_keep_x)[1:-1].replace('.', '_').replace(', ', '_to_') + '_y' + str(domain_keep_x)[1:-1].replace('.', '_').replace(', ', '_to_')
    path_model = os.path.join('model_paths/', path)
    os.makedirs(os.path.dirname(path_model), exist_ok=True)

    # Load data
    multi_step_train_loaders, multi_step_val_loaders, multi_step_test_loaders, multi_step_list, ntrain_list, nval_list, ntest_list = load_airfoil_multi_step(multi_step_num_steps, 
                                                                                                                                                            history, 
                                                                                                                                                            batch_size, 
                                                                                                                                                            ntrain=ntrain, 
                                                                                                                                                            nval=nval,
                                                                                                                                                            ntest=ntest,
                                                                                                                                                            pre_tipping=train_up_to_tipping_point, 
                                                                                                                                                            normalize=normalize,
                                                                                                                                                            re5000=True,
                                                                                                                                                            stall=stall,
                                                                                                                                                            domain_keep_x=domain_keep_x,
                                                                                                                                                            domain_keep_y=domain_keep_y,
                                                                                                                                                            use_divergence_corrected=use_divergence_corrected)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FNO(
        n_modes=(modes1, modes2),
        hidden_channels=width,
        in_channels=dim, 
        out_channels=dim, 
        n_layers=num_layers, 
        positional_embedding="grid" if positional_encoding else None,
        domain_padding=[0.1, 0.1], 
    ).to(device)
    
    print("Model parameters:", count_model_params(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=multi_step_scheduler_step, gamma=multi_step_scheduler_gamma)

    lploss = LpLoss(d=2, p=2, reduce_dims=[0,1])

    print("Begin multi-step training ({} steps):".format(multi_step_num_steps))
    
    for ep in range(multi_step_ep):
        random.shuffle(multi_step_list)
        iters = [iter(loader) for loader in multi_step_train_loaders]

        model.train()
        t1 = default_timer()
        
        train_l2_list = [0.0 for _ in range(len(multi_step_train_loaders))]

        for n in tqdm(multi_step_list):
            num_steps = n + 1

            x, y = next(iters[n])
            x, y = x.to(device), y.to(device)

            x = x[:,-1]
            
            optimizer.zero_grad()
            out = model(x)

            for i in range(num_steps - 1):
                out = model(out)

            l2 = lploss(out, y)
            optimizer.zero_grad()
            l2.backward()

            optimizer.step()
            train_l2_list[n] += l2.item()
        
        scheduler.step()

        model.eval()
        test_l2_list = [0.0 for i in range(len(multi_step_test_loaders))]
        with torch.no_grad():
            for n in range(len(multi_step_test_loaders)):
                num_steps = n + 1
                for x, y in multi_step_test_loaders[n]:
                    x, y = x.to(device), y.to(device)
                    x = x[:,-1]

                    out = model(x)
                    for i in range(num_steps - 1):
                        out = model(out)

                    test_l2_list[n] += lploss(out, y).item()

        for n in range(len(multi_step_train_loaders)):
            train_l2_list[n] /= ntrain_list[n]
            test_l2_list[n] /= ntest_list[n]

        t2 = default_timer()
        print("Multi-step epoch:", ep + 1, "Time:", t2-t1, "Train L2:", train_l2_list, "Test L2:", test_l2_list)

    torch.save(model, path_model)
    print("Weights saved to", path_model)