import torch.nn.functional as F
from timeit import default_timer
from utilities3 import *
from tqdm import tqdm
import pickle

from fourier_1d_ode import *

torch.manual_seed(0)
np.random.seed(0)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    sub = 5 # temporal subsampling rate
    T = 64 # input last T time steps and output next T
    dim = 3 # dimensionality of ODE

    experiment_name = 'PATH/TO/MODEL'
    model = torch.load(experiment_name, map_location=device)

    model_in_dim = dim

    batch_size = 64

    n_intervals = 1

    n_iter = 10

    ################################################################
    # read data
    ################################################################

    # Data is of the shape (number of samples, grid size)
    with open("nonstationary_lorenz_data_15_trajectories.p", "rb") as file:
        data_dict = pickle.load(file)

    dt = data_dict['dt']
    T1 = data_dict['T1']
    T2 = data_dict['T2']
    data = data_dict['data'][:,::sub]
    data = data[:,:-1] # Make shape easier to divide
    
    test_trajectories = 5

    test = data[-test_trajectories:]

    test_idx = round_down(test.shape[1], T)
    test = test[:,:test_idx]

    test = np.reshape(test, (test_trajectories, -1, T, dim))

    ############################################################
    #### Benchmarking
    ############################################################
    # Model memory and size: https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs # in bytes

    print("Model memory footprint: " + str(mem) + " bytes")
    print()

    num_steps = test.shape[1]
    print("Bechmark number of steps:", num_steps)
    print()
    times = []
    for i in range(n_iter + 1):
        model_input = torch.randn(1, T, model_in_dim).to(device)

        # Time: https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        
        out = model(model_input)
        for _ in range(num_steps - 1):
            out = model(out)

        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()

        if i > 0: # exclude first run
            times.append(start.elapsed_time(end))

    print(f"Mean inference time over {n_iter} iterations: {sum(times) / len(times)} ms")
