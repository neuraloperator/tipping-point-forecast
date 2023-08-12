import torch.nn.functional as F
from timeit import default_timer
from utilities import *
from tqdm import tqdm
import pickle
import sys
import random

sys.path.append('../')
from RNO_2d import *

torch.manual_seed(0)
np.random.seed(0)

def round_down(num, divisor): # rounds `num` down to nearest multiple of `divisor`
    return num - (num % divisor)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ntest = 40

    experiment_name = 'MODEL/PATH'
    model = torch.load('model/' + experiment_name, map_location=device)

    dim = 1

    batch_size = 64

    sub = 4 # spatial subsample
    T = 16 # input last T time steps and output next T
    n_intervals = 5

    n_iter = 100

    ################################################################
    # data and preprocessing
    ################################################################

    t1 = default_timer()
    data = np.load('data/tipping_KS_data_200_traj_dt_0_1.npy')[:, ::sub]
    data = torch.tensor(data)

    n_time = data.shape[2]
    n_x = data.shape[1]

    test = data[-ntest:]

    idx = round_down(test.shape[2], T)
    test = test[:,:,:idx]

    test = torch.stack(torch.split(test, T, dim=-1)).permute(1,0,3,2).unsqueeze(-1)


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
        model_input = torch.randn(1, n_intervals, T, n_x, dim).to(device)

        # Time: https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        out = model.predict(model_input, num_steps=num_steps)
        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()

        if i > 0: # exclude first run
            times.append(start.elapsed_time(end))

    print(f"Mean inference time over {n_iter} iterations: {sum(times) / len(times)} ms")
