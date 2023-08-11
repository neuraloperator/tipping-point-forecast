############################################################
##
## Adapted from official Markov neural operator implementation 
## https://github.com/neuraloperator/markov_neural_operator/blob/main/data_generation/lorenz/odelibrary.py
##
############################################################

#!/usr/bin/env python
# # -*- coding: utf-8 -*-
import pdb
import numpy as np
import pickle
from time import time
from scipy.integrate import solve_ivp
from odelibrary import L63_time_dep

import matplotlib
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data_list = []

    # PARAMETERS
    T1 = -600
    T2 = 200
    dt = 0.001
    T0 = T1 - 50

    num_trajectories = 15

    # Taken from Druvit et al., 2022
    rho_0 = 154
    rho_1 = 8
    tau = 100
    b = lambda t: rho_0 + rho_1 * np.exp(t / tau)


    # read in ODE class
    l63 = L63_time_dep(b=b)

    # swap input order for expectation of scipy.integrate.solve_ivp
    f_ode = lambda t, y: l63.rhs(y, t)

    for i in range(num_trajectories):
        print("Generating trajectory", i+1)
        # INTEGRATION
        u0 = l63.get_inits()

        print("Integrating through an initial transient phase to reach the attractor...")
        tstart = time()
        t_span = [T0, T1]
        t_eval = np.array([T1])
        sol = solve_ivp(fun=f_ode, t_span=t_span, y0=u0, t_eval=t_eval, max_step=dt, method='RK45')

        print('took', '{:.2f}'.format((time() - tstart)/60),'minutes')

        print("Integrating trajectory on the attractor...")
        tstart = time()
        u0 = np.squeeze(sol.y)
        t_span = [T1, T2]
        t_eval_tmp = np.arange(T1, T2, dt)
        t_eval = np.zeros(len(t_eval_tmp)+1)
        t_eval[:-1] = t_eval_tmp
        t_eval[-1] = T2
        sol = solve_ivp(fun=f_ode, t_span=t_span, y0=u0, t_eval=t_eval, max_step=dt, method='RK45')
        u = sol.y.T
        data_list.append(u)
        
        print('took', '{:.2f}'.format((time() - tstart)/60),'minutes')

    data_arr = np.array(data_list)
    data = {
        "T1":T1,
        "T2":T2,
        "dt":dt,
        "data":data_arr,
    }

    # save data
    with open("nonstationary_lorenz_data_15_trajectories.p", "wb") as file:
        # Pickle the "data" dictionary using the highest protocol available.
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)

    # # plot trajectory
    # T_plot = 20
    # n_plot = int(T_plot/dt)
    # K = u.shape[1] #number of ode states
    # fig, axes = plt.subplots(nrows=K, ncols=1,figsize=(12, 6))
    # times = dt*np.arange(n_plot)
    # pdb.set_trace()
    # for k in range(K):
    #     axes[k].plot(times, u[:n_plot,k], linewidth=2)
    #     axes[k].set_ylabel('X_{k}'.format(k=k))
    # axes[k].set_xlabel('Time')
    # fig.suptitle('Lorenz 63 Trajectory simulated with RK45')
    # plt.savefig('l63trajectory')
    # plt.close()
