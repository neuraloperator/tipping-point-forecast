"""
Generating nonstationary KS equation with time-varying viscosity. 
Based on stationary KS code here:
https://scicomp.stackexchange.com/questions/37336/solving-numerically-the-1d-kuramoto-sivashinsky-equation-using-spectral-methods

Parameters:
	L : system size
	nx : spatial discretization
	T0 : initial time to start integrating (e.g., to reach attractor)
	T1 : initial time to record
	T2 : final time
	dt : solver timestep
	eval_sub : temporal subsampling rate of final data
	nu : viscosity; function of time t

"""
import numpy as np
import time

def solve_ks(u0, L, T_in, T_out, dt, nu, eval_sub):
	"""
		u0 : initial condition of shape (nx,)
		L : size of physical domain
		T_in : in time
		T_out : out time
		dt : solver time step
		nu : function of time t
		eval_sub : temporal subsampling rate of final data
	"""
	nx = u0.shape[0]
	nt = int((T_out - T_in) / dt)

	k = np.arange(-nx/2, nx/2, 1) # wave number mesh
	t = np.linspace(start=T0, stop=T2, num=nt)
	x = np.linspace(start=0, stop=L, num=nx)

	# solution mesh in real space
	u = np.ones((nx, nt))
	# solution mesh in Fourier space
	u_hat = np.ones((nx, nt), dtype=complex)
	u_hat2 = np.ones((nx, nt), dtype=complex)

	# Fourier transform of initial condition
	u0_hat = (1 / nx) * np.fft.fftshift(np.fft.fft(u0))
	u0_hat2 = (1 / nx) * np.fft.fftshift(np.fft.fft(u0**2))

	# set initial condition in real and Fourier mesh
	u[:,0] = u0
	u_hat[:,0] = u0_hat
	u_hat2[:,0] = u0_hat2

	# resolve EDP in Fourier space
	for idx, time in enumerate(np.arange(T_in, T_out, dt)):
		if idx == nt - 1:
			break
		uhat_current = u_hat[:,idx]
		uhat_current2 = u_hat2[:,idx]
		if idx == 0:
			uhat_last = u_hat[:,0]
			uhat_last2 = u_hat2[:,0]
		else:
			uhat_last = u_hat[:,idx-1]
			uhat_last2 = u_hat2[:,idx-1]

		# Fourier Transform of the linear operator
		FL = (((2 * np.pi) / L) * k) ** 2 - nu(time) * (((2 * np.pi) / L) * k) ** 4
		# Fourier Transform of the non-linear operator
		FN = - (1 / 2) * ((1j) * ((2 * np.pi) / L) * k)

		# compute solution in Fourier space through a finite difference method
		# Cranck-Nicholson + Adam 
		u_hat[:,idx+1] = (1 / (1 - (dt / 2) * FL)) * ( (1 + (dt / 2) * FL) * uhat_current + ( ((3 / 2) * FN) * (uhat_current2) - ((1 / 2) * FN) * (uhat_last2) ) * dt )
		# go back in real space
		u[:,idx+1] = np.real(nx * np.fft.ifft(np.fft.ifftshift(u_hat[:,idx+1])))

		# clean the imaginary part contribution in u_hat
		u_hat[:,idx+1] = (1 / nx) * np.fft.fftshift(np.fft.fft(u[:,idx+1]))
		u_hat2[:,idx+1] = (1 / nx) * np.fft.fftshift(np.fft.fft(u[:,idx+1]**2))
	
	return u[:, ::eval_sub]

if __name__ == '__main__':
	ntraj = 5
	
	save_path = "tipping_KS_data_200_traj_dt_0_01.npy"
	#save_path = "data/nonstationary_KS_superres_data.npy"
	L = 2 * np.pi
	nx = 256
	x = np.linspace(start=0, stop=L, num=nx)

	T0 = -100
	T1 = 0 # recording start time
	T2 = 100
	dt = 0.001
	eval_sub = 100 # superres: 10 # subsampling of final data

	nu0 = 0.073
	nu_scale = 0.0034
	nu_tau = 75.3
	nu = lambda t : nu0 + nu_scale * np.exp(t / nu_tau) # tipping point occurs around t = 67 for initial condition given below

	trajectories = []

	for i in range(ntraj):
		print("Generating trajectory", i + 1)

		# initial condition 
		u0 = np.cos(x/16) * (1 + np.sin(x / 16)) + np.random.uniform(-0.5, 0.5, x.shape)
		u_intermediate = solve_ks(u0, L, T0, T1, dt, nu, eval_sub)[:, -1] # getting to attractor

		start_time = time.time()
		u = solve_ks(u_intermediate, L, T1, T2, dt, nu, eval_sub)
		print(f"  -> Generated in {time.time() - start_time} seconds")
		
		trajectories.append(u)

	# Save result
	traj = np.array(trajectories)
	np.save(save_path, traj)
	print("Data saved to", save_path)
	print("dt:", dt * eval_sub)
	print("T1:", T1)
	print("T2:", T2)