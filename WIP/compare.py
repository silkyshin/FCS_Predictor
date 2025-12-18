import numpy as np
import matplotlib.pyplot as plt
import argparse

# 1. Parse command line arguments
parser = argparse.ArgumentParser(description="Compare FCS data to ODE predictions")
parser.add_argument('--csv_exp', type=str, required=True,
                    help='CSV file with experimental FCS data (time,G0,tau)')
parser.add_argument('--csv_ode', type=str, required=True,
                    help='CSV file with ODE solution (time,M1,M2,...)')
parser.add_argument('--frac', type=float, default=0.002,
                    help='Fraction of fluorescent protein')
parser.add_argument('--sfactor', type=float, default=1.0,
                    help='Scaling factor for diffusion times (v in tau ~ n^v)')
args = parser.parse_args()

# 2. Load experimental data
exp_data = np.genfromtxt(args.csv_exp, delimiter=',', names=True)
t_exp = exp_data['time']
G0_exp = exp_data['G0']
tau_exp = exp_data['tau']

# 3. Load ODE solution
ode_data = np.genfromtxt(args.csv_ode, delimiter=',', names=True)
t_ode = ode_data['t']
M_t = np.vstack([ode_data[name] for name in ode_data.dtype.names if name != 't']).T

# 4. Number of monomers per species
n_species = M_t.shape[1]
n = np.arange(1, n_species+1)

# 5. Predicted number of fluorescent monomers
P = M_t * n * args.frac

# 6. Predicted G0(t)
G0_pred = np.sum(P**2, axis=1) / np.sum(P, axis=1)**2

# 7. Predicted diffusion time
v = args.sfactor
tau_n = n**v
tau_pred = np.sum(P * tau_n, axis=1) / np.sum(P, axis=1)

# 8. Plot comparison
plt.figure(figsize=(8,6))

plt.subplot(2,1,1)
plt.plot(t_exp, G0_exp, 'o', label='Experimental', color='blue')
plt.plot(t_ode, G0_pred, '-', label='Predicted', color='cyan')
plt.xlabel('Time (s)')
plt.ylabel('G0')
plt.title('FCS Amplitude G0 vs Time')
plt.legend()
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(t_exp, tau_exp, 'o', label='Experimental', color='red')
plt.plot(t_ode, tau_pred, '-', label='Predicted', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Diffusion time (s)')
plt.title('FCS Diffusion Time vs Time')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
