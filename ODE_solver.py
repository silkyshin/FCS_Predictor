import numpy as np
from scipy.integrate import solve_ivp
import argparse
import importlib.util
import os


# 1. Parse command line argument

parser = argparse.ArgumentParser(description="Solve stepwise oligomerization ODEs")
parser.add_argument('--model', type=str, required=True,
                    help='Path to the model Python file, e.g., ./ODE/model_1.py')
args = parser.parse_args()
model_path = args.model

# 2. Dynamically import model

module_name = os.path.splitext(os.path.basename(model_path))[0]
spec = importlib.util.spec_from_file_location(module_name, model_path)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)

# The model file needs to have a function called ode_model(t, M, k)

# 3. Ask for kinetics constants depending on the model
# I want it to list the kinetic constants that need to be defined in the model first
# 3.1 Show model info
try:
  print(f"Model: {model_module.model_name}")
  print("Kinetic constants needed")
  for name in model_module.kinetic_constants:
    print(f" - {name}")
except AttributeError:
  raise AttributeError("Model file must define 'model_name' and 'kinetic_constant' variables")
# 3.2 Ask for the constants
k_input = input(f"\nEnter {len(model_module.kinetic_constants)} kinetic constants (comma-separated):\n")
k_list = [float(x.strip()) for x in k_input.split(',')]
if len(k_list) != len(model_module.kinetic_constants):
    raise ValueError(f"You must enter exactly {len(model_module.kinetic_constants)} values")
k = np.array(k_list)
print("\nKinetic constants used:")
for name, val in zip(model_module.kinetic_constants, k):
  print(f" {name} = {val}")

# 4. Set initial concentrations
num_species = None
try:
  num_species = len(model_module.ode_model.__code__.co_varnames) - 2
except:
  pass

M0_input = input("\nEnter initial concentrations (comma separated):")
M0 = np.array([float(x.strip()) for x in M0_input.split(',')])
if len(M0) < 1:
  raise ValueError("You must enter at least one concentration")

# 5. Time span
t_start = float(input("\nEnter start time (s): "))
t_end   = float(input("Enter end time (s): "))
num_points = int(input("Enter the number of timepoints:"))
t_eval = np.linspace(t_start, t_end, num_points)
t_span = (t_start, t_end)

# 6. Solve ODE
sol = solve_ivp(lambda t, M: model_module.ode_model(t, M, k),
                t_span, M0, t_eval=t_eval, method='RK45')

# 7. Save solution to CSV
output_file = input("\nEnter filename to save solution (e.g., ODE_solution.csv): ")
header = "t," + ",".join([f"M{i+1}" for i in range(sol.y.shape[0])])
data_to_save = np.column_stack([sol.t, sol.y.T])  # time in first column, then species
np.savetxt(output_file, data_to_save, delimiter=',', header="t," + ",".join([f"M{i+1}" for i in range(sol.y.shape[0])]), comments='')

print(f"Solution saved to {output_file}")
