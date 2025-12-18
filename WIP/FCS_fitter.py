import argparse
import importlib.util
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit kinetic ODE models to FCS data"
    )
    parser.add_argument("--csv", required=True, help="Experimental FCS CSV file")
    parser.add_argument("--model", required=True, help="Path to ODE model directory")
    return parser.parse_args()

def load_fcs_data(csv_file):
    df = pd.read_csv(csv_file)
    tau = df["tau"].values
    G_exp = df["G0"].values
    return tau, G_exp
  
def load_model(model_path):
    model_path = Path(model_path)

    model_file = model_path / "model.py"
    config_file = model_path / "config.py"

    if not model_file.exists() or not config_file.exists():
        raise FileNotFoundError("Model directory must contain model.py and config.py")

    def load_module(file, name):
        spec = importlib.util.spec_from_file_location(name, file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    model = load_module(model_file, "model")
    config = load_module(config_file, "config")

    return model, config

def solve_kinetics(odes, params, tau, y0):
    sol = solve_ivp(
        odes,
        (tau[0], tau[-1]),
        y0=y0,
        t_eval=tau,
        args=(params,),
        method="LSODA"
    )

    if not sol.success:
        raise RuntimeError("ODE solver failed")

    return sol.y

def residuals(params, tau, G_exp, model, y0):
    kinetic_sol = solve_kinetics(model.kinetic_odes, params, tau, y0)
    G_model = model.predict_fcs(tau, kinetic_sol, params)
    return G_model - G_exp

def fit_model(tau, G_exp, model, config):
    result = least_squares(
        residuals,
        x0=config.initial_params,
        args=(tau, G_exp, model, config.y0),
        bounds=(0, np.inf),
        method="trf"
    )
    return result

def reduced_chi_squared(G_exp, G_model, n_params):
    chi2 = np.sum((G_exp - G_model)**2)
    dof = len(G_exp) - n_params
    return chi2 / dof

def save_predicted_curve(tau, G_model, output_name="FCS_predicted.csv"):
    df = pd.DataFrame({
        "tau": tau,
        "G0_predicted": G_model
    })
    df.to_csv(output_name, index=False)

def main():
    args = parse_args()

    tau, G_exp = load_fcs_data(args.csv)
    model, config = load_model(args.model)

    fit_result = fit_model(tau, G_exp, model, config)
    best_params = fit_result.x

    kinetic_sol = solve_kinetics(
        model.kinetic_odes, best_params, tau, config.y0
    )

    G_fit = model.predict_fcs(tau, kinetic_sol, best_params)
    chi2_red = reduced_chi_squared(G_exp, G_fit, len(best_params))

    print("\nBest-fit parameters:")
    for name, val in zip(config.param_names, best_params):
        print(f"  {name}: {val:.6g}")

    print(f"\nReduced chi^2: {chi2_red:.4f}")

    save_predicted_curve(tau, G_fit)

if __name__ == "__main__":
    main()
