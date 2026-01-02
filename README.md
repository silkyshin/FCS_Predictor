# FCS_predictor
This repo contains scripts to solve ODEs of oligomerization pathways and predict what their FCS values G0 and τ would be, or just the autocorrelation curves at different timepoints. 

Read the README in the ODE directory.  
To predict with different oligomerization models, copy the format of the model scripts.

## Installation
```
git clone https://github.com/silkyshin/FCS_Predictor.git
```

## Usage
```
python ODEsolver.py --model 'path/to/model.py'
python FCS_predictor.py --csv 'path/to/csv' --sfactor 'scaling factor'
python autocorr_predictor.py --csv 'path/to/csv' --sfactor 'scaling factor'
```

### Hydrodynamic radius, therefore diffusion time, scales as such:
Rh_n=Rh_1*n^v  
Rh_n - hydrodynamic radius of n-mer  
Rh_1 - hydrodynamic radius of monomer  
n - number of monomers in n-mer  
v - scaling factor  

### Typical scaling factor values
Folded proteins - 0.3-0.35  
Molten globules - 0.38-0.45  
Compact IDPs - 0.4-0.45  
Theta conditions - 0.5  
Expanded IDPs (expanded chain, unrealistic) - 0.55-0.6
Amyloid fibrils - 0.9-1  
