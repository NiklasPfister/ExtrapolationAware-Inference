# Extrapolation-Aware Nonparametric Statistical Inference

This repository contains Python code for the Xtrapolation procedure
introduced in <a href="#ref1">[1]</a><a id="ref1-back"></a>. It allows
to perform extrapolation-aware nonparametric statistical inference
based on any existing nonparametric estimate.

Additionally the repository contains code to reproduce all numerical
experiments presented in <a href="#ref1">[1]</a><a
id="ref1-back"></a>.


## Applying Xtrapolation

All code required for Xtrapolation is provided in the folder
'./Xtrapolation'.  The code is based on the Python package
[adaXT](https://github.com/NiklasPfister/adaXT) which implements a
fast and extendable tree algorithm and is used to compute random
forest based weights.

To install the requirements use:
```bash
pip install -r requirements.txt

```
To reproduce the real-data experiments [pytorch](https://pytorch.org/)
needs to be installed as well as it is used to train the quantile
neural networks.

The following code snippet provides a minimal working example how to
use Xtrapolation to estimate extrapolation bounds.

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xtrapolation.xtrapolation import Xtrapolation
import experiments.helpers.examples as ex
from matplotlib import pyplot as plt

# Generate data
n = 200
X = np.linspace(-2.5, 2.5, n).reshape(n, 1)
Y = ex.expit(X) + np.random.normal(0, 0.1, n)
xeval = np.linspace(-3.5, 3.5, 50).reshape(-1, 1)

# Fit random forest
rf = RandomForestRegressor(n_estimators=200)
rf.fit(X, Y)
Yhat = rf.predict(X).reshape(-1, 1)
mrss_rf = np.mean((Y - Yhat)**2)

# Apply Xtrapolation
deriv_params = {'rf_pars': {'min_samples_leaf': 10}}
xtra_rf = Xtrapolation(orders=[1], deriv_params=deriv_params, verbose=1)
extra_bounds = xtra_rf.prediction_bounds(X, Yhat, x0=xeval)

# Plot results
plt.scatter(X, Y, color='black', alpha=0.3)
plt.plot(X, Yhat, label='rf predictions')
plt.plot(xeval, extra_bounds[:, 0, 0], label='lower extrapolation bound')
plt.plot(xeval, extra_bounds[:, 0, 1], label='upper extrapolation bound')
plt.legend()
plt.show()
```

## Reproducing numerical experiments

All code to reproduce the experiments is provided in folder
'./experiments'. In order to reproduce the biomass data experiment,
please contact the [owners of the
data](https://doi.org/10.1016/j.foreco.2022.120653) to receive a
copy. Data for all other experiments is downloaded or constructed
within the provided code.

The code is structured as follows.
```
experiments
├── prediction_intervals
│   ├── biomass_example.py		    # Run biomass experiment
│   ├── analyze_results_biomass.py	    # Generate plots after running biomass experiment
│   ├── abalone_example.py		    # Run abalone experiment
│   └── analyze_results_abalone.py          # Generate plots after running abalone experiment
│
├── simulation_experiment
│   ├── simulation_experiment.py	    # Run single seed of simulation experiment
│   └── analyze_simulation_experiments.py   # Generate plots after running all seeds/settings
│
├── visualization
│   ├── linear_vs_xtrapolation.py	    # Reproduce Figure 2
│   └── visualizing_extrapolation_bounds.py # Reproduce Figure 1
│
├── helpers
│   ├── examples.py			    # Example functions used in code
│   ├── regression_methods.py 		    # Sklearn regression piplines
│   ├── cp_methods.py 		    	    # Functions used to fit quantile regressions
│   └── conformal 		    	    # Folder with external code for conformalized methods
│
└── results				    # Folder in which results are saved
```


## License

This project is licensed under the MIT License - see the
[LICENSE](LICENSE) file for details.


## References

<a id="ref1"></a><a href="#ref1-back">[1]</a>: Niklas Pfister and
Peter Bühlmann. 2024. "Extrapolation-Aware Nonparametric Statistical
Inference." Preprint [ArXiv](https://arxiv.org/abs/2402.09758).


If you use this software in your work, please cite it using the
following metadata.

```bibtex
@article{pfister2024xtra,
  title={Extrapolation-Aware Nonparametric Statistical Inference}, 
  author={Niklas Pfister and Peter B\"uhlmann},
  year={2024},
  journal={arXiv preprint arXiv:2402.09758}
}
```