# Generative Model Utils

This repo contains all the utils i use on Synthetic Data Generation Projects. The folder structe is as follows:

- `ml_utility`: Tools for assesing the machine learning utility of generative models
- `statistics`: Tools for assesing the statistical similarity of generative models
- `preprocessing`: Preprocessing tools (Scaling and Categorical Encoding and Outlier Detection)
- `postprocessing`: Postprocessing tools for filtering generated synthetic data
- `synthesizer`: This folder contains the code for the synthetic data generating generative models.

# TODO

- Change the CTABGAN saving mechanism similar to TTVAE.

    > The TTVAE saving mechanism is changed (look at [TTVAE](https://github.com/coksvictoria/TTVAE)). This makes is possible to change the folder structure without breaking the saving mechanism. The break occurs due to fact that pytorch uses pickle module for saving things and pickle needs to see the same folder structure to load the saved object instance. I changed the structure of the class such that there is no import dependency in this repo.
 
# Running Test Scripts

```BASH
export PYTHONPATH="$PYTHONPATH:$(pwd)"; python test/synthesizer_test.py 
```
