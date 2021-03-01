# ToveyTomoTools
A collection of tools for performing reconstructions from X-ray tomography data, such as Total Variation or Total Generalised Variation regularised reconstructions. Supports astra or scikit-image backends.

This small piece of code was originally written for processing tomography data for myself and colleagues. The main reason for making it publically available at this point is to allow for easy distribution and reproducibility of results. I do not claim that it offers anything new or better than the huge number of utilities which have been produced with the same motivations. I am open to criticism (e.g. the reporting of bugs) through this forum, although note that this code is no longer under active development and I may be slow to address any problems which are identified.

## Functionality
The main aim was to wrap the basic functionality of the astra toolbox and scikit-image into a single interface, then combine that with some basic linear algebra tools to easily implement regularised reconstruction methods. Most of the functionality is demonstrated by the jupyter notebook included in the repository which, for example, demonstrates how to ensure the raw data is ordered correctly for processing. The available processing methods are:
- Filtered back-projection, with some choice of filter
- SIRT (only available with the astra backend)
- SART
- Total variation least-squares (TV) reconstruction, giving an approximately piecewise constant density reconstruction
- Second order total variation (TV2) reconstruction, giving an approximately piecewise linear density reconstruction
- Total generalised variation (TGV) reconstruction, giving a piecewise linear density reconstruction. Better at representing jumps than TV2, but more computationally intensive to compute

## Installation
The file itself does not require installation but astra can be painful to install initially. I personally use anaconda where the following commands currently produce a functional python environment.
```
conda create --name new_env -c astra-toolbox/label/dev astra-toolbox=1.9.9
conda activate new_env
conda install numba jupyter scikit-image
```
Note that the first time the module is imported it will write a new file called `_bin.py` containing a few bits of relatively high performance code. If for some reason this fails, try it on a different machine then copy the file back to the first machine.
