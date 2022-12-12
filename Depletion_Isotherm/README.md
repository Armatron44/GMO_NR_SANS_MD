# Scripts to analyse depletion isotherm data

## Required software
* [Originlab] (https://www.originlab.com/)
* [python 3] (https://www.python.org/downloads/)
* [jupyter] (https://jupyter-notebook.readthedocs.io/en/stable/)
* [uncertainties] (https://pythonhosted.org/uncertainties/)

## Overview

The jupyter notebook "FTIR_calibration+analysis" was used to track the values and uncertainties of the GMO standards and samples used in the isotherms.
Originlab was used to baseline the standard and sample absorbance data collected on the FTIR & integrate the carbonyl absorbance between 1665-1800 cm$^{-1}$.
All fits were conducted in Originlab, with the resulting derived parameter values and covariance matrix used within the jupyter notebook for further analysis.
The fits have been exported as .svg files so they can be viewed within the notebook.
