import sys
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bumps.fitproblem import load_problem
from bumps.cli import load_best
from bumps.dream.state import load_state
from bumps.errplot import calc_errors_from_state
from refl1d.errors import align_profiles
from refl1d.errors import show_errors
from refl1d.errors import show_profiles

model, store = sys.argv[1:3]

problem = load_problem(model)
load_best(problem, os.path.join(store, model[:-3]+".par"))

print("chisq %s"%problem.chisq_str())

Rth = list(problem.models)[0].fitness.theory()
Qth = list(problem.models)[0].fitness._data.x
fname = os.path.join(os.getcwd(), "Fit_store_modesim/IQ_GMO_dd.txt")
arraytosave = np.column_stack((Qth, Rth))
np.savetxt(fname, arraytosave)

Rth2 = list(problem.models)[1].fitness.theory()
Qth2 = list(problem.models)[1].fitness._data.x
fname2 = os.path.join(os.getcwd(), "Fit_store_modesim/IQ_medsim_GMO_H2O_dd.txt")
arraytosave2 = np.column_stack((Qth2, Rth2))
np.savetxt(fname2, arraytosave2)

Rth3 = list(problem.models)[2].fitness.theory()
Qth3 = list(problem.models)[2].fitness._data.x
fname3 = os.path.join(os.getcwd(), "Fit_store_modesim/IQ_medsim_GMO_D2O_dd.txt")
arraytosave3 = np.column_stack((Qth3, Rth3))
np.savetxt(fname3, arraytosave3)

raise Exception()