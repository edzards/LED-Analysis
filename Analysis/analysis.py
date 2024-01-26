# -*- coding: utf-8 -*-
#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from lmfit import Model
from lmfit import minimize, Parameters, Parameter, report_fit

def main():

    y = np.genfromtxt('./Data/CH1@DT5725SB_13964_EspectrumR_run01_20230209_121557.txt', unpack=True)
    x = np.arange(len(y))

    x_masked = x[(x>4960) & (x<5300)]
    y_masked = y[(x>4960) & (x<5300)]

    # define our model = Gaussian
    fmodel_cal = Model(Gaussian)
    
    # set start params of fit
    params_cal = fmodel_cal.make_params()
    params_cal['amp'].set(value=1400, min=0.)
    params_cal['mean'].set(value=5100)
    params_cal['sigma'].set(value=200, min=0)

    # do the fit
    fres_cal = fmodel_cal.fit(y_masked, params_cal, x=x_masked, weights=None)
    print(fres_cal.fit_report())

    plt.figure()
    plt.plot(x, y, '-', color='tab:blue')
    plt.plot(x_masked, y_masked, '-', color='tab:orange')
    plt.plot(x_masked, fres_cal.best_fit, '-', color='tab:red')
    plt.yscale('log')
    plt.xlabel('Energy (lsb)')
    plt.ylabel('Counts/xx lsb')
    plt.ylim(bottom=0.5)
    plt.tight_layout()
    plt.show()    

 
######################################################################
# Linear function
def linear_func(x, a, b):
    func = a*x + b
    return func

######################################################################
def Gaussian(x, amp, mean, sigma):
    func = amp * np.exp(-(x-mean)**2 / (2*sigma**2))
    return func


######################################################################
if __name__ == "__main__":
    main()	
