import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def gauss(x, A, mean, sig):
    '''
    Returns a Gaussian Profile
    '''
    gaussian = (A/(sig*np.sqrt(2*np.pi)))*np.exp((-pow((x-mean), 2.0)/2*pow(sig, 2.0)))
    return gaussian


def nonlinear_func_real(x_vals, amp1, amp2, m1, m2, w1, cont):
    '''
    Combines two gaussians into one simulated model spectrum
    '''
    return gauss(x_vals, amp1, m1, w1) + gauss(x_vals, amp2, m2, w1) + cont


def fit_data(wave, spec, amp1, amp2, m1, m2, w1, cont):
    '''
    Fit the two Gaussian Profiles to a spectrum to 
    determine the amplitdues, line centers, widths, and 
    continuum levels of the input spectra
    '''
    print("========================================")
    print('Performing a least square fit...')
    popt, pcov = curve_fit(nonlinear_func_real, wave, spec, p0=[amp1, amp2, m1, m2, w1, cont])
    errs = np.sqrt(np.diag(pcov))
    print('=========================================')
    print("Printing out the best fitting parameters and 1 std errors...")
    print('-----------------------------------------')
    print('Line 1 Amp: ' + str(popt[0]) + ' Amp_error: ' + str(errs[0]) + '\n')
    print('Line 2 Amp: ' + str(popt[1]) + ' Amp_error: ' + str(errs[1]) + '\n')
    print('Line 1 Wavelength: ' + str(popt[2]) + ' Amp_error: ' + str(errs[2]) + '\n')
    print('Line 2 Wavelength: ' + str(popt[3]) + ' Amp_error: ' + str(errs[3]) + '\n')
    print('Line width: ' + str(popt[4]) + ' Sigma_err: ' + str(errs[4]) + '\n')
    print('Continuum level: ' + str(popt[5]) + ' Amp_error: ' + str(errs[5]) + '\n')
    
    print("=========================================")
    print('Plotting the observered and modeled data...')
    y_model = nonlinear_func_real(wave, *popt)
    plt.plot(wave, spec, label = 'data')
    plt.plot(wave, y_model, label = 'model')
    plt.vlines(popt[2], ymin = 0.0, ymax=spec.max(), linestyle = '--', alpha = 0.75, linewidth = 1.0)
    plt.vlines(popt[3], ymin = 0.0, ymax=spec.max(), linestyle = '--', alpha = 0.75, linewidth = 1.0)
    plt.legend()
    plt.show()
    return popt, pcov



def get_redshift(lam0, lame):
    '''
    return to redshift of a source 
    given the observed and rest frame wavelengths
    '''
    return (lam0-lame)/lame
