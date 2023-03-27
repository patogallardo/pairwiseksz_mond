'''
Opens data and covs from C21, open correlation function from Ross and integrate it.

Then fit.
'''

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from sys import argv
from scipy.stats import chi2

first_bin, last_bin = 3, 17
show = True
h = 0.6736 # planck PR3 2018

assert len(argv) == 4
zbin, bincent, obs_name = int(argv[1]), int(argv[2]), argv[3] # usage file.py 1 1 L61_150

def xi_r_sq(r, xi_interp):
    r_sq = r**2
    return (xi_interp(r)) * r_sq


def chisq(amplitude, r, p, inv_cov, f):
    '''Receives an amplitude, r, p, invcov and a function to
    fit, returns the chisq for these values. Use this with an optimizer
    to get the best fit parameter for the amplitude.'''
    delta = (p - f(r) * amplitude)
    chisq_ret = np.dot(np.dot(delta.T, inv_cov), delta)
    return chisq_ret


def get_max_lik_fit(r, p, inv_cov, f):
    '''Received observed r, p inv_cov and a function to fit, fits the amplitude
    and returns it.'''
    opt = optimize.minimize(chisq, (1/15.0), (r, p, inv_cov, f))
    amp = opt.x[0]
    return amp

corr_fun_fname = "Ross_2016_COMBINEDDR12_zbin%i_correlation_function_monopole_post_recon_bincent%i.dat" % (zbin, bincent)
corr_fun_dir = "Ross_2016_COMBINEDDR12"

obsdir_150 = './DR6_res/'
obsdir_090 = './DR6_res/'
obs_fnames = {"L61_150": obsdir_150 + "DR6_150GHz_C21cat_lum_gt_6p1e10.hdf",
              "L43_150": obsdir_150 + "DR6_150GHz_C21cat_lum_gt_4p3e10.hdf",
              "L61_090": obsdir_090 + "DR6_090GHz_C21cat_lum_gt_6p1e10.hdf",
              "L43_090": obsdir_090 + "DR6_090GHz_C21cat_lum_gt_4p3e10.hdf"}


obs_fname = obs_fnames[obs_name]

print("Using:\ncorr_fun:%s\nobs:%s\ncov:%s" % (corr_fun_fname, obs_fname, obs_fname))

# open data

df_curve = pd.read_hdf(obs_fname, 'df_ksz_err')[first_bin:last_bin]
rsep = df_curve.r_mp.values
p_pw = -1.0 * df_curve.ksz_curve.values
df_cov = pd.read_hdf(obs_fname, 'df_cov')
C_pw = df_cov.values[:, 1:].astype(float)[first_bin:last_bin, first_bin:last_bin]


df_corr_fun = pd.read_csv(corr_fun_dir + '/' + corr_fun_fname, delim_whitespace=True, skiprows=3,
                          names=["R_ov_h", "xi", "err_xi"])
df_corr_fun["R_Mpc"] = df_corr_fun.R_ov_h/h
cov = df_cov.values.astype(float)[first_bin:last_bin, first_bin:last_bin]
inv_cov = np.linalg.inv(cov)

# plot corr fun
plt.figure(figsize=[8, 4.5])
plt.scatter(df_corr_fun.R_ov_h, df_corr_fun.xi * df_corr_fun.R_ov_h ** 2)
plt.title("%s" % corr_fun_fname, fontsize=8)
plt.xlabel("$h^{-1}Mpc$")
plt.ylabel("$\\xi$")
if show:
    plt.show()
else:
    plt.savefig('./plots/corr_fun_zbin_%i_bincent_%i.pdf' % (zbin, bincent))
    plt.close()
# end plot corr fun


# compute integrals
xi_interp = interp1d(df_corr_fun.R_Mpc,
                     df_corr_fun.xi,
                     kind='quadratic')
r = np.linspace(6, df_corr_fun.R_Mpc.values[-1] - 1, 50)
I = np.zeros(len(r))

for j in range(len(r)):
    I_j = quad(xi_r_sq, 6, r[j], args=(xi_interp))[0]
    I[j] = I_j
# end compute integrals

# interpolate g
g = interp1d(r, I/r**2, kind='cubic')
sqrt_g = interp1d(r, np.sqrt(I)/r, kind='cubic')
# end interpolation

measured_curve = -df_curve.ksz_curve.values
rsep = df_curve.r_mp
errorbars = df_curve.errorbar.values


# fit and pte
fitted_amplitude_g = get_max_lik_fit(rsep, measured_curve, inv_cov, g)
fitted_amplitude_sqrt_g = get_max_lik_fit(rsep, measured_curve, inv_cov, sqrt_g)
rv = chi2(df=len(rsep))
chisq_g = chisq(fitted_amplitude_g, rsep, measured_curve, inv_cov, g)
chisq_sqrt_g = chisq(fitted_amplitude_sqrt_g, rsep, measured_curve, inv_cov, sqrt_g)
pte_g = 1 - rv.cdf(chisq_g)
pte_sqrt_g = 1 - rv.cdf(chisq_sqrt_g)
# end fit and pte

plt.figure(figsize=[8, 4.5])
plt.plot(r, fitted_amplitude_g * g(r), label='$g$, $\\chi^2=%1.1f$, pte=%1.2f' % (chisq_g, pte_g))
plt.errorbar(rsep, measured_curve, yerr=errorbars,
             marker='o', ls='')
plt.plot(r, fitted_amplitude_sqrt_g * sqrt_g(r), label='$\\sqrt{g}$, $\\chi^2=%1.1f$ pte=%1.2f' % (chisq_sqrt_g,
                                                                                   pte_sqrt_g))

plt.text(100, 3.5, "$g=\\frac{1}{r^2}\\int \\xi(r) r^2 dr}$")

plt.legend()
plt.xlabel("r[Mpc]")
plt.ylabel("")
plt.axhline(0, color='black')

plt.figtext(0.5, 0.98,
            "%s\n%s" % (obs_name, corr_fun_fname),
            fontsize='small',
            va='top', ha='center' )
if show:
    plt.show()
else:
    plt.savefig("plots/nocov_adj_zbin_%i_bincent_%i_%s.pdf" % (zbin, bincent, obs_name))
    plt.close()
