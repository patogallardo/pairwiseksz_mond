'''
Opens data and covs from C21, open correlation function from Ross and integrate it.

Then fit.
'''

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.stats import chi2

first_bin, last_bin = 2, 17
show = False
h = 0.6736 # planck PR3 2018

zbin = 2
bincent = 0
obs_name = "L43_150"

def xi_r_sq(r, xi_interp):
    r_sq = r**2
    return (xi_interp(r)) * r_sq


def get_max_lik_fit(r, p, inv_cov, f):
    '''Received observed r, p inv_cov and a function to fit, fits the amplitude
    and returns it.'''
    opt = optimize.minimize(chisq, (1/15.0), (r, p, inv_cov, f))
    amp = opt.x[0]
    return amp


corr_fun_fname = "Ross_2016_COMBINEDDR12_zbin%i_correlation_function_monopole_post_recon_bincent%i.dat" % (zbin, bincent)
corr_fun_dir = "Ross_2016_COMBINEDDR12"
obsdir_150 = './C21_data/'
obsdir_090 = './C21_data/'
obs_fnames = {"L61_150": obsdir_150 + "S18_coadd_150GHz_V20DR15_V3_lum_gt_06p1_bs_dt_ksz_curve_and_errorbars.csv",
              "L43_150": obsdir_150 + "S18_coadd_150GHz_V20DR15_V3_lum_gt_04p3_bs_dt_ksz_curve_and_errorbars.csv",
              "L61_090": obsdir_090 + "S18_coadd_090GHz_V20DR15_V3_lum_gt_06p1_bs_dt_ksz_curve_and_errorbars.csv",
              "L43_090": obsdir_090 + "S18_coadd_090GHz_V20DR15_V3_lum_gt_04p3_bs_dt_ksz_curve_and_errorbars.csv"}
obs_fname = obs_fnames[obs_name]

# open data
df_corr_fun = pd.read_csv(corr_fun_dir + '/' + corr_fun_fname, delim_whitespace=True, skiprows=3,
                          names=["R_ov_h", "xi", "err_xi"])
df_corr_fun["R_Mpc"] = df_corr_fun.R_ov_h/h

rsep = np.array([  5. ,  15. ,  25. ,  35. ,  45. ,  55. ,  65. ,  75. ,  85. , 95. , 105. , 115. , 125. , 135. , 145. , 175. , 225. , 282.5, 355. ])[first_bin:last_bin]

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
g = interp1d(r, 1/(1+xi_interp(r)) * I/r**2, kind='cubic',
             bounds_error=False,
             fill_value='extrapolate' ) # do it up to 17th bin
sqrt_g = interp1d(r, np.sqrt(g(r)), kind='cubic',
                  bounds_error=False,
                  fill_value='extrapolate')
# end interpolation


# make figure

plt.plot(r, g(r), label='g', color='C0')
plt.plot(r, 2 * sqrt_g(r), label='2 sqrt(g)', color='C2')

plt.scatter(rsep, g(rsep), color='C0')
plt.scatter(rsep, 2 * sqrt_g(rsep), color='C2')

plt.legend()
if show:
    plt.show()
else:
    plt.savefig('plots/diagnosticplot_sdss_ksz_prediction.pdf')
    plt.close()

df_out = pd.DataFrame({'rsep': rsep, 
                       'g': g(rsep),
                       'sqrt_g': sqrt_g(rsep)})
df_out.to_csv('covariances/sdss_g_sqrtg.csv')
