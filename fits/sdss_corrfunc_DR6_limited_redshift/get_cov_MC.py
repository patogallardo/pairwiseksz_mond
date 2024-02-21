import matplotlib
#matplotlib.use('Agg')

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import os
from scipy.interpolate import interp1d
from progressbar import progressbar
import seaborn as sns
#from common import fnames_curves as ksz_meas_fnames

np.random.seed(0)

first_bin, last_bin = 2, 17
show = False
NREALIZATIONS = 1000
h = 0.6736

zbin = 2
bincent = 0

def xi_r_sq(r, xi_interp):
    r_sq = r**2
    return (xi_interp(r)) * r_sq

corr_fun_fname = "Ross_2016_COMBINEDDR12_zbin%i_correlation_function_monopole_post_recon_bincent%i.dat" % (zbin, bincent)
corr_fun_dir = "Ross_2016_COMBINEDDR12"
covs_fname = "Ross_2016_COMBINEDDR12_zbin2_covariance_monoquad_post_recon_bincent0.dat"

df_corr_fun = pd.read_csv(corr_fun_dir + '/' + corr_fun_fname,
                          delim_whitespace=True,
                          skiprows=3,
                          names=["R_ov_h", "xi", "err_xi"])
df_corr_fun["R_Mpc"] = df_corr_fun.R_ov_h/h
df_covs = pd.read_csv(os.path.join(corr_fun_dir, covs_fname),
                      comment='#',
                      delim_whitespace='True',
                      names=range(72))
df_ksz_meas = pd.read_hdf('DR6_res/DR6_150GHz_C21cat_lum_gt_4p3e10_and_zgt0p44_and_zlt0p66.hdf',
                          key='df_ksz_err')[first_bin:last_bin]

xi_mean = df_corr_fun.xi.values
r_mpc_corrfunc = df_corr_fun.R_Mpc.values
cov = df_covs.values[:36, :36]

yerrs = np.sqrt(np.diag(df_covs.values))[:36]

xi_realizations = np.random.multivariate_normal(mean=xi_mean,
                                                cov=cov,
                                                size=NREALIZATIONS)

# plot corr func win 
r_toshow = np.linspace(r_mpc_corrfunc[0], r_mpc_corrfunc[-1], 100)
for j in range(NREALIZATIONS):
    xi_interp = interp1d(r_mpc_corrfunc,
                         xi_realizations[j, :], 
                         kind="quadratic")
    plt.scatter(r_mpc_corrfunc, xi_realizations[j, :] * r_mpc_corrfunc**2,
                alpha=50/NREALIZATIONS, marker='.', color='blue')
    plt.plot(r_toshow, xi_interp(r_toshow) * r_toshow**2,
             alpha=50/NREALIZATIONS, color='blue')

if show:
    plt.show()
else:
    plt.savefig('./plots/corr_func_random_realizations.pdf')
    plt.close()


# now compute the integral
NSAMPLES_INT = 30
integrals = np.zeros([NREALIZATIONS, NSAMPLES_INT])
r_integral = np.linspace(6, r_mpc_corrfunc[-1]-1, NSAMPLES_INT)

for realization in progressbar(range(NREALIZATIONS)):
    xi_interp = interp1d(r_mpc_corrfunc,
                         xi_realizations[realization, :], 
                         kind="quadratic")
    for sample_int in range(NSAMPLES_INT):
        integrals[realization, sample_int] = quad(xi_r_sq, 6, r_integral[sample_int], args=(xi_interp))[0]
prefactor = 1/(1+xi_interp(r_integral))
gs = integrals/r_integral**2 * prefactor
sqrt_gs = np.sqrt(integrals * prefactor)/r_integral
# resample to ksz measurement bins
ksz_meas_rsep = df_ksz_meas.r_mp.values
#ksz_meas_rsep = df_ksz_meas.rsep.values
g_sampled = np.zeros([NREALIZATIONS, len(ksz_meas_rsep)])
sqrt_g_sampled = np.zeros([NREALIZATIONS, len(ksz_meas_rsep)])

for realization in range(NREALIZATIONS):
    g_smooth = interp1d(r_integral, gs[realization, :])
    sqrt_gs_smooth = interp1d(r_integral, sqrt_gs[realization, :])
    g_sampled[realization] = g_smooth(ksz_meas_rsep)
    sqrt_g_sampled[realization] = sqrt_gs_smooth(ksz_meas_rsep)


# plot integrals g and sqrt(g)
for j in range(NREALIZATIONS):
    plt.plot(r_integral, gs[j, :],
             color='blue', alpha=50/NREALIZATIONS)
    plt.plot(r_integral, 2 * sqrt_gs[j, :],
             color='C2', alpha=50/NREALIZATIONS)
    plt.scatter(ksz_meas_rsep, g_sampled[j, :],
                color='blue', alpha=50/NREALIZATIONS, marker='.')
    plt.scatter(ksz_meas_rsep, 2 * sqrt_g_sampled[j, :],
                color='C2', alpha=50/NREALIZATIONS, marker='.')
plt.plot([], [], color='blue', label='$g$')
plt.plot([], [], color='C2', label= "$2 \\times \\sqrt{g}$")
plt.legend()
if show:
    plt.show()
else:
    plt.savefig('./plots/MC_integral.pdf')
    plt.close()
# End plot integrals

# plot covariances
#  save g
plt.figure(figsize=[8, 8])
sns.heatmap(np.corrcoef(g_sampled.T), annot=True)
plt.title('$g$')
if show:
    plt.show()
else:
    plt.savefig('./plots/MC_cov_g.pdf')
    plt.close()
np.savetxt('covariances/covariances_sdss_g.txt', np.cov(g_sampled.T))


# plot covariances and save sqrt(g)
plt.figure(figsize=[8, 8])
sns.heatmap(np.corrcoef(sqrt_g_sampled.T), annot=True)
plt.title("$\\sqrt{g}$")
if show:
    plt.show()
else:
    plt.savefig('./plots/MC_cov_sqrt_g.pdf')
    plt.close()
np.savetxt('covariances/covariances_sdss_sqrt_g.txt', np.cov(sqrt_g_sampled.T))
