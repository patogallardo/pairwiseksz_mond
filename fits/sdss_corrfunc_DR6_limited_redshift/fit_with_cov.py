import numpy as np
import pandas as pd
from sys import argv
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chi2

show = True
sdss_pairwise_curves_fname = "covariances/sdss_g_sqrtg.csv"
sdss_cov_g_fname = "covariances/covariances_sdss_g.txt"
sdss_cov_sqrtg_fname = "covariances/covariances_sdss_sqrt_g.txt"

print("Usage example: .py L61_150")
obs_name = argv[1]
print("Using %s" % obs_name)
first_bin, last_bin = 2, 17

df_sdss_pw_curves = pd.read_csv(sdss_pairwise_curves_fname)
p_sdss_g = df_sdss_pw_curves.g.values
p_sdss_sqrt_g = df_sdss_pw_curves.sqrt_g.values
C_g = np.loadtxt(sdss_cov_g_fname)
C_sqrt_g = np.loadtxt(sdss_cov_sqrtg_fname)

obsdir_150 = './DR6_res/'
obsdir_090 = './DR6_res/'
obs_fnames = {"L61_150": obsdir_150 + "DR6_150GHz_C21cat_lum_gt_6p1e10.hdf",
              "L43_150": obsdir_150 + "DR6_150GHz_C21cat_lum_gt_4p3e10.hdf",
              "L61_090": obsdir_090 + "DR6_090GHz_C21cat_lum_gt_6p1e10.hdf",
              "L43_090": obsdir_090 + "DR6_090GHz_C21cat_lum_gt_4p3e10.hdf",
              "L43_150_REST_Z": obsdir_150 + "DR6_150GHz_C21cat_lum_gt_4p3e10_and_zgt0p44_and_zlt0p66.hdf"}
obs_fname = obs_fnames[obs_name]

df_pw_obs_curve = pd.read_hdf(obs_fname, 'df_ksz_err')[first_bin:last_bin]
rsep = df_pw_obs_curve.r_mp.values
p_pw = df_pw_obs_curve.ksz_curve.values
df_cov = pd.read_hdf(obs_fname, 'df_cov')
C_pw = df_cov.values.astype(float)[first_bin:last_bin, first_bin:last_bin]

def combine_covs(scale, C_pw, C_sdss):
    C = C_pw + scale**2 * C_sdss
    return C


def chisq(amplitude, r, p_pw, p_sdss, C_pw, C_sdss):
    '''Receives an amplitude r, p_pw, p_sdss, C_pw, C_sdss and fits 
    an amplitude.
    '''
    delta = (p_pw - p_sdss * amplitude)
    C = combine_covs(amplitude, C_pw, C_sdss)
    C_inv = np.linalg.inv(C)
    chisq_toreturn = np.dot(np.dot(delta.T, C_inv), delta)
    return chisq_toreturn


def get_max_lik_fit(r, p_pw, p_sdss, C_pw, C_sdss):
    opt = minimize(chisq, (-3/5), (r, p_pw, p_sdss, C_pw, C_sdss))
    amp = opt.x[0]
    return amp


def sample_likelyhood(Nsamples, best_amplitude, max_amp_factor,
                      rsep, p_pw, p_sdss_g_or_sqrt_g, C_pw,
                      C_g_or_sqrtg):
    amplitude = np.linspace(0, best_amplitude * max_amp_factor, Nsamples)
    likelihood = np.zeros_like(amplitude)
    for j in range(len(amplitude)):
        likelihood[j] = np.exp(-0.5 * chisq(amplitude[j], rsep, p_pw,
                               p_sdss_g_or_sqrt_g, C_pw, C_g_or_sqrtg))
    return amplitude, likelihood/np.max(likelihood)

def get_one_sigma(Nsamples_res, best_amplitude, max_amp_factor,
                  rsep, p_pw, p_sdss_g_or_sqrt_g, C_pw,
                  C_g_or_sqrtg):
    amp_sweep, L = sample_likelyhood(Nsamples_res, best_amplitude, max_amp_factor,
                                     rsep, p_pw, p_sdss_g_or_sqrt_g, C_pw,
                                     C_g_or_sqrtg)
    sel = L > np.exp(-0.5)
    sigma = (amp_sweep[sel].max() - amp_sweep[sel].min())/2.0
    return sigma


fitted_g = get_max_lik_fit(rsep, p_pw, p_sdss_g, C_pw, C_g)
fitted_sqrt = get_max_lik_fit(rsep, p_pw, p_sdss_sqrt_g, C_pw, C_sqrt_g)

fitted_g_chisq = chisq(fitted_g, rsep, 
                       p_pw, p_sdss_g, C_pw, C_g)
fitted_sqrt_g = chisq(fitted_sqrt, rsep,
                           p_pw, p_sdss_sqrt_g, C_pw, C_sqrt_g)

sigma_g = get_one_sigma(10000, fitted_g, 5, rsep, p_pw, p_sdss_g, C_pw, C_g)
sigma_sqrt_g = get_one_sigma(10000, fitted_sqrt, 5, rsep, p_pw,
                             p_sdss_sqrt_g, C_pw, C_sqrt_g)

plt.subplots(constrained_layout=True, figsize=[5, 3])
#plt.figure(figsize=[5,3])
plt.rcParams.update({'font.size':12})

plt.errorbar(rsep, p_pw,
             yerr=np.sqrt(np.diag(C_pw)),
             marker='o', ls='', color='C0',
             label='pairwise SZ')
#plt.scatter(rsep, p_sdss_g * fitted_g,
#            marker='o', color='C1',
#            label='sdss $g$')
#plt.scatter(rsep, p_sdss_sqrt_g * fitted_sqrt, 
#            marker='o', color='C2',
#            label='sdss $\\sqrt{g}$')
dof = len(rsep)
rv = chi2(dof)
pte_g = 1 - rv.cdf(fitted_g_chisq)
pte_sqrt_g = 1 - rv.cdf(fitted_sqrt_g)

plt.fill_between(rsep,
                 y1 = p_sdss_g * (fitted_g + sigma_g),
                 y2 = p_sdss_g * (fitted_g - sigma_g), 
                 color='C1', alpha=0.2,
                 label='$g$, $\\chi^2=$%1.2f, PTE=%1.2f' % (fitted_g_chisq,
                                                            pte_g))
plt.plot(rsep, p_sdss_g * fitted_g, color='C1')

plt.fill_between(rsep,
                 y1 = p_sdss_sqrt_g * (fitted_sqrt + sigma_sqrt_g),
                 y2 = p_sdss_sqrt_g * (fitted_sqrt - sigma_sqrt_g),
                 color='C2', alpha=0.2,
                 label='$\\sqrt{g}$ (MOND), $\\chi^2$=%1.2f, PTE=%1.2f' % (fitted_sqrt_g,
                                                                pte_sqrt_g))
plt.plot(rsep, p_sdss_sqrt_g * fitted_sqrt,
         color='C2')
plt.legend(fontsize=8)
plt.axhline(0, color='black')
plt.xlabel('r [Mpc]')
plt.ylabel('$\\hat p_{kSZ}$ [$\\rm {\\mu K}$]')
if show:
    plt.show()
else:
    plt.savefig('%s_uncertainty_bands.pdf' % obs_name)
    plt.close()
