'''
Opens data and covs from C21, open correlation function from Ross and integrate it.

Then fit.
'''
import matplotlib
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy import optimize
# plt.rcParams.update({'font.size': 13})

# plotting stuff
import seaborn as sns
# set fig params
sns.set_context("paper")
sns.set_style('ticks')
sns.set_palette('colorblind')
figparams = {
    'text.latex.preamble': r'\usepackage{amsmath} \boldmath',
    'text.usetex': True,
    'axes.labelsize': 16.,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': [10., 8.],
    'font.family': 'DejaVu Sans',
    'legend.fontsize': 18}
plt.rcParams.update(figparams)
cs = plt.rcParams['axes.prop_cycle'].by_key()['color']

show = False

first_bin, last_bin = 2, 17
show = False
h = 0.6736  # planck PR3 2018

zbin = 2
bincent = 0
obs_name = "L43_150"


def xi_r_sq(r, xi_interp):
    r_sq = r**2
    return (xi_interp(r)) * r_sq

# chisq(amp, exp, rsep, df_curve.ksz_curve.values,
    # amp_exponent_function_tofit, C_pw)


def get_max_lik_fit(r, p, cov, amp_exponent_function_tofit):
    '''Received observed r, p inv_cov and a function to fit, fits the amplitude
    and returns it.'''
    x0 = np.array([1/15, 0.8])
    opt = optimize.minimize(chisq, x0,
                            args=(r, p, amp_exponent_function_tofit, C_pw, cov))
    amp, exp = opt.x[0], opt.x[1]
    res = {'amp': amp,
           'exp': exp}
    print(res)
    return res


corr_fun_fname = "Ross_2016_COMBINEDDR12_zbin%i_correlation_function_monopole_post_recon_bincent%i.dat" % (
    zbin, bincent)
corr_fun_dir = "Ross_2016_COMBINEDDR12"
obsdir_150 = './DR6_res/'
obsdir_090 = './DR6_res/'
obs_fnames = {"L61_150": obsdir_150 + "DR6_150GHz_C21cat_lum_gt_6p1e10.hdf",
              "L43_150": obsdir_150 + "DR6_150GHz_C21cat_lum_gt_4p3e10.hdf",
              "L61_090": obsdir_090 + "DR6_090GHz_C21cat_lum_gt_6p1e10.hdf",
              "L43_090": obsdir_090 + "DR6_090GHz_C21cat_lum_gt_4p3e10.hdf",
              "L43_150_REST_Z": obsdir_150 + "DR6_150GHz_C21cat_lum_gt_4p3e10_and_zgt0p44_and_zlt0p66.hdf"}
obs_fname = obs_fnames[obs_name]


# open data
df_corr_fun = pd.read_csv(corr_fun_dir + '/' + corr_fun_fname, delim_whitespace=True, skiprows=3,
                          names=["R_ov_h", "xi", "err_xi"])
df_corr_fun["R_Mpc"] = df_corr_fun.R_ov_h/h
df_curve = pd.read_hdf(obs_fname, 'df_ksz_err')[first_bin:last_bin]

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
pow_g = interp1d(r, np.power(I/r**2, 0.5),
                 kind='cubic',
                 bounds_error=False,
                 fill_value='extrapolate')
# end interpolation

rsep = df_curve.r_mp.values


def amp_exponent_function_tofit(amplitude, exponent,
                                rsep, I):
    f_pow_g = interp1d(r, np.power(I/r**2, exponent),
                       kind='cubic',
                       bounds_error=False,
                       fill_value='extrapolate')
    g_exp_power_times_A = -1.0 * amplitude * f_pow_g(rsep)
    return g_exp_power_times_A


# start likelihood exploration
df_cov = pd.read_hdf(obs_fname, 'df_cov')
C_pw = df_cov.values.astype(float)[first_bin:last_bin, first_bin:last_bin]


def chisq(amp_exp,
          rsep, p_pw, amp_exponent_function_tofit,
          C_pw, C_sdss=None):
    '''Receives an amplitude and exponent,
    r, p_pw, p_sdss, C_pw, C_sdss and fits 
    an amplitude.
    First implementation C_sdss=None, only use the pairwise cov matrix
    '''
    amplitude, exponent = amp_exp[0], amp_exp[1]
    p_predicted = amp_exponent_function_tofit(amplitude, exponent, rsep, I)
    delta = (p_pw - p_predicted)
    #C = combine_covs(amplitude, C_pw, C_sdss)
    C = C_pw
    C_inv = np.linalg.inv(C)
    chisq_toreturn = np.dot(np.dot(delta.T, C_inv), delta)
    return chisq_toreturn


# Plot contours
N_samples = 200
amp_range = [0.0, 0.04]
exp_range = [0.3, 2.00]

amps = np.linspace(amp_range[0], amp_range[1], N_samples)
exps = np.linspace(exp_range[0], exp_range[1], N_samples)

amp_mat, exp_mat = np.meshgrid(amps, exps)
chisq_mat = np.zeros_like(amp_mat)

res = get_max_lik_fit(rsep, df_curve.ksz_curve.values,
                      C_pw, amp_exponent_function_tofit)

for j in range(len(amps)):
    for k in range(len(exp_mat)):
        amp, exp = amp_mat[j, k], exp_mat[j, k]
        amp_exp = [amp, exp]
        chisq_mat[j, k] = chisq(amp_exp, rsep, df_curve.ksz_curve.values,
                                amp_exponent_function_tofit, C_pw)
Likelihood = np.exp(-chisq_mat/2)
Likelihood = Likelihood/Likelihood.max()


plt.subplots(figsize=[4, 4],
             constrained_layout=True)
levels = np.exp(-np.arange(3, -1, -1)**2/2)
plt.contour(amp_mat, exp_mat, Likelihood, levels=levels,
            colors='black')
plt.xlabel(r'$\mathrm{Amplitude}$')
plt.ylabel(r'$\mathrm{Force~Law~Index,~}n$')
plt.scatter(res['amp'], res['exp'], label='ML', color='black', marker='X')
plt.axhline(1, color=cs[0], alpha=0.5, ls='dashed')
plt.text(0.032, 1.01, r'$\Lambda\mathrm{CDM}$',
         color=cs[0])
plt.axhline(0.5, color=cs[1], alpha=0.5, ls='dashed')
plt.text(0.032, 0.51, r'$\mathrm{MOND}$',
         color=cs[1])
#plt.savefig('plots/contour_plot.pdf')
# End plot contours
plt.show()

# plot curves
plt.subplots(figsize=[5, 3], constrained_layout=True)
plt.errorbar(df_curve.r_mp.values,
             df_curve.ksz_curve.values,
             yerr=df_curve.errorbar.values,
             ls='', marker='o',
             capsize=2)
plt.axhline(0, color='black', lw=2)

labels = ['n=0.5 (fixit?)', 'n=1 (NG)', 'n=%1.2f (ML)' % res['exp']]
for label, amp, exp in zip(labels, [.02, .02, res['amp']], [1, 0.5, res['exp']]):
    plt.plot(rsep, amp_exponent_function_tofit(amp, exp, rsep, I),
             label=label)
plt.legend(loc='lower right')
plt.show()
# plt.savefig('plots/2_param_pksz.pdf')
# end plot curves

L_n = Likelihood.sum(axis=0)
L_n = L_n/L_n.max()
th = np.exp(-1/2.)
plt.plot(exps, L_n)
plt.axvline(res['exp'])
plt.axhline(th)

cumsum = np.cumsum(L_n)
cumsum = cumsum/cumsum.max()

lowerlimit = 0.5-0.68/2
upperlimit = 0.5+0.68/2
sel = np.logical_and(cumsum>lowerlimit, cumsum<upperlimit)
lower_exp = exps[sel][0]
upper_exp = exps[sel][-1]
sigma_n = 0.5 * (upper_exp - lower_exp)
print(sigma_n)

plt.axvspan(xmin=lower_exp, xmax=upper_exp, alpha=0.1)
plt.show()
