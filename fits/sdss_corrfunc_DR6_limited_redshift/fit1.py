'''
Fit two models to data and compute 
uncertainty in amplitude.
Make figure and export PTE.

P. Gallardo, K. Pardo
'''
import matplotlib
import numpy as np
import pandas as pd
from sys import argv
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chi2
from matplotlib.ticker import MultipleLocator

# Plot formatting
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
matplotlib.use('Agg')

# Filenames setup
sdss_pairwise_curves_fname = "covariances/sdss_g_sqrtg.csv"
sdss_cov_g_fname = "covariances/covariances_sdss_g.txt"
sdss_cov_sqrtg_fname = "covariances/covariances_sdss_sqrt_g.txt"

obs_name = "L43_150_REST_Z"
print("Using %s" % obs_name)
first_bin, last_bin = 2, 17

df_sdss_pw_curves = pd.read_csv(sdss_pairwise_curves_fname)
p_sdss_g = df_sdss_pw_curves.g.values
p_sdss_sqrt_g = df_sdss_pw_curves.sqrt_g.values
C_g = np.loadtxt(sdss_cov_g_fname)
C_sqrt_g = np.loadtxt(sdss_cov_sqrtg_fname)

obsdir_150 = './DR6_res/'

obs_fnames = {"L43_150_REST_Z": obsdir_150 + "DR6v4_simple_v1_coadd_150GHz_mond_lum_gt_4p3e10_and_zgt0p44_and_zlt0p66_step_10mpc.hdf"}
obs_fname = obs_fnames[obs_name]

df_pw_obs_curve = pd.read_hdf(obs_fname, 'df_ksz_err')[first_bin:last_bin]
rsep = df_pw_obs_curve.r_mp.values
p_pw = df_pw_obs_curve.ksz_curve.values
df_cov = pd.read_hdf(obs_fname, 'df_cov')
C_pw = df_cov.values.astype(float)[first_bin:last_bin, first_bin:last_bin]

def combine_covs(scale, C_pw, C_sdss):
    '''Trivially combine covariances from observation and SDSS.'''
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
    '''Fit an amplitude'''
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

# compute PTEs
dof = len(rsep)
print("degrees of freedom: %i" % dof)
rv = chi2(dof)
pte_g = 1 - rv.cdf(fitted_g_chisq)
pte_sqrt_g = 1 - rv.cdf(fitted_sqrt_g)
df_table = pd.DataFrame(data=[[fitted_g_chisq, pte_g],
                              [fitted_sqrt_g, pte_sqrt_g]],
                        index=['$\\Lambda \\rm CDM$', 'MOND'],
                        columns=['$\\chi^2$', 'PTE'])
df_table.to_latex('table_ptes.tex',
                  escape=False, 
                  float_format="%1.2f")
# write variables
def convert_variable_to_latex_command(varname, value, decplaces=2):
    s = "\\newcommand{\\%s}{%1.2f}" % (varname, value)
    if decplaces == 1:
        s = "\\newcommand{\\%s}{%1.1f}" % (varname, value)
    if decplaces == 3:
        s = "\\newcommand{\\%s}{%1.3f}" % (varname, value)
    if decplaces == 0:
        s = "\\newcommand{\\%s}{%i}" % (varname, value)
    if decplaces == "sci":
        string_value = "%1.2e}$" % value
        string_value = string_value.replace("e-0", '$\\times 10^{-')
        s = "\\newcommand{\\%s}{%s}" % (varname, string_value)
    return s + "\n"
string_out = ""
string_out += convert_variable_to_latex_command("DOF", dof, 0)
string_out += convert_variable_to_latex_command("LCDMCHISQ", fitted_g_chisq, 1)
string_out += convert_variable_to_latex_command("MONDCHISQ", fitted_sqrt_g, 1)
string_out += convert_variable_to_latex_command("LCDMPTE", pte_g)
string_out += convert_variable_to_latex_command("MONDPTE", pte_sqrt_g, decplaces='sci')
print(string_out)
# End PTE computation

# Make plot
f, ax = plt.subplots(constrained_layout=True, figsize=[5, 3])
plt.rcParams.update({'font.size': 12})

fig_er = plt.errorbar(rsep, p_pw,
             yerr=np.sqrt(np.diag(C_pw)),
             marker='o', ls='', color='black',
             label=r'$\mathrm{Measured~Pairwise~SZ}$')

fig_lcdm_fill = plt.fill_between(rsep,
                 y1=p_sdss_g * (fitted_g + sigma_g),
                 y2=p_sdss_g * (fitted_g - sigma_g),
                 color=cs[0], alpha=0.2,
                 label=r'$\Lambda \mathrm{CDM}$')

fig_lcdm_line = plt.plot(rsep, p_sdss_g * fitted_g, color=cs[0])

fig_mond_fill = plt.fill_between(rsep,
                 y1=p_sdss_sqrt_g * (fitted_sqrt + sigma_sqrt_g),
                 y2=p_sdss_sqrt_g * (fitted_sqrt - sigma_sqrt_g),
                 color=cs[1], alpha=0.2,
                 label=r'$\mathrm{MOND}$')

fig_mond_line = plt.plot(rsep, p_sdss_sqrt_g * fitted_sqrt,
         color=cs[1])

# plot C21 data
firstbin = 2
fname = 'C21_data/L43_S18_ksz_vij_iz1.dat'
tau = 0.54e-4
d_tau = 0.12e-4
T_cmb_over_c = 2.726e6/3e5
df = pd.read_csv(fname, delim_whitespace=True,
                 names=['id1', 'id2', 'r', 'p'])
r = df.r.values[firstbin:]
p = df.p.values[firstbin:]
fig_c21 = plt.plot(r, -p * T_cmb_over_c * tau,
                   ls='--', label=r'$\Lambda\mathrm{CDM~best{-}fit~(Calafut~et~al.~2021)}$', color='black')
plt.xlim([0, 250])

ax.xaxis.set_minor_locator(MultipleLocator(10))
ax.yaxis.set_minor_locator(MultipleLocator(0.005))
handles, labels = plt.gca().get_legend_handles_labels()
order = [3, 1, 2, 0]
#plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
#           fontsize=8)
plt.legend([fig_er,(fig_mond_fill, fig_mond_line[0]), fig_c21[0], (fig_lcdm_fill, fig_lcdm_line[0])], 
           [fig_er.get_label(), fig_mond_fill.get_label(), fig_c21[0].get_label(), fig_lcdm_fill.get_label()],
            fontsize=8 )
plt.axhline(0, color='black')
plt.xlabel(r'$r~[\mathrm{Mpc}]$')
plt.ylabel(r'$\hat{p}_{\mathrm{kSZ}}~[\mathrm{\mu K}]$')
plt.savefig('plots/pksz.pdf')

print(df_table)