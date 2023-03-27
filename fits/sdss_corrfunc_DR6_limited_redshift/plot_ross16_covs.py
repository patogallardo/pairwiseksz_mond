import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from statsmodels.stats.moment_helpers import cov2corr

fnames = ['Ross_2016_COMBINEDDR12_zbin2_covariance_monoquad_post_recon_bincent0.dat']
covs_dir = "Ross_2016_COMBINEDDR12"

fname = fnames[0]

df = pd.read_csv(os.path.join(covs_dir, fname), comment='#',
                 delim_whitespace='True', names=range(72))

corr = cov2corr(df.values)

sns.heatmap(corr)
plt.savefig(os.path.join("plots", "%s.pdf" % fname))
plt.close()
