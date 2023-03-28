import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import glob
import matplotlib

font = {'size': 18}
matplotlib.rc('font', **font)
show = True


def power(r, a, b):
    return a * r ** (-b)


firstbin = 5
fnames = glob.glob('L*S18*.dat')
plt.figure(figsize=[8, 4.5])
for j, fname in enumerate(fnames):
    df = pd.read_csv(fname, delim_whitespace=True,
                     names=['id1', 'id2', 'r', 'p'])

    r = df.r.values
    p = df.p.values

    sol = curve_fit(power, r[firstbin:], p[firstbin:], (2, 0.8))[0]

    print(sol)

    x = np.linspace(df.r.values[firstbin], df.r.max(), 500)
    plt.plot(x, power(x, sol[0], sol[1]))

    plt.scatter(df.r, df.p, marker='.', label=fname + ' b=%1.1f' % sol[1])

plt.ylim([0, 180])
plt.legend(fontsize=10)
if show:
    plt.show()
else:
    plt.savefig('theory_curves.png', dpi=120)
