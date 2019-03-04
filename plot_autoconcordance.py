import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def set_style():
    # This sets reasonable defaults for font size for
    # a figure that will go in a paper
    sns.set_context("paper")
    # Set the font to be serif, rather than sans
    sns.set(font='serif', font_scale=1.5)
    sns.set_palette('muted')
    # Make the background white, and specify the
    # specific font family
    sns.set_style("ticks", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })


plt.rcParams["text.usetex"] = True
set_style()
saving_fig = False  # active to True to save images


def w_f(tau, nu):

    if nu == 2:
        return (np.exp(tau) - tau - 1) / tau ** 2

    elif nu == 3:
        return (-tau - np.log(1 - tau)) / tau ** 2

    elif nu == 4:
        return ((1 - tau) * np.log(1 - tau) + tau) / tau ** 2

    else:
        c1 = (nu - 2) / (4 - nu)
        c2 = (nu - 2) / (2 * (3 - nu))
        c3 = 2 * (3 - nu) / (2 - nu)
        return (c1 / tau) * ((c2 / tau) * ((1 - tau) ** c3 - 1) - 1)


tau = np.arange(-10, 2., 0.101)
plt.figure()
for nu in [2, 3, 4, 5, 6]:
    w2 = w_f(tau, nu)
    plt.plot(tau, w2, label=r"$\nu = $" + str(nu))

plt.xlabel(r"$\tau$")
plt.ylabel(r"$w_{\nu}(\tau)$")
plt.legend()
plt.tight_layout()
if saving_fig:
    if not os.path.exists("img"):
        os.makedirs("img")
    plt.savefig("img/autoconco.pdf", format="pdf")
plt.show()
