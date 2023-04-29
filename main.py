import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.collections as mcoll
import fmath
from fmath import VanillaOption

plt.rcParams['figure.figsize'] = [16.0, 10.0]
plt.rcParams['font.size'] = 15
CMAP = plt.get_cmap('copper')

st.title('Option and Stock simulation app')
st.header('Use following options to configure the simulation')

st.subheader('Underlying parameters')
S0 = st.slider('Spot price', 100, 1000, 100, 100, key="1")
r = st.slider('Risk free interest rate', 0.0, 0.1, 0.02, 0.01, key="2")
sigma = st.slider('Volatility', 0.0, 1.0, 0.2, 0.05, key="3")
st.subheader('Option parameters')
E = st.slider('Strike price', 0, 2000, 100, 10, key="4")
T = st.slider('Maturity in years', 0, 10, 5, 1, key="5")
st.subheader('Number of paths (distributions are multiplied by this)')
N = st.number_input('Number of simulations', 1, 200, 160, 10, key="6")



def get_line_collection(x:np.array, y:np.array, cmap:plt.colormaps=plt.get_cmap('copper'), alpha_min=0.02):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    z = np.linspace(0.0, 1.0, len(x))
    alphas = np.linspace(alpha_min, 1, len(x))
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, linewidths=(0.8), norm=plt.Normalize(0.0, 1.0), alpha=alphas)
    return lc

def run_stock_simulation():
    plt.figure('Stock price process')
    plt.clf()

    # Stock simulations
    left = plt.subplot2grid((1, 3), (0, 0), rowspan=1, colspan=2)
    S = fmath.mc_stock_price(S0, T, r, sigma, N)
    for i in range(len(S[0])):
        x = np.linspace(0, len(S), len(S))
        lc = get_line_collection(x, S[:,i])
        plt.gca().add_collection(lc)
    left.plot(S[:, 0], c='r', lw=0.8, label='Possible stock path')
    ticks = np.arange(1, len(S[:,0]), 255)
    plt.xticks(ticks, np.arange(0, T, 1))
    plt.margins(x=0)
    plt.ylabel('Price')
    plt.xlabel('Time [Years]')
    plt.legend()

    # Stock model prices
    right = plt.subplot2grid((1, 3), (0, 2), rowspan=1, colspan=1)
    S = fmath.mc_stock_price(S0, T, r, sigma, N*20)
    s = fmath.model_stock_price(S0, T, r, sigma, N*20)
    right.hist(s[:,0], bins=70, color='r', alpha=0.5, ec='black', orientation="horizontal", label='Distribution of\nStock pricing model at $T$')
    right.hist(S[-1,:], bins=70, color='grey', alpha=0.5, orientation="horizontal", label='Distribution of\nsimulations at $T$')
    mean = np.mean(S[-1,:])
    plt.axhline(y=mean, c='black', ls='-.', label=f'Mean is {mean:.1f}')

    plt.setp(plt.gca(), ylim=left.get_ylim())
    plt.ylabel('Price')
    plt.gca().yaxis.tick_right()
    plt.gca().yaxis.set_label_position("right")
    plt.gca().set_xticks([])
    plt.xlabel('Density')
    plt.legend()

    plt.subplots_adjust(wspace=0)
    plt.gcf().suptitle(f'{N} Stock simulations run with [$r_f={r}$, $\sigma={sigma}$, $T={T}$]')
    st.pyplot(plt.gcf())

def run_option_simulation():
    mean, dist = fmath.mc_call_option_price(S0, E, T, r, sigma, samples=N*200)
    dist = dist[dist < 2*S0]
    option = VanillaOption(S0, E, T, r, sigma)
    analytical_mean = option.price

    plt.figure('Call Option pricing')
    plt.clf()
    # Histogram
    n, bins, patches = plt.hist(dist, bins=100, alpha=1, label=f'MC Call distribution with {N*200} samples')
    z = np.linspace(0.0, 1.0, len(bins))
    for p, c in zip(patches, z):
        plt.setp(p, 'facecolor', CMAP(c))
    for i in range(0, len(bins[bins<=0]) -1):
        patches[i].set_alpha(0.4)
    # Mean lines
    plt.axvline(x=mean, c='black', ls='-.', label=f'Mean of MC Black Scholes price distribution is {mean:.1f}')
    plt.axvline(x=analytical_mean, c='orange', ls='-.', label=f'Mean of Analytical Black Scholes price is {analytical_mean:.1f}')

    plt.legend()
    plt.ylabel('Density')
    plt.xlabel('Present value')
    plt.title(f'Call option [$S_0={S0}, E={E}, T={T}, r_f={r}, \sigma={sigma}$]')
    st.pyplot(plt.gcf())

state = st.button("Run")
if state == True:
    run_stock_simulation()
    run_option_simulation()
