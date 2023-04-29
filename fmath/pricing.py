import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy as jscipy

def mc_stock_price(S0:float, T:float, r:float, sigma:float, samples=1000):
    n = 255*T # assuming trading days in a year
    dt = T/n
    S = jnp.ones([n, samples])*S0
    random_step = jrandom.normal(jrandom.PRNGKey(1), shape=[n-1, samples])
    for i in range(1, n):
        S = S.at[i, :].set(S[i-1, :] * (1 + r*dt + sigma*jnp.sqrt(dt) * random_step[i-1, :]))
    return S


def model_stock_price(S0:float, T:float, r:float, sigma:float, samples=1000):
    random_step  = jrandom.normal(jrandom.PRNGKey(1), shape=[samples, 1])
    S = jnp.ones(samples)*S0
    S = S*jnp.exp(T*(r - 0.5*sigma**2) + sigma*jnp.sqrt(T)*random_step)
    return S


def mc_call_option_price(S0:float, E:float, T:float, r:float, sigma:float, samples=1000):
    random_step  = jrandom.normal(jrandom.PRNGKey(1), shape=[samples])
    S = jnp.ones(samples)*S0
    E = jnp.ones(samples)*E
    S = S*jnp.exp(T*(r - 0.5*sigma**2) + sigma*jnp.sqrt(T)*random_step)
    # Option is excecuted only if it's positive
    option_price_dist = jnp.zeros([samples, 2])
    option_price_dist = option_price_dist.at[:, 0].set(S-E)
    option_price = jnp.max(option_price_dist, axis=1)
    # Discounted price
    option_price = jnp.exp(-r*T)*option_price
    option_price_dist = jnp.exp(-r*T)*option_price_dist
    # Mean of uniform probability distribution is the average
    mean = jnp.sum(option_price)/samples
    return mean, option_price_dist[:, 0]

def bs_call_option_price(S0:float, E:float, T:float, r:float, sigma:float, samples=1000):

    d1 = (jnp.log(S0/E) + (r + sigma**2/2)*T) / (sigma*jnp.sqrt(T))
    d2 = d1 - sigma*jnp.sqrt(T)
    price = S0*jscipy.stats.norm.cdf(d1, 0, 1) - E*jnp.exp(-r*T)*jscipy.stats.norm.cdf(d2, 0, 1) 

    return price