import jax.numpy as jnp
import jax.scipy as jscipy

class VanillaOption():
    def __init__(self, S0:float, E:float, T:float, r:float, sigma:float, type:chr='c'):
        '''
        Defines plain vanilla european option ether as a Call or Put,
        and computes common derivatives automatically.

        S0: price of the underlying at t0
        E: strike price
        r: risk free interest rate
        T: time to maturity
        sigma: standard deviation (volatility)
        type: Call (c) or Put (p)
        '''
        self._S0 = S0
        self._E = E
        self._r = r
        self._T = T
        self._sigma = sigma
        self._type = type
        self._price = self._compute_bs_price()

    @property
    def price(self):
        return self._price

    def _compute_bs_price(self):
        '''
            Returns closed form price using Black Scholes equation
        '''
        d1 = (jnp.log(self._S0/self._E) + (self._r + self._sigma**2/2)*self._T) / (self._sigma*jnp.sqrt(self._T))
        d2 = d1 - self._sigma*jnp.sqrt(self._T)
        if self._type == 'c':
            price = self._S0*jscipy.stats.norm.cdf(d1, 0, 1) - self._E*jnp.exp(-self._r*self._T)*jscipy.stats.norm.cdf(d2, 0, 1) 
        elif self._type == 'p':
            price =self._E*jnp.exp(-self._r*self._T)*jscipy.stats.norm.cdf(d2, 0, 1) - self._S0*jscipy.stats.norm.cdf(d1, 0, 1)
        else:
            raise ValueError('Option type must be in (c, p)')
        return price
    
    
