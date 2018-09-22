import numpy as np

class LogScaledDistribution(object):
    def __init__(self, dist, a, b, alpha = 0.0, beta = 1.0, dist_args = None):
        if dist_args == None:
            self.dist = dist()
        else:
            self.dist = dist(**dist_args)
        
        self.a = a
        self.b = b
        self.alpha = alpha
        self.beta = beta
    
    def _F(self, x):
        return np.exp((np.log(self.a)*(x - self.beta) + np.log(self.b)*(self.alpha - x))/(self.alpha - self.beta))
    
    def rvs(self, size = None, random_state = None):
        if type(size) == type(None):
            return self._F(self.dist.rvs(random_state = random_state))
        else:
            return self._F(self.dist.rvs(size = size, random_state = random_state))