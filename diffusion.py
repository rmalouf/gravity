#!/opt/local/bin/python3

import random
from math import log, sqrt, cos, sin, pi
from collections import Counter, namedtuple
from multiprocessing import Pool

from toolz import concat

import numpy as np
import scipy
import statsmodels.api as sm

import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('seaborn-white')
plt.rcParams['font.size'] = 7
plt.rcParams['axes.labelweight'] = 'normal'

class Landscape(object):

    def __init__(self, N, n_pop):

        # create populations

        self.N = N
        pop = list(range(N))
        for _ in range(n_pop-N):
            pop.append(random.choice(pop))
        self.pop = np.array(sorted(Counter(pop).values(), reverse=True))

        # lay out locations
        
        self.x, self.y = np.zeros((N)), np.zeros((N))
        for i in range(1, N):
            r, theta = self.generate_position()
            self.x[i] = r * cos(theta)
            self.y[i] = r * sin(theta)

    def plot(self, filename, scores=None):
    
        plt.figure(figsize=(2.5,2.5))
        plt.tick_params(labelbottom='off', labelleft='off')
        if scores:  
            plt.scatter(self.x, self.y, s=self.pop/10, alpha=0.3, c=scores, cmap='viridis')
        else:
            plt.scatter(self.x, self.y, s=self.pop/10, alpha=0.3)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.savefig(filename)
        plt.close()

class UniformLandscape(Landscape):
    def generate_position(self):  
        """Uniformly distributed within a unit circle"""     
        return sqrt(np.random.random()), np.random.random() * 2 * pi

class NonUniformLandscape(Landscape):
    def generate_position(self):  
        """Non-uniformly distributed within a unit circle"""     
        return np.random.power(4), np.random.random() * 2 * pi

class Simulation(object):
    
    def __init__(self, L, V):
    
        self.L = L        
        self.V = V

        # initialize agents

        self.agents = [ np.zeros((pop, V)) for pop in L.pop ]
        self.agents[0][:] = 1.0

        # pre-compute distances

        self.distance = np.zeros((L.N))
        for i in range(L.N):
            self.distance[i] = sqrt((L.x[0]-L.x[i])**2+(L.y[0]-L.y[i])**2)        


    def run(self, R, q):
    
        for _ in range(R):
            # choose a destination city (other than the capital), weighted by distance from capital
            c = multinomial(self.p)
            a = np.random.randint(0, self.L.pop[c])
            # diffuse
            for i in range(self.V):
                if np.random.random() < q:
                    self.agents[c][a, i] = 1
                    
        self.dialects = [ 1-np.dot(normalize(np.sum(self.agents[0], axis=0)), 
                                   normalize(np.sum(a, axis=0))) 
                                for a in self.agents ]

    def results(self):
        return self.distance[1:], self.L.pop[1:], self.dialects[1:]

    def plot_distance(self, filename):
    
        plt.figure(figsize=(4,4))
        x, y = zip(*[(a,b) for (a,b) in zip(self.distance[1:], self.dialects[1:]) if b < 2.0])
        plt.scatter(x, y, alpha=0.3)
        x, y = zip(*sm.nonparametric.lowess(y, x, frac=.5))
        plt.plot(x,y)
        plt.xlabel('Geographic distance')
        plt.ylabel('Dialect difference')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def plot_population(self, filename):
    
        plt.figure(figsize=(4,4))
        x, y = zip(*[(log(a),b) for (a,b) in zip(self.L.pop[1:], self.dialects[1:]) if b < 2.0])
        x = x[1:]
        y = y[1:]
        plt.scatter(x, y, alpha=0.3)
        x1, y1 = zip(*sm.nonparametric.lowess(y, x, frac=.5))
        plt.plot(x1, y1)
        plt.xlabel('Population (log)')
        plt.ylabel('Dialect difference')
        plt.xlim(0, max(x))
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

class Gravity(Simulation):
    def __init__(self, L, V):
        super().__init__(L, V)       
        self.p = self.L.pop/self.distance
        self.p[0] = 0.0

class Radiation(Simulation):
    def __init__(self, L, V):
        super().__init__(L, V)       

        s = np.zeros((L.N))
        for i in range(L.N):
            for j in range(L.N):
                if self.distance[j] <= self.distance[i]:
                    s[i] += self.L.pop[j]
            s[i] = s[i] - self.L.pop[0] - self.L.pop[i]
        s[0] = 0.0

        self.p = (self.L.pop[0]*self.L.pop) / ((self.L.pop[0]+s)*(self.L.pop[0]+self.L.pop+s)) 
        self.p[0] = 0.0

        
def multinomial(p):
    """Draw from a multinomial distribution"""
    N = np.sum(p)
    p = p/N
    return int(np.random.multinomial(1, p).nonzero()[0])

def normalize(v):
    """Normalize a vector"""
    L = np.sqrt(np.sum(v**2))
    if L > 0:
        return v/L
    else:
        return v
        
def main():

    P = 0.01
    R = 50000

    uniform = UniformLandscape(500, 100000)
    print('fig1.pdf : uniform landscape')
    uniform.plot('fig1.pdf')

    expr1 = Gravity(uniform, 100)
    expr1.run(R, P)
    print('fig2.pdf : distance (gravity, uniform)')
    expr1.plot_distance('fig2.pdf')
    print('fig3.pdf : population (gravity, uniform)')
    expr1.plot_population('fig3.pdf')

    expr2 = Radiation(uniform, 100)
    expr2.run(R, P)
    print('fig4.pdf : distance (radiation, uniform)')
    expr2.plot_distance('fig4.pdf')
    print('fig5.pdf : distance (population, uniform)')
    expr2.plot_population('fig5.pdf')

    if False:
        non_uniform = NonUniformLandscape(500, 100000)
        print('fig6.pdf : non-uniform landscape')
        non_uniform.plot('fig6.pdf')
       
        expr3 = Gravity(non_uniform, 100)
        expr3.run(R, P)
        print('fig7.pdf : distance (gravity, non_uniform)')
        expr3.plot_distance('fig7.pdf')
        print('fig8.pdf : population (gravity, non_uniform)')
        expr3.plot_population('fig8.pdf')

        expr4 = Radiation(non_uniform, 100)
        expr4.run(R, P)
        print('fig9.pdf : distance (radiation, non_uniform)')
        expr4.plot_distance('fig9.pdf')
        print('fig10.pdf : distance (population, non_uniform)')
        expr4.plot_population('fig10.pdf')


main()
