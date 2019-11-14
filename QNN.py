import numpy as np
import random
from math import log, pi

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram
from qiskit.aqua import Operator, run_algorithm

from scipy.optimize import minimize
from pyswarm import pso

#from .particle import PSO


def generate_bars_and_stripes(length, num_samples):
    """
    Creates a dataset containing samples showing bars or stripes.
    """
    data = np.zeros((num_samples, length * length))
    for i in range(num_samples):
        values = np.dot(np.random.randint(low=0, high=2,
                                              size=(length, 1)),
                          np.ones((1, length)))
        if np.random.random() > 0.5:
            values = values.T
        data[i, :] = values.reshape(length * length)
    return data

def get_target(data):
    '''
    Obtain a target probability distribution from a given dataset
    '''
    target = {}
    for sample in data:
        key = ''.join(map(str, [int(i) for i in sample]))
        if key not in target.keys():
            target[key] = 1
        else:
            target[key] += 1

    # Normalize target distribution before returning
    for key in target.keys():
        target[key] /= len(data)

    return target

def KLDiv(target, learned):
    '''
    Compute the  KL Divergence between a learned and target distribution
    '''
    epsilon = 0.01
    cost = 0

    for key in target:
        if key not in learned:
            learned[key] = 0         # adds any necessary keys to learned for which we get no counts

        if target[key] != 0:
            cost += target[key]*(log(target[key]) - log(max(epsilon, learned[key])))

    return cost


'''
Functions for building and running circuits
'''

def rotations(circ, q, n, params):
    '''
    Apply 2-parameter rotations to each qubit
    '''
    for i in range(n):
        circ.u3(params[2*i], params[2*i+1], 0, q[i])

def XX(circ, q1, q2, n, angle):
    '''
    Implements paramaterized Molmer-Sorensen XX gate
    '''
    circ.cx(q1, q2)
    circ.rx(angle, q1)
    circ.cx(q1, q2)

def FCcx(circ, q, n):
    '''
    Fully connected CNOTs
    Applies a CNOT between every pair of qubits,
    where every qubit gets to be the control and target of every other qubit
    '''
    for i in range(n):
        for j in range(n):
            if i != j:
                circ.cx(q[i], q[j])

def FCXX(circ, q, n, params):
    '''
    Fully connected XX
    Applies a paramaterized Molmer-Sorensen XX gate between every pair of qubits
    '''
    a = 0
    for i in range(n):
        for j in range(i+1, n):
            XX(circ, q[i], q[j], n, params[a])
            a += 1

def layercx(circ, q, n, params):
    '''
    Creates layer of rotations followed by fully connected CNOTs
    '''
    rotations(circ, q, n, params)
    FCcx(circ, q, n)

def layerXX(circ, q, n, params):
    '''
    Creates layer of rotations followed by fully connected XX gates
    '''
    rotations(circ, q, n, params[:2*n])
    FCXX(circ, q, n, params[2*n:])


class QuantumBornMachine:
    '''
    The Quantum Born Machine is trained to be able to reproduce samples from
    a target probability distribution.
    See eg. https://arxiv.org/pdf/1908.10778.pdf
    '''
    def __init__(self, n, backend, shots = 3000, entangler = 'XX'):
        '''
        n: number of qubits in register
        backend: the quantum computing backend used. For now just the Aer qasm qasm_simulator
        shots: number of times circuit is run during each evaluation; more shots will give
               more consistent statistics and thus less noise when training
        entangler: the choice of entangling gate used. Parameterized XX is fairly optimal
        '''

        self.n = n
        self.backend = backend
        self.shots = shots
        self.entangler = entangler

        self.q = QuantumRegister(self.n)
        self.c = ClassicalRegister(self.n)

        self.params = {}           # parameters of the circuit
        self.learnedparams = {}    # the learned optimal parameters
        self.learned = {}          # the learned output distribution
        self.target = {}           # the ditribution we are trying to match

    def build(self, params, layers):
        '''
        Initializes and builds the circuit according to a chosen number of layers
        and a set of parameters
        '''
        self.circ = QuantumCircuit(self.q, self.c)
        self.params = params
        self.layers = layers

        n = self.n
        for k in range(layers):             # Build circuit layer by layer
            if self.entangler == 'cx':
                layercx(self.circ, self.q, n, params[2*n*k : 2*n*(k+1)])
            if self.entangler == 'XX':
                perlayer = int(n*(n+3)/2)
                layerXX(self.circ, self.q, n, params[perlayer*k : perlayer*(k+1)])


    def run(self):
        '''
        Runs circuit and outputs measurement statistics
        '''
        for i in range(self.n):
            self.circ.measure(self.q[i], self.c[i])

        output = execute(self.circ, self.backend, shots = self.shots).result().get_counts(self.circ)
        for key in output:
            output[key] /= self.shots     # normalize counts from measurements

        return output

    def cost(self, target):
        '''
        Obtain cost (KL divergence) of the circuit output wrt a target distribution
        '''
        output = self.run()
        cost = KLDiv(target, output)
        return cost

    def costParams(self, params):
        '''
        Obtain cost (KL divergence) from parameters of circuit, assuming a
        target distribution already specified in self.target
        This is the function over which we run the optimization
        '''
        self.build(params, self.layers)
        cost = self.cost(self.target)
        return cost

    def train(self, target, method = 'COBYLA'):
        '''
        Primary method to train the circuit with respect to a provided target distribution
        '''
        self.target = target
        ret = {}

        method = 'COBYLA'   # TEMPORARY: constrain until more options added
        if method == 'COBYLA':
            opt = minimize(self.costParams, self.params, method='COBYLA', tol = 1e-8)
            self.learnedparams = opt.x
            self.learned = self.run()
            ret = {'params': opt.x, 'cost': opt.fun}

        return ret
