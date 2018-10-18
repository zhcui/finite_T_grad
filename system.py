#! /usr/bin/env python

import numpy as np
from scipy.optimize import minimize
import scipy.linalg as la

def fermi_smearing_occ(mu, mo_energy, beta):
    # get rho_mo
    occ = np.zeros_like(mo_energy)
    de = beta * (mo_energy - mu) 
    occ[de < 100] = 1.0 / (np.exp(de[de < 100]) + 1.0)
    return occ

def kernel(h, nelec, beta, mu0 = None, fix_mu = False):
    
    mo_energy, mo_coeff = la.eigh(h)
    f_occ = fermi_smearing_occ

    if mu0 is None:
        mu0 = mo_energy[min(nelec, len(mo_energy)) - 1]

    def nelec_cost_fn(mu):
        mo_occ = f_occ(mu, mo_energy, beta)
        return (mo_occ.sum() - nelec)**2

    if not fix_mu:
        res = minimize(nelec_cost_fn, mu0, method = 'Nelder-Mead', options = \
                {'maxiter': 10000, 'xatol': 2e-15, 'fatol': 2e-15})
        if not res.success:
            print "WARNING: fitting mu (fermi level) fails."
        mu = res.x[0]
    else:
        mu = mu0
    mo_occ = f_occ(mu, mo_energy, beta)

    return mo_energy, mo_coeff, mo_occ, mu

def make_rdm1(mo_occ, mo_coeff):
    return mo_coeff.dot(np.diag(mo_occ).dot(mo_coeff.conj().T))

def get_h_random(norb, seed = None):
    
    if seed is not None:
        np.random.seed(seed)
    h = np.random.random((norb, norb))
    h = h + h.T.conj()
    return h

def get_h_random_deg(norb, deg_orbs = [], deg_energy = [], seed = None):
    
    if seed is not None: 
        np.random.seed(seed)
    h = np.random.random((norb, norb))
    h = h + h.T.conj()
    mo_energy, mo_coeff = la.eigh(h)

    for i in range(len(deg_orbs)):
        mo_energy[deg_orbs[i]] = deg_energy[i]

    h = mo_coeff.dot(np.diag(mo_energy).dot(mo_coeff.T.conj()))

    return h

def triu_mat2arr(mat):
    norb = mat.shape[0]
    return mat[np.triu_indices(norb)]

def triu_arr2mat(arr):
    norb = int(np.sqrt(len(arr) * 2))
    mat = np.zeros((norb, norb), dtype = arr.dtype)
    mat[np.triu_indices(norb)] = arr
    mat = mat + mat.T.conj()
    mat[np.arange(norb), np.arange(norb)] *= 0.5
    return mat

if __name__ == '__main__':
    
    np.set_printoptions(5, linewidth =1000)
    np.random.seed(1)

    norb = 8
    nelec = 5
    beta = 20.0

    #deg_orbs = []
    #deg_energy = []
    deg_orbs = [[0,3], [1,2], [4,5,6], [7]]
    deg_energy = [1.0 , 0.1, 0.8, 3.0]
    h = get_h_random_deg(norb, deg_orbs = deg_orbs, deg_energy = deg_energy)

    print "h"
    print h

    mo_energy, mo_coeff, mo_occ, mu = kernel(h, nelec, beta)

    print "mo_energy"
    print mo_energy
    print "mo_occ"
    print mo_occ
    print "mu"
    print mu


