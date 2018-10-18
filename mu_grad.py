#! /usr/bin/env python

import sys
import numpy as np
from scipy.optimize import minimize
import scipy.linalg as la

def get_mu_grad(mo_energy, mo_coeff, mu, beta):
    """
    gradient corresponding to mf fermi level change term.
    d mu / d v_{kl} [where kl is triu part of the potential]

    Math:
        d mu / d v_{kl} = sum_i f_i * de_{i}/dv_{kl}/ (sum_i f_i)
        where
        f_i = n_i * (1 - n_i) = exp / (1 + exp)^2 

    """
    norb = mo_coeff.shape[0]
    exp_func = np.exp(beta * (mo_energy - mu))
    rho = 1.0 / (1.0 + exp_func)

    E_grad = np.einsum('ki, li -> kli', mo_coeff.conj(), mo_coeff)
    f = exp_func * rho * rho
    
    mu_grad = np.dot(E_grad, f) / (f.sum())
    mu_grad = mu_grad + mu_grad.T
    mu_grad[np.arange(norb), np.arange(norb)] *= 0.5
    mu_grad = mu_grad[np.triu_indices(norb)]
    return mu_grad

def fermi_smearing_occ(mu, mo_energy, beta):
    occ = np.zeros_like(mo_energy)
    de = beta * (mo_energy - mu) 
    occ[de < 100] = 1.0 / (np.exp(de[de < 100]) + 1.0)
    return occ

def kernel(h, nelec, beta, mu0 = None):
    
    mo_energy, mo_coeff = la.eigh(h)
    f_occ = fermi_smearing_occ

    if mu0 is None:
        mu0 = mo_energy[min(nelec, len(mo_energy)) - 1]

    def nelec_cost_fn(mu):
        mo_occ = f_occ(mu, mo_energy, beta)
        return (mo_occ.sum() - nelec)**2

    res = minimize(nelec_cost_fn, mu0, method = 'Nelder-Mead', options = \
            {'maxiter': 10000, 'xatol': 2e-15, 'fatol': 2e-15})
    if not res.success:
        print "WARNING: fitting mu (fermi level) fails."
    mu = res.x
    mo_occ = f_occ(mu, mo_energy, beta)

    return mo_energy, mo_coeff, mo_occ, mu

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
    #np.random.seed(1)

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

    mu_grad = get_mu_grad(mo_energy, mo_coeff, mu, beta)

    print "mu grad analytical"
    print mu_grad

    # check numerical chemical potential derivative
    mu_grad_num = np.zeros_like(mu_grad)
    h_arr = h[np.triu_indices(norb)]

    du = 1e-5
    for i in range(len(h_arr)):
        h_arr_tmp = h_arr.copy()
        h_arr_tmp[i] += du
        h_tmp = triu_arr2mat(h_arr_tmp)
        mu_tmp = kernel(h_tmp, nelec, beta)[-1]
        mu_grad_num[i] = (mu_tmp - mu)/du

    print "mu grad numerical"
    print mu_grad_num
    print "norm diff"
    print la.norm(mu_grad_num - mu_grad)

