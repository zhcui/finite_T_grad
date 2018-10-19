#! /usr/bin/env python

from system import *
import numpy as np
from scipy.optimize import minimize
import scipy.linalg as la

def get_rho_grad_fix_mu(mo_energy, mo_coeff, mu, beta):
    """
    partial gradient corresponding to rho change term. [mu fixed]
    partial rho_{ij} / partial v_{kl} [where kl is triu part of the potential]

    Math:
        partial rho_ij / partial v_kl = - C_{ip} C^*_{kp} K_pq C_{lq} C^*_{jq}
        where
        K_pq = n_p * (1 - n_q) * [exp(beta * (e_p - e_q)) - 1] / (e_p - e_q)

    """

    norb = mo_coeff.shape[0]
    
    rho_elec = fermi_smearing_occ(mu, mo_energy, beta)
    rho_hole = 1.0 - rho_elec

    de_mat = mo_energy[:, None] - mo_energy
    beta_de_mat = beta * de_mat
    beta_de_mat[beta_de_mat > 300] = 300
    exp_ep_minus_eq = np.exp(beta_de_mat)

    zero_idx = np.where(np.abs(de_mat) < 1.0e-13)
    de_mat[zero_idx] = np.inf
    de_mat_inv = 1.0 / de_mat

    K = np.einsum('p, q, pq, pq -> pq', rho_elec, rho_hole,\
            exp_ep_minus_eq - 1.0, de_mat_inv)

    for p, q in zip(*zero_idx):
        K[p, q] = rho_elec[p] * rho_hole[q] * beta

    rho_grad = -np.einsum('mp, lp, pq, sq, nq -> lsmn', \
            mo_coeff, mo_coeff.conj(), K, mo_coeff, mo_coeff.conj())
    # symmetrize
    rho_grad = rho_grad + rho_grad.transpose(1,0,2,3)
    rho_grad[np.arange(norb), np.arange(norb)] *= 0.5
    rho_grad = rho_grad[np.triu_indices(norb)]

    return rho_grad



if __name__ == '__main__':
    
    np.set_printoptions(5, linewidth =1000)
    np.random.seed(1)

    norb = 8
    nelec = 5
    beta = 10.0

    #deg_orbs = []
    #deg_energy = []
    deg_orbs = [[0,3], [1,2], [4,5,6], [7]]
    deg_energy = [1.0 , 0.1, 0.8, 3.0]
    h = get_h_random_deg(norb, deg_orbs = deg_orbs, deg_energy = deg_energy)
    
    #h = np.eye(h.shape[0])
    #
    #for i in range(h.shape[0]):
    #    h[i, i] +=  i

    print "h"
    print h

    mo_energy, mo_coeff, mo_occ, mu = kernel(h, nelec, beta)

    print "mo_energy"
    print mo_energy
    print "mo_occ"
    print mo_occ
    print "mu"
    print mu

    rho_grad = get_rho_grad_fix_mu(mo_energy, mo_coeff, mu, beta)

    print "rho grad analytical"
    print rho_grad

    # check numerical chemical potential derivative

    rho_grad_num = np.zeros_like(rho_grad) 
    rho_ao_ref = make_rdm1(mo_occ, mo_coeff)

    print "mo_occ"
    print mo_occ
    print "rho_ao_ref"
    print rho_ao_ref

    h_arr = h[np.triu_indices(norb)]

    du = 1.0e-6
    for i in range(len(h_arr)):
        h_arr_tmp = h_arr.copy()
        h_arr_tmp[i] += du
        h_tmp = triu_arr2mat(h_arr_tmp)

        # keep mu unchanged
        mo_energy_tmp, mo_coeff_tmp, mo_occ_tmp, _ = \
                kernel(h_tmp, nelec, beta, mu0 = mu, fix_mu = True)
        rho_ao_tmp = make_rdm1(mo_occ_tmp, mo_coeff_tmp)
        rho_grad_num[i] = (rho_ao_tmp - rho_ao_ref)/du

    print "rho_grad numerical"
    print rho_grad_num
    
    print "norm diff"
    print la.norm(rho_grad - rho_grad_num)
