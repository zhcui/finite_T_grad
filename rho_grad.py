#! /usr/bin/env python

from system import *
import numpy as np
from scipy.optimize import minimize
import scipy.linalg as la

def get_rho_grad(mo_energy, mo_coeff, mu, beta):
    """
    partial gradient corresponding to rho change term.
    partial rho_{ij} / partial v_{kl} [where kl is triu part of the potential]

    Math:
        partial rho_ij / partial v_kl = - C_{ip} C^*_{kp} K_pq C_{lq} C^*_{jq}
        where
        K_pq = n_p * (1 - n_q) * [exp(beta * (e_p - e_q)) - 1] / (e_p - e_q)

    """


de_mat = mo_energy[:,None] - mo_energy

exp_ep_minus_eq = np.exp(beta * de_mat)

zero_idx = np.where(np.abs(de_mat) < 1.0e-6)
de_mat[zero_idx] = np.inf
de_mat_inv = 1./de_mat

K = np.einsum('p, q, pq, pq -> pq', occ_func, 1.0 - occ_func, exp_ep_minus_eq - 1.0, de_mat_inv)

#K[zero_idx] =  beta * (1.0 - occ_func[zero_idx[0]]) * occ_func[zero_idx[0]]
for idxp, p in enumerate(zero_idx[0]):
    q = zero_idx[1][idxp]
    K[p, q] = occ_func[p] * (1 - occ_func[q]) * beta

print "K"
print K

rho_grad = -np.einsum('mp, lp, pq, sq, nq -> lsmn', mo_coeff, mo_coeff.conj(), K, mo_coeff, mo_coeff.conj())
# symmetrize
rho_grad = rho_grad + rho_grad.transpose(1,0,2,3)
rho_grad[np.arange(norb), np.arange(norb)] *= 0.5
print la.norm(rho_grad - rho_grad.transpose(1,0,2,3))
rho_grad = rho_grad[np.triu_indices(norb)]

print rho_grad






    norb = mo_coeff.shape[0]
    
    rho_elec = np.zeros(norb)
    de = beta * (mo_energy - mu) 
    rho_elec[de < 100] = 1.0 / (np.exp(de[de < 100]) + 1.0)

    rho_hole = np.zeros(norb)
    de = -de
    rho_hole[de < 100] = 1.0 / (np.exp(de[de < 100]) + 1.0)
    
    #f = exp_func * rho * rho
    f = rho_elec * rho_hole    

    E_grad = np.einsum('ki, li -> kli', mo_coeff.conj(), mo_coeff)
    
    mu_grad = np.dot(E_grad, f) / (f.sum())
    mu_grad = mu_grad + mu_grad.T
    mu_grad[np.arange(norb), np.arange(norb)] *= 0.5
    mu_grad = mu_grad[np.triu_indices(norb)]
    return mu_grad


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

rho_grad_num = np.zeros_like(rho_grad) 


#rho_mat_ref = la.inv(np.eye(norb) + la.expm(beta*(h - mu * np.eye(norb))))
'''
print rho_mat_ref
print la.expm(beta*(h - mu * np.eye(norb)))
exit()
'''
dm_mo = 1.0 / (1.0 + np.exp(beta * (mo_energy - mu)))
#print dm_mo
dm_ao = np.einsum('ip, p, jp -> ij', mo_coeff, dm_mo, mo_coeff.conj())
rho_mat_ref = dm_ao

print "rho_mat_ref"
print rho_mat_ref
#exit()

h_arr = h[np.triu_indices(norb)]
#array2matrix( array, Nimp, mtype = np.double)

du = 1.0e-3
for i in xrange(len(h_arr)):
    h_arr_tmp = h_arr.copy()
    h_arr_tmp[i] += du
    h_tmp = array2matrix(h_arr_tmp, norb/2, mtype = np.double)

    e, c = la.eigh(h_tmp)
    res_tmp = get_mu(mu, e, Nelec_target, beta = beta, smear_type = 'Fermi')
    mu_tmp = res_tmp.x[0]
    mu_tmp = mu
    print "mu"
    print mu
    print res_tmp.x[0]
    dm_mo_tmp = 1.0 / (1.0 + np.exp(beta * (e - mu_tmp)))
    dm_ao_tmp = np.einsum('ip, p, jp -> ij', c, dm_mo_tmp, c)
    rho_mat = dm_ao_tmp
    #rho_mat = la.inv(np.eye(norb) + la.expm(beta*(h_tmp - mu * np.eye(norb))))
    
    rho_grad_num[i] = (rho_mat - rho_mat_ref)/du

    '''
    e, c = la.eig(h_tmp)
    idx_sort = np.argsort(e)
    e = e[idx_sort]
    c = c[:, idx_sort]
    #print e
    #print c
    #exit()

    mo_occ_tmp = c[:, :num_occ]
    dm_tmp = mo_occ_tmp.dot(mo_occ_tmp.conj().T)
    #dm_tmp = (mo_occ.dot(mo_occ_tmp.conj().T) + (mo_occ_tmp.dot(mo_occ.conj().T)))/2.0
    B_num[:,:, i] = (dm_tmp - dm)/du
    '''

print "rho_grad_num"
print rho_grad_num

#print np.argmax(np.abs(rho_grad - rho_grad_num))
print "norm diff"
print la.norm(rho_grad - rho_grad_num)
print "DIFF"
print np.abs(rho_grad - rho_grad_num)
#print (rho_grad / rho_grad_num)

exit()
