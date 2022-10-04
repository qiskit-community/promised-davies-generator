# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License. 

import numpy as np

def colstack(rho):
    return rho.reshape((rho.shape[0]**2,1),order="F") # column stacked

def uncolstack(rho):
    dim = int(np.sqrt(rho.shape[0]))
    return rho.reshape((dim,dim),order="F") 

# calJ is the dictionary lists of (x,y) pairs for each omega
# pi is a function so that pi(x) is the projector onto the x'th energy space
# couplers is a list of coupling operators. These must be hermitian,
#    and the sum of their spectral norms squared must be 1. The function
#    checks neither of these.
def get_lindblad(calJ, pi, couplers, beta):
    dim = pi(0).shape[0]
    
    def G(omega):
        return min(1, np.exp(-omega*beta))
    
    def S(omega, coupler):
        S_omega = np.zeros((dim,dim)).astype(complex)
        for (x,y) in calJ[omega]:
            S_omega += pi(x) @ coupler @ pi(y)
        return S_omega
    
    out = np.zeros((dim**2,dim**2)).astype(complex)

    for coupler in couplers:
        for omega in calJ.keys():
            L = np.sqrt(G(omega)) * S(omega, coupler)
            out += np.kron(L, L.conj())
            LL = L.conj().T @ L
            out -= 0.5 * np.kron(LL, np.eye(dim))
            out -= 0.5 * np.kron(np.eye(dim), LL.T)

    return out

def get_spectral_gap(L):
    L_eigvals = np.real(np.linalg.eigvals(L))

    error = None
    
    if len(list(filter(lambda x: np.allclose(x,0),L_eigvals))) != 1:
        error = "Non-degenerate nullspace."

    unique_eigvals = np.unique(L_eigvals)
    gap = -np.max(list(filter(lambda x: not np.allclose(x,0), unique_eigvals)))

    return gap, error


def check_steady_state(L, rho):
    rho_vec = colstack(rho)
    return np.allclose( np.zeros(rho_vec.shape), L @ rho_vec)


def get_steady_state(L):
    error = None

    # only extract one vector in the null space
    #steady_vec = np.linalg.lstsq(L, np.zeros(L.shape[0]),rcond=None)[0]
    L_eigvals, L_eigvecs = np.linalg.eig(L)

    if len(list(filter(lambda x: np.allclose(x,0),L_eigvals))) != 1:
        error = "Non-degenerate nullspace."

    steady = uncolstack(L_eigvecs[:,0])
    
    if np.allclose(np.trace(steady),0):
        return steady, "Steady state is traceless."

    steady /= np.trace(steady)

    if error is not None and not np.allclose(steady, steady.conj().T):
        error = "Steady state is not hermitian."
    
    if error is not None and not all(np.real(np.eigvals(steady)) >= 0):
        error = "Steady state is not positive semi-definite."

    return steady, error


############## Synthesis of coupling operators
## These are just pauli-X and pauli-Z on each qubit

def single_qubit_operator(eigv, idx, nqubits, matkey):
    mat = np.array({
        "X": [[0,1],[1,0]],
        "Z": [[1,0],[0,-1]],
        }[matkey]).astype(complex)
    
    out = np.eye(2**idx).astype(complex)
    out = np.kron(out, mat)
    out = np.kron(out, np.eye(2**(nqubits-idx-1)).astype(complex))
    
    return eigv.conj().T @ out @ eigv

def make_couplers(eigv):
    nqubits = int(np.log2(eigv.shape[0]))
    norm_factor = (2*nqubits)**(-0.5)
    return [single_qubit_operator(eigv, i,nqubits,key)*norm_factor
            for i in range(nqubits)
            for key in ["X","Z"]]