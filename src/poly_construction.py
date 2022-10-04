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
from math import erf
from scipy.special import eval_chebyt

# Short-hand for applying a function to the Hamiltonian.
# Essentially implements Lemma 34 without the Chebyshev part.
def apply_to_H(f, eigs):
    return np.diag([-eval_chebyt(3,f(eig)/2) for eig in eigs])

# Lemma 33 in Appendix A.
# Rather than actually use the polynomial, we leverage 
# that the polynomial is actually an exponentially-accurate
# approximation of erf(kx). 
def scaled_erf(eig, l,r, delta=0.001):
    assert l <= r

    # neglect exponentially accurate tails
    if eig <= l: return 0
    if eig >= r: return 1

    midpoint = np.mean([l,r])
    width = r-l

    # Lemma 10 of 1707.05391, to determine 
    # when erf(kx) is delta-close to 1 or -1.
    kappa = 0.5 * np.sqrt(2 * np.log(2 / (np.pi * (2*delta)**2)))
    return erf( (eig - midpoint) * 2*kappa/width )*0.5 + 0.5

# Evaluates the result of Lemma 34
def projection_poly(ai,bi,ci):
    def eval_poly(eig):
        for i in range(len(ai)):
            if eig < ai[i]:
                if i == 0:
                    # less than a[0]
                    return ci[0]
                # in between bi[i-1], ai[i]
                if ci[i-1] == ci[i]:
                    return ci[i]
                
                if ci[i] == 1: # increasing
                    return scaled_erf(eig,bi[i-1],ai[i])

                # decreasing
                return 1-scaled_erf(eig,bi[i-1],ai[i])

            if eig <= bi[i]:
                # in between a[i], b[i]
                return ci[i]

        # greater than b[-1]
        return ci[-1]

    return eval_poly
     

# Eigenvalue transformation for P_x from Proposition 8
# This implementation does not correctly capture behavior
# outside of the rounding promise, but that doesn't matter
# because we will truncate the P_x to the promised subspace anyway.
def promised_energy_projector_f(x, promise):
    ai,bi = zip(*promise)
    ci = [1 if i==x else 0 for i in range(len(promise))]
    return projection_poly(ai,bi,ci)

def promised_energy_projector(eigs,x, promise):
    f = promised_energy_projector_f(x, promise)
    return apply_to_H(f,eigs)


# Eigenvalue transformation for A^(M) from Lemma 29
def attenuation_operator_f(promise, gamma):
    min_gap = 1
    for x in range(1,len(promise)):
        this_gap = promise[x][0] - promise[x-1][1] 
        if this_gap < min_gap: min_gap = this_gap
    w = gamma * min_gap

    ai_bi_ci = []
    for i in range(len(promise)):
        l,r = promise[i]
        ai_bi_ci.append((l+w, r-w, 1))
        if i != len(promise)-1:
            next_l,_  = promise[i+1]
            ai_bi_ci.append((r, next_l, 0))

    return projection_poly(*zip(*ai_bi_ci))

def attenuation_operator(eigs, promise, gamma):
    f = attenuation_operator_f(promise, gamma)
    return apply_to_H(f,eigs)