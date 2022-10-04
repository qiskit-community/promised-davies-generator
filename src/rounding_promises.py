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

# Construct some rounding promises that are of interest to this analysis.
def get_rounding_promise(n,r,side,j):
    assert side in ["left","right"]
    assert int(j) == j and 0 <= j < 2**r
        
    # Fine rounding promise construction: 
    # Perform energy estimation to n+r bits of precision, and only
    # consider the least significant bit. Amplify the result and
    # measure it. The resulting state's energies are now guaranteed
    # to be supported on either the 'left' or 'right' rounding promise
    # with idx=None.
    
    fine_promise = []
    pos = 0
    dx = 2**(-n-r-2)
    for i in range(2**(n+r)):
        if side == 'left': fine_promise.append([max(0,pos-dx), pos+2*dx])
        else: fine_promise.append([pos+dx, pos+4*dx])
        pos += 4*dx
    if side == 'left': fine_promise.append([1-dx,1])
    
    if j is None: return  fine_promise

    
    # Coarse rounding promise construction:
    # We merge all gaps between the intervals those = idx (mod 2**r).
    # These coarse rps have the property that for every energy, there is
    # at most one idx such that the energy is not in the idx'th promise.
    
    out = []
    current = [0,0]
    for i, interval in enumerate(fine_promise):
        if (i-1) % 2**r == j:
            out.append(current)
            current = [interval[0],interval[1]]
        else:
            current = [current[0],interval[1]]
    out.append(current)
    if out[0][1] == 0: out = out[1:] 
    return out

# Returns midpoint of connected component if eig is in the promise
# and returns None otherwise.
def rounded_eig(eig, promise):
    for interval in promise:
        # The interval is closed on the left and open on the right.
        # This only matters for the ideal_rp rounding promise.
        if interval[0] <= eig and eig < interval[1]:
            return np.mean(interval)
    return None

# Given a promise, returns a dictionary where
# omegas are keys, and values are a list of pairs of indexes.
def make_calJ(promise):
    calJ = {}
    for x in range(len(promise)):
        for y in range(len(promise)):
            omega = np.mean(promise[x])-np.mean(promise[y])
            if omega not in calJ: calJ[omega] = []
            calJ[omega].append((x,y))
    return calJ

# An isometry from the promised subspace into the full hilbert space
def promise_isometry(eigs, promise):
    idxs = [idx for idx,eig in enumerate(eigs)
            if rounded_eig(eig,promise) is not None]

    out = np.zeros((len(eigs),len(idxs)))
    for i,idx in enumerate(idxs):
        out[idx,i] = 1

    return out


def get_thermal_state(eigs, beta):
    exps = [np.exp(-eig*beta) for eig in eigs]
    return np.diag(exps)/sum(exps)


# returns a state on the promised subspace
def get_promised_thermal_state(eigs, promise, beta):
    rounded_eigs = [rounded_eig(eig, promise) for eig in eigs
                    if rounded_eig(eig,promise) is not None]
    
    return get_thermal_state(rounded_eigs, beta)

    
