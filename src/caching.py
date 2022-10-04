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

from inspect import signature
from multiprocessing import Process, JoinableQueue
import numpy as np
import json
import time
import os.path
from datetime import timedelta


# A fairly overengineered tool for:
#  - saving the results of a function into a cache file
#    and retriving it whenever possible,
#  - evaluating the function with several values in a single
#    call by passing lists as arguments,
#  - parallelizing evaluation over the values that are not
#    in the cache,
#  - truncating the precision of all floating point arguments
#    to avoid evaluating similar input values twice.
#  - print timing information

# Functions of interest:
#   - collect_data(*args,fn=None,filename=None,nthreads=1,prec=3,timing=True)
#   - print_available_data(filename,fn)


def collect_data(*args,fn=None,filename=None,nthreads=1,prec=3,timing=True):
    if fn is None: raise ValueError("Provide a function.")
    if filename is None: raise ValueError("Provide a filename.")
    
    # Extract function type signature and make sure
    # all the values are serializable.
    sig = signature(fn).parameters
    params = []
    for param in sig.keys():
        annot = sig[param].annotation
        if annot not in [int,float,str]:
            raise ValueError("Function",fn.__name__,"argument",
                            param,"is not of type int, float, or str.")
        params.append((param,annot))
    

    # assemble a list of tuples for the arguments
    keys = [()]
    for arg,(param,annot)in zip(args,params):
        # make it a list of values either way
        try:
            vals = list(arg)
        except TypeError:
            vals = [arg]                

        for i in range(len(vals)):
            # try to cast each value to the desired type
            try:
                if annot == str: vals[i] = str(vals[i])
                if annot == float:
                    # truncate precision
                    vals[i] = np.round(float(vals[i]),prec)
                if annot == int:
                    vals[i] = int(vals[i])
            except TypeError:
                raise TypeError("Failed to cast argument",
                                param,"with value",vals[i],"to",annot,".")

        keys = [key + (val,) for key in keys for val in vals]


    # Obtain values that are already in the cache
    output_values = retrieve_from_cache(filename, fn.__name__, keys)

    # Keys that we actually need to evaluate:
    compute_keys = [key for key in output_values.keys() if output_values[key] is None]

    # Actually evaluate the function.
    if len(compute_keys) == 1:
        # Just one case to evaluate? Just call the function.
        t0 = time.time()
        print("Evaluating:",compute_keys,"at",t0,flush=True)
        out = fn(*compute_keys[0])
        t1 = time.time()
        if timing: print("Finished",compute_keys,"at",t1,"with elapsed time:", timedelta(milliseconds=(t1 - t0)),flush=True)
        insert_into_cache(filename, fn.__name__, compute_keys[0], out)
        output_values[compute_keys[0]] = out     

    elif nthreads == 1:
        # Only one thread? Just do it on the main thread and
        # don't even use multiprocessing.Pool.
        for key in compute_keys:
            print("Evaluating:",key)
            t0 = time.time()
            out = fn(*key)
            if timing: print("Elapsed time:", timedelta(milliseconds=(time.time() - t0)))
            insert_into_cache(filename, fn.__name__, key, out)
            output_values[compute_keys[0]] = out     
    else:
        # Multiple threads.

        input_queue, output_queue = JoinableQueue(), JoinableQueue()
        
        # Populate queue with tasks
        for task in compute_keys:
            input_queue.put(task)

        # Spawn threads to process tasks
        workers = []
        for _ in range(nthreads):
            p = Process(target=_worker, args=(fn, input_queue, output_queue))
            p.start()
            workers.append(p)

        # Save data as it is sent back.
        for _ in compute_keys:
            key,out,dt = output_queue.get()
            insert_into_cache(filename, fn.__name__, key, out)
            if timing: print(key,"elapsed time:", dt)
            output_values[key] = out     
            output_queue.task_done()
        
        input_queue.join()
        output_queue.join()
        for p in workers: p.join()
    
    return output_values

def _worker(fn, input_queue, output_queue):
    while True:
        if input_queue.empty(): return 
        task = input_queue.get()
        print("Evaluating:",task)
        t0 = time.time()
        out = fn(*task)
        output_queue.put((task,out, timedelta(milliseconds=(time.time() - t0))))
        input_queue.task_done()

##############################
# Cache insertion and deletion

def retrieve_from_cache(filename,funcname,keys):

    out = {key:None for key in keys}

    if not os.path.exists(filename):
        # File doesn't exist. Don't create it, leave that to insert_into_cache.
        return out

    lookup = {json.dumps({"func":funcname, "args":key}):key for key in keys}
    
    with open(filename,"r") as f:
        for l in f.readlines():
            l = json.loads(l.strip())
            if l["key"] in lookup:
                key = lookup[l["key"]]
                if out[key] is not None:
                    raise ValueError("Key",key,"is duplicate in file",
                                     filename,"for function",funcname)
                out[key] = l["value"]

    return out

def insert_into_cache(filename,funcname,key,value):
    with open(filename,"a") as f:
        f.write(json.dumps({
                "key": json.dumps({"func":funcname, "args":key}), 
                "value": value})+"\n")


def print_available_data(filename,fn):
    if not os.path.exists(filename): return print("No such file:",filename)

    # Get list of function argument names
    params = list(signature(fn).parameters.keys())

    # Collect all matching entries from file
    keys = []
    with open(filename,"r") as f:
        for l in f.readlines():
            l = json.loads(l.strip())
            key = json.loads(l["key"])
            if key["func"] == fn.__name__:
                keys.append({p:k for p,k in zip(params,key["args"])})

    # Factor the entries.
    for i in range(len(params)):
        done_keys = []
        remaining_keys = keys
        
        while len(remaining_keys) > 0:
            pivot_key = remaining_keys.pop()
            pivot_key[params[i]] = [pivot_key[params[i]]]
            new_remaining_keys = []
            for key in remaining_keys:
                good = True
                for j in range(len(params)):
                    if i == j: continue
                    if json.dumps(key[params[j]]) != json.dumps(pivot_key[params[j]]):
                        good = False
                        break
                if good: pivot_key[params[i]].append(key[params[i]])
                else: new_remaining_keys.append(key)
            remaining_keys = new_remaining_keys

            pivot_key[params[i]] = list(sorted(pivot_key[params[i]]))
            done_keys.append(pivot_key)
        
        keys = done_keys

    # Print the resulting factored entries.
    for key in keys: print(key)


def subroutine(x):
    return x+1

def test_function(x: float, y: str):
    time.sleep(np.random.random()*3)
    return subroutine(x), y+"1"


if __name__ == "__main__":
    collect_data([9],[12],fn=test_function,filename="test.json",nthreads=3)
    print_available_data("test.json",test_function)

