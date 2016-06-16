# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 14:40:13 2016

@author: bjarke

Set of tools for using PCA to analyze multiplex networks
"""

from __future__ import division

import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import operator
import scipy as sp
import sys

def _find_pos(_list, target):
    '''Returns the index of the first element in _list which is larger than
    target.'''

    if len(_list) < 2:
        raise ValueError("Can't find position - list too short")
    
    if _list[-1] <= target:
        raise ValueError("All values are larger than target")
    
    if _list[0] >= target:
        return 0
    
    lower = 1
    upper = len(_list) - 1
    found = False    
    while not found:
        ind = int(0.5*(upper + lower))
        
        if _list[ind-1] < target <= _list[ind]:
            found = True
        elif _list[ind-1] >= target:
            upper = ind-1
        elif target > _list[ind]:
            lower = ind + 1
        else:
            raise ValueError
        #
    return ind
        
def select_nonuniform(_list, cdf):
    '''Randomly selects an element of _list using cumulative distribution
    cdf, i.e. if cdf = [0.3, 0.7, 1.0] the three corresponding elements
    are selected with probabilities 0.3, 0.4 and 0.3, respectively.'''
    
    target = np.random.uniform()
    try:
        i = _find_pos(cdf, target)
    except ValueError:
        print cdf, target
        raise
    return _list[i]

def get_random_graph(n, directional = False):
    '''Generates a random graph in which every possible link does or doesn't
    exist with p = 50%.'''
    
    if directional:
        A = np.random.rand(n, n).round()
        for i in xrange(n):
            A[i, i] = 0
    else:
        A = np.zeros(shape = (n, n))
        for i in xrange(n):
            for j in xrange(i+1, n):
                if np.random.uniform() >= 0.5:
                    A[i,j] = 1
                    A[j,i] = 1
                #
            #
        #
    G = nx.from_numpy_matrix(A)
    return G

def rewire_ba(G, n = 1):
    '''Rewires the input graph n times in the following fashion: 
    A random link (u, v) is selected and removed. u is then connected to a 
    random node that a) is not and b) is not already connected to u.
    The probability of selecting a given node as the endpoint is proportional
    to its degree, as per the Barabasi-Albert algorithm for growing networks
    with preferential attachment.'''
    
    nodelist = G.nodes()
    np.random.shuffle(nodelist)
    for _ in xrange(n):
        # Select and remove a random edge
        idx = np.random.choice(len(G.edges()))
        u, v = G.edges()[idx]
        G.remove_edge(u, v)
        
        # Rewire to random other node weighted by their degrees
        candidates = set(G.nodes_iter()) - set([u]) - set(G.neighbors_iter(u))
        try:
            norm = 1.0/reduce(operator.add, itertools.imap(G.degree, candidates))
            cdf = np.cumsum(map(lambda n: G.degree(n)*norm, candidates))
            endnode = select_nonuniform(list(candidates), cdf)
        except ZeroDivisionError:
            # All candidates have degree zero Just select a random one
            endnode = np.random.choice(list(candidates))
        G.add_edge(u, endnode)

#def progress(G, rewire_method, rewire_args, n_steps, picname):
#    does n step progressions
#    optionally saves pics
#    returns np.matrix

def unroll_adjacency_matrix(G):
    '''"Unrolls" the adjacency matrix of the input graph into a vector. This
    is done by extracting all off-diagonal elements of the nxn adjacency matrix
    and concatenating them into an n(n - 1)/2 dimensional array.
    Example:
    [[0, 1, 0],
     [1, 0, 1],
     [0, 1, 0]]
     gives [1, 0, 1].'''

    # Number of nodes in the graph
    n = len(G)
    # Length of the unrolled matrix
    dim = n*(n - 1)//2
    # Sparse matrix to hold the results
    result = sp.sparse.lil_matrix((1, dim))
    # Adjacency matrix for the graph
    M = nx.to_scipy_sparse_matrix(G, format = "coo")

    for i,j,v in zip(M.row, M.col, M.data):
        # Only care about northeastern corner of the matrix
        if not j > i:
            continue
        ind = i*n - (i*(i+1))//2 + j - i - 1 # Nothing to see here, move along.
        # Add the encountered element at the appropriate index of result
        result[0, ind] = v
    
    return result

def evolve_graph(G, method, argdict, n_steps, picdir = None, pos = None):
    if picdir and pos == None:
        pos = nx.spring_layout(G)
    
    
    M = unroll_adjacency_matrix(G)
    for i in xrange(n_steps):
        method(**argdict)
        if picdir:
            nx.draw(G, pos = pos)
            filename = picdir+str(i).zfill(len(str(n_steps)))+".png"
            plt.savefig(filename);
            plt.clf();
        row = unroll_adjacency_matrix(G)
        M = sp.sparse.vstack((M, row))
    
    #
    return M

if __name__ == "__main__":
    G = nx.barabasi_albert_graph(10,2)
    M = evolve_graph(G, rewire_ba, argdict = {'G' : G}, n_steps = 5)
    print M.todense()