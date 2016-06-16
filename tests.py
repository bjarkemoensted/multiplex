# -*- coding: utf-8 -*-

from __future__ import division

import mptools
import networkx as nx
import numpy as np
import unittest

#class TestStringMethods(unittest.TestCase):
#
#    def test_upper(self):
#        self.assertEqual('foo'.upper(), 'FOO')
#    def test_isupper(self):
#        self.assertTrue('FOO'.isupper())
#        self.assertFalse('Foo'.isupper())
#
#    def test_split(self):
#        s = 'hello world'
#        self.assertEqual(s.split(), ['hello', 'world'])
#        # check that s.split fails when the separator is not a string
#        with self.assertRaises(TypeError):
#            s.split(2)


class TestFinderThingy(unittest.TestCase):
    
    def test_failonshort(self):
        '''Make sure valueerror is raised when the list is too short'''
        self.assertRaises(ValueError, mptools._find_pos, [], 1)
        self.assertRaises(ValueError, mptools._find_pos, [1], 1)
    
    def findpos_brute(self, _list, target):
        '''Helper method to find the correct index position in O(n) time.
        Used to verify the output of a method doing it in O(logn).'''
        i = 0
        for i in xrange(len(_list)):
            if _list[i] > target:
                return i
            #
        raise ValueError('All list elements are larger than target')
        
    def test_find(self):
        for _ in xrange(10**3):
            l = sorted([np.random.uniform() for _ in xrange(100)])
            target = np.random.uniform(0, max(l))
            msg = "Failed with: target = "+str(target)+", list = "+str(l)
            try:
                correct = self.findpos_brute(l, target)
            except ValueError:
                self.assertRaises(ValueError, mptools._find_pos, l, target)
                continue
            
            try:
                self.assertEqual(mptools._find_pos(l, target), correct, msg=msg)
            except:
                print l
                raise

class Testunroll_adjacency_matrix(unittest.TestCase):
    '''Tests if the unroller method correctly extracts an adjacency matrix
    from a graph object and unrolls it into a list of its off-diagonal
    elements (only the northeastern matrix corner if the graph is
    unidirectional)'''
    
    def npunroller(self, G):
        '''Performs unrolling using a simple numpy matrix. This should be
        correct and slow :)'''
        
        result = []
        M = nx.to_numpy_matrix(G)
        rows, cols = M.shape
        for i in xrange(rows):
            for j in xrange(i + 1, cols):
                result.append(M[i, j])
            #
        return np.array(result)

    def test_unroll_adjacency_matrix(self):
        n_nodes = [2, 3, 5, 10, 42, 60]
        for n in n_nodes:
            G = nx.barabasi_albert_graph(n, n//2)            
            correct = self.npunroller(G)
            test = mptools.unroll_adjacency_matrix(G)
            test_dense = test.toarray()
            msg = "Correct: "+str(test_dense)
            msg += "\n Test result: "+str(test_dense)
            self.assertTrue(np.array_equiv(correct, test_dense), msg = msg)
    

if __name__ == '__main__':
    unittest.main()