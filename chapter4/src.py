#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions and classes used to generate results for Chapter 4, "Predicting schedule length under uncertainty". 
"""

import networkx as nx
import numpy as np
from math import sqrt
from psutil import virtual_memory
from statistics import NormalDist
from networkx.utils import pairwise
from itertools import islice
from sys import getsizeof

class RV:
    """
    Random variable class.
    Defined only by mean and variance so can in theory be from any distribution but e.g., addition and multiplication assume
    RV is Gaussian. 
    """
    def __init__(self, mu=0.0, var=0.0): 
        self.mu = mu
        self.var = var
        self.sd = sqrt(var)
    def __repr__(self):
        return "RV(mu = {}, var = {})".format(self.mu, self.var)
    # Overload addition operator.
    def __add__(self, other): 
        if isinstance(other, float) or isinstance(other, int):
            return RV(self.mu + other, self.var)
        return RV(self.mu + other.mu, self.var + other.var) 
    __radd__ = __add__ 
    # Overload subtraction operator.
    def __sub__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return RV(self.mu - other, self.var)
        return RV(self.mu - other.mu, self.var + other.var)
    __rsub__ = __sub__ 
    # Overload multiplication operator.
    def __mul__(self, c):
        return RV(c * self.mu, c * c * self.var)
    __rmul__ = __mul__ 
    # Overload division operators.
    def __truediv__(self, c): 
        return RV(self.mu / c, self.var / (c * c))
    __rtruediv__ = __truediv__ 
    def __floordiv__(self, c): 
        return RV(self.mu / c, self.var / (c * c))
    __rfloordiv__ = __floordiv__ 
    
class StochDAG:
    """Represents a graph with stochastic node and edge weights."""
    def __init__(self, graph):
        """
        Initialize with topology. 

        Parameters
        ----------
        graph : Networkx DiGraph with RV node and edge weights
            DAG topology and weights.

        Returns
        -------
        None.
        """
        self.graph = graph
        self.top_sort = list(nx.topological_sort(self.graph))    # Often saves time.  
        self.size = len(self.top_sort)
        
    def set_random_weights(self, mu_cov):
        """
        Set random weights for STG DAGs.

        Parameters
        ----------
        mu_cov : FLOAT
            Mean coefficient of variation.
        
        Notes
        -------
        1. mu_cov is mean coefficient of variation of the weights but standard deviation always equal 0.1*mu_cov.

        Returns
        -------
        None.

        """
        
        task_means = np.random.uniform(1, 100, self.size)
        edge_means = np.random.uniform(1, 100, self.graph.number_of_edges())
        
        task_covs = np.random.gamma(100, mu_cov*0.01, self.size) # standard deviation always equal 0.1*mu_cov.
        edge_covs = np.random.gamma(100, mu_cov*0.01, self.graph.number_of_edges())
                                
        # Task weights.
        for i, t in enumerate(self.top_sort):
            mu = task_means[i]
            sd = task_covs[i] * mu
            self.graph.nodes[t]['weight'] = RV(mu, sd**2)
            
        # Edge weights.
        for i, e in enumerate(self.graph.edges):
            mu = edge_means[i]
            sd = edge_covs[i] * mu
            self.graph[e[0]][e[1]]['weight'] = RV(mu, sd**2)        
       
    def number_of_paths(self):
        """
        Count the number of paths through DAG.
        (Typically only used to show how enormous and impractical it is.)

        Returns
        -------
        INT
            Number of distinct paths through DAG.

        """      
        paths = {}
        for t in self.top_sort:
            parents = list(self.graph.predecessors(t))
            if not parents:
                paths[t] = 1
            else:
                paths[t] = sum(paths[p] for p in parents)                
        return paths[self.top_sort[-1]]  
    
    def get_scalar_graph(self, scal_func=lambda r : r.mu, aoa=True, johnson=False):
        """
        Return a ScaDAG object by applying scal_func to all the weights.

        Parameters
        ----------
        scal_func : LAMBDA, optional
            The scalarization function to apply to weight RVs. The default is lambda r : r.mu.
        aoa : BOOL, optional
            Activity-on-arc - i.e., task weights are incorporated into edge weights. This is useful for exploiting some Networkx functions.
            The default is True.
        johnson : BOOL, optional
            Transform scalar graph as in Johnson's algorithm to convert longest to shortest path function. Again, useful for compatability
            with some Networkx functions. The default is False.

        Returns
        -------
        ScaDAG
            Graph with scalar weights.

        """
        
        # Copy the topology.
        A = self.graph.__class__()
        A.add_nodes_from(self.graph)
        A.add_edges_from(self.graph.edges)
        
        # Set the weights.
        if aoa:
            for t, s in self.graph.edges:
                A[t][s]['weight'] = scal_func(self.graph.nodes[t]['weight']) 
                if s == self.top_sort[-1]:
                    A[t][s]['weight'] += scal_func(self.graph.nodes[s]['weight']) 
                try:
                    A[t][s]['weight'] += scal_func(self.graph[t][s]['weight'])  
                except AttributeError: 
                    pass
            if johnson: # Rewrite the weights to convert longest to shortest path problem.
                H = A.__class__()
                H.add_nodes_from(A)
                H.add_edges_from(A.edges)
                for t, s in A.edges:
                    H[t][s]['weight'] = -A[t][s]['weight']
                # Add q node.
                for t in A.nodes:
                    H.add_edge("q", t)
                    H["q"][t]['weight'] = 0.0
                h, _ = nx.single_source_bellman_ford(H, "q")                
                for t, s in A.edges:
                    A[t][s]['weight'] = -A[t][s]['weight'] + h[t] - h[s] 
        else:
            for t in self.top_sort:
                A.nodes[t]['weight'] = scal_func(self.graph.nodes[t]['weight']) 
                for s in self.graph.successors(t):
                    try:
                        A[t][s]['weight'] = scal_func(self.graph[t][s]['weight']) 
                    except AttributeError: # TODO: correct error?
                        A[t][s]['weight'] = 0.0
                
        # Return ScaDAG object.      
        return ScaDAG(A, aoa, johnson)
            
    def CPM(self, variance=False, full=False):
        """
        Classic PERT-CPM bound on the expected value of the longest path.

        Parameters
        ----------
        variance : BOOL, optional
            If True, also returns the variance of the path with maximal expected value. The default is False.
        full : BOOL, optional
            If True, return CPM bounds for all tasks not just the exit task (i.e., through the whole DAG). The default is False.

        Returns
        -------
        FLOAT
            CPM lower bound on the expected value.

        """
        C = {}       
        for t in self.top_sort:
            st = 0.0
            if variance:
                v = 0.0
            for p in self.graph.predecessors(t):
                pst = C[p] if not variance else C[p].mu
                try:
                    pst += self.graph[p][t]['weight'].mu
                except AttributeError: # Disjunctive edge.
                    pass
                st = max(st, pst)  
                if variance and st == pst:
                    v = C[p].var
                    try:
                        v += self.graph[p][t]['weight'].var
                    except AttributeError:
                        pass
            m = self.graph.nodes[t]['weight'].mu
            if not variance:
                C[t] = m + st      
            else:
                var = self.graph.nodes[t]['weight'].var
                C[t] = RV(m + st, var + v)   
        if full:
            return C                                 
        return C[self.top_sort[-1]]   
    
    def kamburowski(self):
        """
        Kamburowski's bounds on the mean and variance.

        Returns
        -------
        FLOAT
            Lower bound on the mean.
        FLOAT
            Upper bound on the mean.
        FLOAT
            Lower bound on the variance.
        FLOAT
            Upper bound on the variance.

        """
        lm, um, ls, us = {},{}, {}, {}
        for t in self.top_sort:
            nw = self.graph.nodes[t]['weight']
            parents = list(self.graph.predecessors(t))
            # Entry task(s).
            if not parents:
                lm[t], um[t] = nw.mu, nw.mu
                ls[t], us[t] = nw.var, nw.var
                continue
            # Lower bound on variance.
            if len(parents) == 1:
                ls[t] = ls[parents[0]] + nw.var
                try:
                    ls[t] += self.graph[parents[0]][t]['weight'].var
                except AttributeError:
                    pass
            else:
                ls[t] = 0.0
            # Upper bound on variance.
            v = 0.0
            for p in parents:
                sv = us[p] + nw.var
                try:
                    sv += self.graph[p][t]['weight'].var
                except AttributeError:
                    pass
                v = max(v, sv)
            us[t] = v
            # Lower bound on mean.
            Xunder = []
            for p in parents:
                pmu = lm[p] + nw.mu
                pvar = ls[p] + nw.var
                try:
                    pmu += self.graph[p][t]['weight'].mu
                    pvar += self.graph[p][t]['weight'].var
                except AttributeError:
                    pass
                Xunder.append(RV(pmu, pvar))
            Xunder = list(sorted(Xunder, key=lambda x:x.var))
            lm[t] = funder(Xunder)
            # Upper bound on mean.
            Xover = []
            for p in parents:
                pmu = um[p] + nw.mu
                pvar = us[p] + nw.var
                try:
                    pmu += self.graph[p][t]['weight'].mu
                    pvar += self.graph[p][t]['weight'].var
                except AttributeError:
                    pass
                Xover.append(RV(pmu, pvar))
            Xover = list(sorted(Xover, key=lambda x:x.var))
            um[t] = fover(Xover)        
        return lm[self.top_sort[-1]], um[self.top_sort[-1]], ls[self.top_sort[-1]], us[self.top_sort[-1]]      

    def sculli(self, reverse=False):
        """
        Sculli's method for estimating the longest path distribution for a DAG with stochastic weights.
        'The completion time of PERT networks,'
        Sculli (1983).  

        Parameters
        ----------
        reverse : BOOL, optional
            If True, reverses the DAG before applying method. The default is False.

        Returns
        -------
        RV
            The approximated longest path distribution (i.e., its mean and variance).

        """
        
        if reverse:
            return StochDAG(self.graph.reverse()).sculli() 
        
        L = {}
        for t in self.top_sort:
            parents = list(self.graph.predecessors(t))
            try:
                p = parents[0]
                m = self.graph[p][t]['weight'] + L[p] 
                for p in parents[1:]:
                    m1 = self.graph[p][t]['weight'] + L[p]
                    m = clark(m, m1, rho=0)
                L[t] = m + self.graph.nodes[t]['weight']
            except IndexError:  # Entry task.
                L[t] = self.graph.nodes[t]['weight']  
        return L[self.top_sort[-1]]            
    
    def corLCA(self, reverse=False):
        """
        CorLCA heuristic for estimating the longest path distribution for a DAG with stochastic weights.
        'Correlation-aware heuristics for evaluating the distribution of the longest path length of a DAG with random weights,' 
        Canon and Jeannot (2016). 

        Parameters
        ----------
        reverse : BOOL, optional
            If True, reverses the DAG before applying method. The default is False.
        
        Notes
        -------
        1. Assume single source and sink.
        2. Orginally explicitly constructed correlation tree as a DiGraph but this version is faster.
        3. Dominant parents identified by comparing expected values.

        Returns
        -------
        RV
            The approximated longest path distribution (i.e., its mean and variance).

        """  
        
        if reverse:
            return StochDAG(self.graph.reverse()).corLCA() 
        
        # L represents longest path estimates. V[task ID] = variance of longest path of dominant ancestors (used to estimate rho).
        L, V, dominant_ancestors = {}, {}, {}
        for t in self.top_sort:     # Traverse the DAG in topological order. 
            nw = self.graph.nodes[t]['weight']
            dom_parent = None 
            for parent in self.graph.predecessors(t):
                pst = self.graph[parent][t]['weight'] + L[parent]   
                                    
                # First parent.
                if dom_parent is None:
                    dom_parent = parent 
                    dom_parent_ancs = set(dominant_ancestors[dom_parent])
                    dom_parent_sd = V[dom_parent]
                    try:
                        dom_parent_sd += self.graph[dom_parent][t]['weight'].var
                    except AttributeError:
                        pass
                    dom_parent_sd = sqrt(dom_parent_sd) 
                    st = pst
                    
                # At least two parents, so need to use Clark's equations to compute eta.
                else:                    
                    # Find the lowest common ancestor of the dominant parent and the current parent.
                    for a in reversed(dominant_ancestors[parent]):
                        if a in dom_parent_ancs:
                            lca = a
                            break
                        
                    # Estimate the relevant correlation.
                    parent_sd = V[parent]
                    try:
                        parent_sd += self.graph[parent][t]['weight'].var
                    except AttributeError:
                        pass
                    parent_sd = sqrt(parent_sd) 
                    r = V[lca] / (dom_parent_sd * parent_sd)
                        
                    # Find dominant parent for the maximization (by comparing expected values).
                    if pst.mu > st.mu: 
                        dom_parent = parent
                        dom_parent_ancs = set(dominant_ancestors[parent])
                        dom_parent_sd = parent_sd
                    
                    # Compute eta.
                    st = clark(st, pst, rho=r)  
            
            if dom_parent is None: # Entry task...
                L[t] = nw  
                V[t] = nw.var
                dominant_ancestors[t] = [t]
            else:
                L[t] = nw + st 
                V[t] = dom_parent_sd**2 + nw.var
                dominant_ancestors[t] = dominant_ancestors[dom_parent] + [t] 
                          
        return L[self.top_sort[-1]]   
    
    def monte_carlo(self, samples, dist="NORMAL"):
        """
        Monte Carlo method to estimate the distribution of the longest path. 

        Parameters
        ----------
        samples : INT
            Number of realizations of entire graph to do.
        dist : STRING, optional
            The probability distribution that the weights follow (assumed to be the same for all). The default is "NORMAL".
            
        Notes
        -------
        1. Take absolute values for normal and uniform distributions, although rarely needed because of weight coefficients of variation.
        2. Can hit memory wall for large graphs and/or numbers of samples, so need to call function repeatedly if that seems likely.  

        Returns
        -------
        LIST/NUMPY ARRAY
            The empirical longest path distribution.

        """  
        mem_limit = virtual_memory().available // 10 # Just a guideline...
        if self.size*samples < mem_limit:   
            L = {}
            for t in self.top_sort:
                m = self.graph.nodes[t]['weight'].mu
                if dist in ["N", "n", "NORMAL", "normal"]:  
                    # w = np.random.normal(m, self.graph.nodes[t]['weight'].sd, samples)
                    w = abs(np.random.normal(m, self.graph.nodes[t]['weight'].sd, samples))
                elif dist in ["G", "g", "GAMMA", "gamma"]:
                    v = self.graph.nodes[t]['weight'].var
                    sh, sc = (m * m)/v, v/m
                    w = np.random.gamma(sh, sc, samples)
                elif dist in ["U", "u", "UNIFORM", "uniform"]:
                    u = sqrt(3) * self.graph.nodes[t]['weight'].sd
                    # w = np.random.uniform(-u + m, u + m, samples) 
                    w = abs(np.random.uniform(-u + m, u + m, samples))
                parents = list(self.graph.predecessors(t))
                if not parents:
                    L[t] = w 
                    continue
                pmatrix = []
                for p in parents:
                    try:
                        m = self.graph[p][t]['weight'].mu
                        if dist in ["N", "n", "NORMAL", "normal"]: 
                            # e = np.random.normal(m, self.graph[p][t]['weight'].sd, samples)
                            e = abs(np.random.normal(m, self.graph[p][t]['weight'].sd, samples))
                        elif dist in ["G", "g", "GAMMA", "gamma"]:
                            v = self.graph[p][t]['weight'].var
                            sh, sc = (m * m)/v, v/m 
                            e = np.random.gamma(sh, sc, samples)
                        elif dist in ["U", "u", "UNIFORM", "uniform"]:
                            u = sqrt(3) * self.graph[p][t]['weight'].sd
                            # e = np.random.uniform(-u + m, u + m, samples)  
                            e = abs(np.random.uniform(-u + m, u + m, samples))  
                        pmatrix.append(np.add(L[p], e))
                    except AttributeError:
                        pmatrix.append(L[p])
                st = np.amax(pmatrix, axis=0)
                L[t] = np.add(w, st)
            return L[self.top_sort[-1]] 
        else:
            E = []
            mx_samples = mem_limit//self.size
            runs = samples//mx_samples
            extra = samples % mx_samples
            for _ in range(runs):
                E = np.concatenate((E, self.monte_carlo(samples=mx_samples, dist=dist)))
            E = np.concatenate((E, self.monte_carlo(samples=extra, dist=dist)))
            return E 
        
    def monte_carlo_paths(self, samples, dist="U"):
        """
        Use Monte Carlo method to identify critical paths. 

        Parameters
        ----------
        samples : INT
            Number of realizations of entire graph to do.
        dist : STRING, optional
            The probability distribution that the weights follow (assumed to be the same for all). The default is "U".
            
        Notes
        -------
        1. Take absolute values for normal and uniform distributions, although rarely needed because of weight coefficients of variation.
        2. Can hit memory wall for large graphs and/or numbers of samples, so need to call function repeatedly if that seems likely. 
        3. Still not sure why original critical_paths version (commented out) didn't work.

        Returns
        -------
        Q : DICT
            The observed critical paths and their frequencies.
        E : LIST/NUMPY ARRAY
            The empirical MC longest path distribution.
        """      
              
        mem_limit = virtual_memory().available // 10 
        if self.size*samples < mem_limit:   
            L = {}
            critical_parents = {}
            # critical_paths = {}
            for t in self.top_sort:
                m = self.graph.nodes[t]['weight'].mu
                if dist in ["N", "n", "NORMAL", "normal"]:  
                    w = abs(np.random.normal(m, self.graph.nodes[t]['weight'].sd, samples))
                elif dist in ["G", "g", "GAMMA", "gamma"]:
                    v = self.graph.nodes[t]['weight'].var
                    sh, sc = (m * m)/v, v/m
                    w = np.random.gamma(sh, sc, samples)
                elif dist in ["U", "u", "UNIFORM", "uniform"]:
                    u = sqrt(3) * self.graph.nodes[t]['weight'].sd
                    w = abs(np.random.uniform(-u + m, u + m, samples))
                parents = list(self.graph.predecessors(t))
                if not parents:
                    L[t] = w 
                    # critical_paths[t] = [(t,) for _ in range(samples)]
                    continue
                pmatrix = []
                for p in parents:
                    try:
                        m = self.graph[p][t]['weight'].mu
                        if dist in ["N", "n", "NORMAL", "normal"]: 
                            e = abs(np.random.normal(m, self.graph[p][t]['weight'].sd, samples))
                        elif dist in ["G", "g", "GAMMA", "gamma"]:
                            v = self.graph[p][t]['weight'].var
                            sh, sc = (m * m)/v, v/m 
                            e = np.random.gamma(sh, sc, samples)
                        elif dist in ["U", "u", "UNIFORM", "uniform"]:
                            u = sqrt(3) * self.graph[p][t]['weight'].sd
                            e = abs(np.random.uniform(-u + m, u + m, samples))
                        pmatrix.append(np.add(L[p], e))
                    except AttributeError:
                        pmatrix.append(L[p]) 
                # Path lengths. 
                st = np.amax(pmatrix, axis=0)
                L[t] = np.add(w, st)
                # Identify critical parents.
                critical_parents[t] = [parents[j] for j in np.argmax(pmatrix, axis=0)]
                # critical_paths[t] = [critical_paths[parents[j]][i] + (t,) for i, j in enumerate(np.argmax(pmatrix, axis=0))]                 
            # Return paths and empirical distribution.
            paths = {}
            for i, p in enumerate(critical_parents[self.top_sort[-1]]):
                path = (self.top_sort[-1], p)
                nxt = p
                while True:
                    try:
                        nxt = critical_parents[nxt][i]
                        path += (nxt,)
                    except KeyError:
                        break
                path = tuple(reversed(path))
                try:
                    paths[path] += 1
                except KeyError:
                    paths[path] = 1
            return paths, L[self.top_sort[-1]]
        else:
            Q, E = {}, []
            mx_samples = mem_limit//self.size
            runs = samples//mx_samples
            extra = samples % mx_samples
            for _ in range(runs):
                P, D = self.monte_carlo_paths(samples=mx_samples, dist=dist)
                E = np.concatenate((E, D))
                for delta, v in P.items():
                    try:
                        Q[delta] += v
                    except KeyError:
                        Q[delta] = v                        
            P, D = self.monte_carlo_paths(samples=extra, dist=dist)
            for delta, v in P.items():
                try:
                    Q[delta] += v
                except KeyError:
                    Q[delta] = v
            E = np.concatenate((E, D))
            return Q, E
    
    def monte_carlo_paths_with_max(self, path_samples, lp_samples, dist="U"):
        """
        Use Monte Carlo method to identify critical paths and then approximates their max. Not used anywhere. 

        Parameters
        ----------
        path_samples : INT
            Number of realizations of entire graph to do.
        lp_samples : INT
            Number of samples to take when approximating the max.
        dist : STRING, optional
            The probability distribution that the weights follow (assumed to be the same for all). The default is "U".
            
        Notes
        -------
        1. Only intended/works for small numbers of samples (e.g., hits memory wall for 100 samples with largest Cholesky DAG and nb = 1024).

        Returns
        -------
        LIST/NUMPY ARRAY
            Empirical distribution.
        LIST/NUMPY ARRAY
            The approximated maximum
        """   
        
        L, critical_paths, reals = {}, {}, {}
        
        # Do the traditional MC method with path_samples realizations to identify critical paths.
        for t in self.top_sort:
            m = self.graph.nodes[t]['weight'].mu
            if dist in ["N", "n", "NORMAL", "normal"]:  
                w = abs(np.random.normal(m, self.graph.nodes[t]['weight'].sd, path_samples))
            elif dist in ["G", "g", "GAMMA", "gamma"]:
                v = self.graph.nodes[t]['weight'].var
                sh, sc = (m * m)/v, v/m
                w = np.random.gamma(sh, sc, path_samples)
            elif dist in ["U", "u", "UNIFORM", "uniform"]:
                u = sqrt(3) * self.graph.nodes[t]['weight'].sd
                w = abs(np.random.uniform(-u + m, u + m, path_samples))
            reals[t] = w
            parents = list(self.graph.predecessors(t))
            if not parents:
                L[t] = w 
                critical_paths[t] = [(t,) for _ in range(path_samples)]
                continue
            pmatrix = []
            for p in parents:
                try:
                    m = self.graph[p][t]['weight'].mu
                    if dist in ["N", "n", "NORMAL", "normal"]: 
                        e = abs(np.random.normal(m, self.graph[p][t]['weight'].sd, path_samples))
                    elif dist in ["G", "g", "GAMMA", "gamma"]:
                        v = self.graph[p][t]['weight'].var
                        sh, sc = (m * m)/v, v/m 
                        e = np.random.gamma(sh, sc, path_samples)
                    elif dist in ["U", "u", "UNIFORM", "uniform"]:
                        u = sqrt(3) * self.graph[p][t]['weight'].sd
                        e = abs(np.random.uniform(-u + m, u + m, path_samples))
                    pmatrix.append(np.add(L[p], e))
                    reals[(p, t)] = e
                except AttributeError:
                    pmatrix.append(L[p]) 
            # Path lengths. 
            st = np.amax(pmatrix, axis=0)
            L[t] = np.add(w, st)
            # Identify critical parents.
            critical_paths[t] = [critical_paths[parents[j]][i] + (t,) for i, j in enumerate(np.argmax(pmatrix, axis=0))] 
                    
        # Approximate the maximum of the path lengths.
        paths = list(set(critical_paths[self.top_sort[-1]]))
        if len(paths) == 1: # Only one path observed to be critical.
            # Use empirical mean and sd to define distribution rather than theoretical values. 
            # (Effectively just extending the empirical distribution.)
            mu, sd = np.mean(L[self.top_sort[-1]]), np.std(L[self.top_sort[-1]])
            # Return empirical dist and approximated path maximum.
            return L[self.top_sort[-1]], np.random.normal(mu, sd, lp_samples)  
        # Get the realizations of each path length. 
        lengths, means = [], []
        for path in paths:
            length = sum(reals[t] for t in path) + sum(reals[e] for e in pairwise(path) if e in reals)
            means.append(np.mean(length))
            lengths.append(length)  
        # Estimate the covariance matrix. Use empirical covariance rather than analytical result used in e.g, path_max. 
        cov = np.cov(lengths) 
        # Generate realizations of path length RVs. 
        data = np.random.default_rng().multivariate_normal(means, cov, lp_samples) 
        # Return empirical dist and approximated path maximum.                           
        return L[self.top_sort[-1]], np.amax(data, axis=1)
            
    def path_length(self, path):
        """
        Estimate the length of a path using the CLT - i.e., sum the weight means and variances. Occasionally useful.

        Parameters
        ----------
        path : ITERABLE
            Iterable of task IDs.

        Returns
        -------
        RV
            The approximate length distribution of the path. 

        """
        
        task_weights = sum(self.graph.nodes[t]['weight'] for t in path)
        edge_weights = sum(self.graph[u][v]['weight'] for u,v in pairwise(path))
        return task_weights + edge_weights        
    
    def path_max(self, paths, samples=10, correlations=True):
        """
        Approximate the maximum of the paths. 

        Parameters
        ----------
        paths : ITERABLE
            The paths to be maximized.
        samples : INT, optional
            The number of realizations to perform. The default is 10.
        correlations : BOOL, optional
            Consider path correlations or not. The default is True.

        Returns
        -------
        LIST/NUMPY ARRAY
            The approximated empirical longest path distribution.

        """
        
        if len(paths) == 1: # Only one path observed to be critical.
            delta = paths[0]
            task_mu = sum(self.graph.nodes[w]['weight'].mu for w in delta)
            task_var = sum(self.graph.nodes[w]['weight'].var for w in delta)
            edge_mu = sum(getattr(self.graph[e[0]][e[1]]['weight'], 'mu', 0.0) for e in pairwise(delta)) 
            edge_var = sum(getattr(self.graph[e[0]][e[1]]['weight'], 'var', 0.0) for e in pairwise(delta))             
            return np.random.normal(task_mu + edge_mu, sqrt(task_var + edge_var), samples)  
                
        # First get the path lengths.
        lengths = {}       
        for delta in paths:
            task_mu = sum(self.graph.nodes[w]['weight'].mu for w in delta)
            task_var = sum(self.graph.nodes[w]['weight'].var for w in delta)
            edge_mu = sum(getattr(self.graph[e[0]][e[1]]['weight'], 'mu', 0.0) for e in pairwise(delta)) 
            edge_var = sum(getattr(self.graph[e[0]][e[1]]['weight'], 'var', 0.0) for e in pairwise(delta)) 
            lengths[delta] = RV(task_mu + edge_mu, task_var + edge_var)
            
        # If specified, compute correlations.
        means = list(lengths[delta].mu for delta in paths)
        if not correlations:
            sds = list(lengths[delta].sd for delta in paths)
            data = np.random.normal(means, sds, size=(samples, len(paths)))
            return np.amax(data, axis=1)
        # Compute the covariance matrix.
        cov = []
        for i, delta in enumerate(paths):
            row = []
            # Copy already computed covariances.
            row = [cov[j][i] for j in range(i)]
            # Add diagonal - just the variance.
            row.append(lengths[delta].var)
            # Compute covariance with other paths.
            for other in paths[i + 1:]: 
                common_tasks = sum(self.graph.nodes[t]['weight'].var for t in delta if t in other)
                common_edges = sum(getattr(self.graph[e[0]][e[1]]['weight'], 'var', 0.0) for e in pairwise(delta) if e in pairwise(other))
                row.append(common_tasks + common_edges)
            cov.append(row)  
        data = np.random.default_rng().multivariate_normal(means, cov, samples) 
        return np.amax(data, axis=1)
    
    def get_dominating_paths(self, p=5, limit=1000):
        """
        Get all paths with probability >= p% of being longer than the classic PERT/CPM longest path (i.e., with greatest mean).

        Parameters
        ----------
        p : FLOAT/INT. 
            p in [0, 50). Intuitively, a p value of x retains all paths with >= x% probability of exceeding the path with greatest mean.
        
        limit : INT:
            Maximum number of path candidates. If > limit, return empty set (i.e., algorithm fails).

        Returns
        -------
        The set of longest path candidates.
        
        """
        
        # Compute comparison for determining if path is retained.
        x = NormalDist().inv_cdf(p/100.0)
        
        # Compute length of paths from source to all tasks with maximal mean.  
        C = self.CPM(variance=True, full=True)
        
        Q, lengths = {}, {}        
        for t in self.top_sort:
            parents = list(self.graph.predecessors(t))
            # Identify all possible paths.
            if not parents:
                Q[t] = [(t,)]
                lengths[(t,)] = self.graph.nodes[t]['weight']
            else: 
                Q[t] = []
                comp_length = C[t]
                for p in parents:
                    edge_weight = self.graph.nodes[t]['weight'] + self.graph[p][t]['weight']
                    for delta in Q[p]:
                        length = lengths[delta] + edge_weight
                        y = (length.mu - comp_length.mu)/sqrt(length.var + comp_length.var)
                        # Test if path is dominated.
                        if y > x:
                            new_delta = delta + (t,)
                            Q[t].append(new_delta)
                            lengths[new_delta] = length
                if len(Q[t]) > limit:
                    return []
        return Q[self.top_sort[-1]]
    
    def get_kdominant_paths(self, k=1000):
        """
        Alternative to previous function that retains the K most dominant paths (i.e., even if they're unlikely to beat the CPM path).

        Parameters
        ----------
        k : int. 
            Number of most dominant paths to retain.

        Returns
        -------
        The set of longest path candidates.
        
        """
                
        # Compute length of paths from source to all tasks with maximal mean.  
        C = self.CPM(variance=True, full=True)
        
        Q, lengths = {}, {}        
        for t in self.top_sort:
            parents = list(self.graph.predecessors(t))
            # Identify all possible paths.
            if not parents:
                Q[t] = [(t,)]
                lengths[(t,)] = self.graph.nodes[t]['weight']
            else: 
                comp_length = C[t]
                provL = {}
                for p in parents:
                    edge_weight = self.graph.nodes[t]['weight'] + self.graph[p][t]['weight']
                    for delta in Q[p]:
                        provL[delta + (t,)] = lengths[delta] + edge_weight
                # Sort all paths according to the probability they will beat the mean longest.
                sorted_paths = list(reversed(sorted(provL, key=lambda delta : (provL[delta].mu-comp_length.mu)/sqrt(provL[delta].var+comp_length.var))))
                Q[t] = sorted_paths[:k]
                for delta in Q[t]:
                    lengths[delta] = provL[delta]  
        return Q[self.top_sort[-1]]
    
    def critical_graph(self, paths):
        """
        Not used anywhere. Returns another graph only comprising the specified paths.  

        Parameters
        ----------
        paths : ITERABLE
            The paths to be retained.

        Returns
        -------
        The pruned graph. 

        """
        
        critical_tasks, critical_edges = tuple(), []
        for delta in paths:
            critical_tasks += delta
            critical_edges += pairwise(delta)
        critical_tasks = list(set(critical_tasks))
        critical_edges = list(set(critical_edges))
        R = self.graph.__class__()
        R.add_nodes_from(critical_tasks)
        R.add_edges_from(critical_edges)
        S = StochDAG(R)    
        for t in S.top_sort:
            S.graph.nodes[t]['weight'] = self.graph.nodes[t]['weight'] 
        for u, v in S.graph.edges:
            S.graph[u][v]['weight'] = self.graph[u][v]['weight']
        return S
    
class ScaDAG:
    """Represents a graph with scalar node and edge weights."""
    def __init__(self, graph, aoa=True, johnson=False):
        """
        Initialize with topology. 

        Parameters
        ----------
        graph : Networkx DiGraph with scalar node and edge weights
            DAG topology and weights.
        aoa : BOOL
            Activity-on-arc - i.e., task weights are incorporated into edge weights. This is useful for exploiting some Networkx functions.
            The default is True.
        johnson : BOOL, optional
            Transform scalar graph as in Johnson's algorithm to convert longest to shortest path function. Again, useful for compatability
            with some Networkx functions. The default is False.

        Returns
        -------
        None.
        """
        self.graph = graph
        self.top_sort = list(nx.topological_sort(self.graph))    # Often saves time.  
        self.size = len(self.top_sort)
        self.aoa = aoa
        self.johnson = johnson
        
    def longest_path(self):
        """
        Compute longest path.

        Returns
        -------
        FLOAT/INT
            Longest path length.

        """
        
        if self.aoa:
            lp = nx.algorithms.dag.dag_longest_path(self.graph, topo_order=self.top_sort)
            return sum(self.graph[u][v]['weight'] for u, v in pairwise(lp))
        
        L = {}
        for t in self.top_sort:
            L[t] = self.graph.nodes[t]['weight']
            try:
                L[t] += max(self.graph[p][t]['weight'] + L[p] for p in self.graph.predecessors(t))
            except ValueError:
                pass 
        return L[self.top_sort[-1]]
    
    def yen_klongest_paths(self, k):
        """
        Yen's algorithm for the K longest paths. Not used anywhere. Slow for large DAGs.
        See Networkx documentation for shortest_simple_paths.

        Parameters
        ----------
        k : INT
            Number of longest paths.

        Returns
        -------
        LIST
            The k longest paths through the DAG.

        """      
        assert self.aoa and self.johnson, 'ScaDAG not set-up for use with shortest_simple_paths'
        return list(islice(nx.shortest_simple_paths(self.graph, self.top_sort[0], self.top_sort[-1], weight="weight"), k))
        
    
    def dag_klongest_paths(self, k):
        """
        TODO.

        Parameters
        ----------
        k : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
            
        Q, lengths = {}, {}        
        for t in self.top_sort:
            parents = list(self.graph.predecessors(t))
            # Identify all possible paths.
            if not parents:
                Q[t] = [(t,)]
                lengths[(t,)] = 0.0 if self.aoa else self.graph.nodes[t]['weight']
            else: 
                # Get all possible paths.
                R, prov = [], {}
                for p in parents:
                    edge_weight = self.graph[p][t]['weight'] 
                    if not self.aoa:
                        edge_weight += self.graph.nodes[t]['weight']
                    for delta in Q[p]:
                        path = delta + (t,)
                        R.append(path)
                        # lengths[path] = lengths[delta] + edge_weight 
                        prov[path] = lengths[delta] + edge_weight 
                # Sort R.
                R = list(reversed(sorted(R, key=prov.get)))
                Q[t] = R[:k]
                for delta in Q[t]:
                    lengths[delta] = prov[delta]
        return Q[self.top_sort[-1]]
                
            
            
                   
        
# =============================================================================
# GENERAL FUNCTIONS.
# =============================================================================  

def get_StochDAG(G, exp_cov):
    """
    G assumed to be DiGraph with float weights.

    Parameters
    ----------
    G : TYPE
        DESCRIPTION.
    cov : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    
    task_covs = np.random.gamma(100, exp_cov*0.01, len(G))
    edge_covs = np.random.gamma(100, exp_cov*0.01, G.number_of_edges())
    
    # Convert graph to ScaDAG.
    A = G.__class__()
    A.add_nodes_from(G)
    A.add_edges_from(G.edges)
    
    for i, t in enumerate(G.nodes):
        mu = G.nodes[t]['weight']
        sd = task_covs[i] * mu
        A.nodes[t]['weight'] = RV(mu, sd**2)
    
    for i, e in enumerate(G.edges):
        u, v = e
        mu = G[u][v]['weight']
        if mu == 0.0:
            A[u][v]['weight'] = 0.0
        else:
            sd = edge_covs[i] * mu
            A[u][v]['weight'] = RV(mu, sd**2)
                
    return StochDAG(A)   
    
    
def clark(r1, r2, rho=0, minimization=False):
    """
    Returns a new RV representing the maximization of self and other whose mean and variance
    are computed using Clark's equations for the first two moments of the maximization of two normal RVs.
    TODO: minimization from one of Canon's papers, find source and cite.
    See:
    'The greatest of a finite set of random variables,'
    Charles E. Clark (1983).
    """
    a = sqrt(r1.var + r2.var - 2 * r1.sd * r2.sd * rho)     
    b = (r1.mu - r2.mu) / a           
    cdf = NormalDist().cdf(b)
    mcdf = 1 - cdf 
    pdf = NormalDist().pdf(b)      
    if minimization:
        mu = r1.mu * mcdf + r2.mu * cdf - a * pdf 
        var = (r1.mu**2 + r1.var) * mcdf
        var += (r2.mu**2 + r2.var) * cdf
        var -= (r1.mu + r2.mu) * a * pdf
        var -= mu**2 
    else:
        mu = r1.mu * cdf + r2.mu * mcdf + a * pdf      
        var = (r1.mu**2 + r1.var) * cdf
        var += (r2.mu**2 + r2.var) * mcdf
        var += (r1.mu + r2.mu) * a * pdf
        var -= mu**2         
    return RV(mu, var)  

# =============================================================================
# Helper functions for Kamburowski.
# =============================================================================

def h(mu1, var1, mu2, var2):
    """Helper function for Kamburowski method."""
    alpha = sqrt(var1 + var2)
    beta = (mu1 - mu2)/alpha
    cdf_beta = NormalDist().cdf(beta) 
    return mu1*cdf_beta + mu2*(1-cdf_beta) + alpha*NormalDist().pdf(beta)               
                
def funder(X):
    """
    Helper function for Kamburowksi method.
    X is any iterable of RVs, sorted in ascending order of their variance.
    """
    if len(X) == 1:
        return X[0].mu
    elif len(X) == 2:
        return h(X[0].mu, X[0].var, X[1].mu, X[1].var)
    else:
        return h(funder(X[:-1]), 0, X[-1].mu, X[-1].var)

def fover(X):
    """
    Helper function for Kamburowksi method.
    X is any iterable of RVs, sorted in ascending order of their variance.
    """
    if len(X) == 1:
        return X[0].mu
    elif len(X) == 2:
        return h(X[0].mu, X[0].var, X[1].mu, X[1].var)
    else:
        return h(fover(X[:-1]), X[-2].var, X[-1].mu, X[-1].var)  