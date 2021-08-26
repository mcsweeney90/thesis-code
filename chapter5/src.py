#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stochastic scheduling simulator.
"""

import random
import networkx as nx
import numpy as np
import itertools as it
from math import sqrt, radians, tan
from psutil import virtual_memory
from functools import partial
from statistics import NormalDist 

class RV:
    """
    Random variable class.
    Notes:
        - Defined by only mean and variance so can in theory be from any distribution but some functions
          e.g., addition and multiplication assume (either explicitly or implicitly) RV is Gaussian.
          (Addition/mult only done when RV represents a finish time/longest path estimate so is assumed to be
          at least roughly normal anyway.)
        - ID attribute is often useful (e.g., for I/O).
        - Doesn't check that self.mu and self.var are nonzero for gamma. Obviously shouldn't be but sometimes tempting
          programmatically.
        - Random.random faster than numpy for individual realizations.
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

class ADAG:
    """
    Represents a graph with average node and edge weights.
    TODO: rename this - is scalar graph so SDAG arguably makes more sense...
    """
    def __init__(self, graph):
        """Graph is a NetworkX digraph with {Processor ID : float} node and edge weights. Usually output by functions elsewhere..."""
        self.graph = graph
        self.top_sort = list(nx.topological_sort(self.graph))    # Often saves time.  
        self.size = len(self.top_sort)
        
    def node_mean(self, task, weighted=False):
        """TODO."""
        if not weighted:
            return sum(self.graph.nodes[task]['weight'].values())/len(self.graph.nodes[task]['weight'])
        s = sum(1/v for v in self.graph.nodes[task]['weight'].values())
        return len(self.graph.nodes[task]['weight']) / s 
    
    def edge_mean(self, parent, child, weighted=False):
        """TODO."""
        if not weighted:
            return 2 * sum(self.graph[parent][child]['weight'].values()) / len(self.graph.nodes[parent]['weight'])**2
        s1 = sum(1/v for v in self.graph.nodes[parent]['weight'].values())
        s2 = sum(1/v for v in self.graph.nodes[child]['weight'].values())
        cbar = 0.0
        for k, v in self.graph[parent][child]['weight'].items():
            t_w = self.graph.nodes[parent]['weight'][k[0]]
            c_w = self.graph.nodes[child]['weight'][k[1]]             
            cbar += v/(t_w * c_w) 
        cbar *= 2 
        cbar /= (s1 * s2)
        return cbar 
        
    def get_upward_ranks(self, weighted=False):
        ranks = {}
        backward_traversal = list(reversed(self.top_sort))
        for t in backward_traversal:
            ranks[t] = self.node_mean(t, weighted=weighted)
            try:
                ranks[t] += max(self.edge_mean(t, s, weighted=weighted) + ranks[s] for s in self.graph.successors(t))
            except ValueError:
                pass   
        return ranks  
    
    def comm_cost(self, parent, child, source, dest):
        """
        Get the communication/edge cost between parent and child when they are scheduled on source and dest (respectively).
        Assumes communication is symmetric.  
        """
        if source == dest:
            return 0.0
        elif source < dest:
            return self.graph[parent][child]['weight'][(source, dest)]
        else:
            return self.graph[parent][child]['weight'][(dest, source)] # symmetric.
    
    def priority_scheduling(self, 
                        priorities, 
                        policy="EFT", 
                        lookahead_table=None,
                        assignment=None,
                        return_assignment=True):
        """
        Simulates the scheduling of the tasks in priority_list.
        """ 
        
        # Get list of workers - often useful.
        workers = self.graph.nodes[self.top_sort[0]]['weight'].keys()
        
        # Build schedule. Keep track of finish times and where tasks are scheduled.
        schedule = {w : [] for w in workers}
        finish_times, where = {}, {}
        
        # Initialize ready tasks - assumes single entry task.
        ready_tasks = [self.top_sort[0]] 
        # Start the simulation.
        while len(ready_tasks): 
            # Get the highest priority task.
            task = max(ready_tasks, key=priorities.get)
            
            # Find parents.
            parents = list(self.graph.predecessors(task))
            
            # Simulate scheduling on each worker.
            worker_finish_times = {}
            for w in workers:
                if policy == "AMT" and assignment[task] != w:
                    worker_finish_times[w] = (0, float('inf'), 0) # TODO.
                    continue
                task_cost = self.graph.nodes[task]['weight'][w]
                
                # Find the data-ready time.       
                drt = 0.0 if not parents else max(finish_times[p] + self.comm_cost(p, task, where[p], w) for p in parents)
                
                # Find time worker can actually execute the task (assumes insertion). 
                if not schedule[w]:
                    worker_finish_times[w] = (drt, drt + task_cost, 0)
                else:
                    found, prev_finish_time = False, 0.0
                    for i, t in enumerate(schedule[w]):
                        if t[1] < drt:
                            prev_finish_time = t[2]
                            continue
                        poss_start_time = max(prev_finish_time, drt) 
                        poss_finish_time = poss_start_time + task_cost
                        if poss_finish_time <= t[1]:
                            found = True
                            worker_finish_times[w] = (poss_start_time, poss_finish_time, i)                            
                            break
                        prev_finish_time = t[2]    
                    # No valid gap found.
                    if not found:
                        st = max(schedule[w][-1][2], drt)
                        worker_finish_times[w] = (st, st + task_cost, -1) 
                        
            # Find the fastest worker.
            if policy == "EFT" or policy == "AMT":
                min_worker = min(workers, key=lambda w:worker_finish_times[w][1])
            elif policy == "PEFT":
                min_worker = min(workers, key=lambda w:worker_finish_times[w][1] + lookahead_table[task][w])
            
            # Schedule the task on min_worker and update the helper dicts.
            where[task] = min_worker            
            st, ft, idx = worker_finish_times[min_worker]
            finish_times[task] = ft            
            if not schedule[min_worker] or idx < 0:             
                schedule[min_worker].append((task, st, ft))  
            else: 
                schedule[min_worker].insert(idx, (task, st, ft)) 
                
            # Update ready tasks.
            ready_tasks.remove(task)
            for c in self.graph.successors(task):
                if all(p in where for p in self.graph.predecessors(c)):
                    ready_tasks.append(c)      
        
        # Return schedule and the where assignment dict if specified (useful for e.g., building the disjunctive graph).
        if return_assignment:
            return schedule, where
        return schedule
    
    def optimistic_cost_table(self, include_current=False):
        """
        Used in PEFT heuristic.
        TODO: downward version.
        """
        
        workers = self.graph.nodes[self.top_sort[0]]['weight'].keys()
        
        OCT = {} 
        backward_traversal = list(reversed(self.top_sort))
        for task in backward_traversal:
            OCT[task] = {}
            for w in workers:
                OCT[task][w] = self.graph.nodes[task]['weight'][w] if include_current else 0.0
                child_values = []
                for child in self.graph.successors(task):
                    if include_current:
                        action_values = [OCT[child][v] + self.comm_cost(task, child, w, v) for v in workers]
                    else:
                        action_values = [OCT[child][v] + self.comm_cost(task, child, w, v) + self.graph.nodes[child]['weight'][v] for v in workers]
                    child_values.append(min(action_values))   
                OCT[task][w] += max(child_values) if len(child_values) else 0.0 # Don't like...
        return OCT
    
    def path_bound(self):
        """Compute a lower bound on the makespan."""
        return min(self.optimistic_cost_table(include_current=True)[self.top_sort[0]].values())
    
    def work_bound(self):
        n_workers = len(self.graph.nodes[self.top_sort[0]]['weight'])
        min_work = sum(min(self.graph.nodes[t]['weight'].values()) for t in self.top_sort)
        return min_work / n_workers
        

class SDAG:
    """Represents a graph with stochastic node and edge weights."""
    def __init__(self, graph):
        """Graph is an NetworkX digraph with RV nodes and edge weights. Usually output by functions elsewhere..."""            
        self.graph = graph
        self.top_sort = list(nx.topological_sort(self.graph))    # Often saves time.  
        self.size = len(self.top_sort)
        
    def adjust_uncertainty(self, new_cov):
        """
        Modify the uncertainty (i.e., the variance) of all weights.
        Assumes that mean (and initial variance values) are already set.
        Useful for observing the effect of uncertainty on a computed schedule.

        Parameters
        ----------
        new_cov : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        for task in self.top_sort:
            mu = self.graph.nodes[task]['weight'].mu
            new_var = new_cov * mu # TODO: randomness?
            new_sd = sqrt(new_var)
            self.graph.nodes[task]['weight'].var = new_var
            self.graph.nodes[task]['weight'].sd = new_sd
            for child in self.graph.successors(task): 
                if type(self.graph[task][child]['weight']) == float or type(self.graph[task][child]['weight']) == int:
                    continue
                mu = self.graph[task][child]['weight'].mu
                new_var = new_cov * mu # TODO: randomness?
                new_sd = sqrt(new_var)
                self.graph[task][child]['weight'].var = new_var
                self.graph[task][child]['weight'].sd = new_sd
                        
    def CPM(self, variance=False, full=False):
        """
        Returns the classic PERT-CPM bound on the expected value of the longest path.
        If variance == True, also returns the variance of this path to use as a rough estimate
        of the longest path variance.
        TODO: convert to ScaDAG method and compare timing.
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
        
    def longest_path(self, method="MC", mc_dist="U", mc_samples=1000, full=False):
        """
        Evaluate the longest path through the entire DAG.
        TODO: Few possible issues with MC method:
            - Better way to determine memory limit.
            - Full/exit task only versions make the code unwieldy. Check everything works okay.
            - Worth ensuring positivity for normal and uniform? Do negative realizations ever occur?
        """
        
        if method in ["S", "s", "SCULLI", "sculli", "Sculli"]:
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
            return L[self.top_sort[-1]] if not full else L
        
        elif method in ["C", "c", "CORLCA", "CorLCA", "corLCA"]:
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
                            
                        # Find dominant parent for the maximization.
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
            return L[self.top_sort[-1]] if not full else L
        
        elif method in ["MC", "mc", "MONTE CARLO", "Monte Carlo", "monte carlo"]:
            mem_limit = virtual_memory().available // 10 # Size of numpy random array ~ 8 * samples
            if self.size*mc_samples < mem_limit:        
                L = {}
                for t in self.top_sort:
                    m, s = self.graph.nodes[t]['weight'].mu, self.graph.nodes[t]['weight'].sd
                    if mc_dist in ["N", "NORMAL", "normal"]:  
                        w = abs(np.random.normal(m, s, mc_samples))
                        # w = np.random.normal(m, s, mc_samples)
                    elif mc_dist in ["G", "GAMMA", "gamma"]:
                        v = self.graph.nodes[t]['weight'].var
                        sh, sc = (m * m)/v, v/m
                        w = np.random.gamma(sh, sc, mc_samples)
                    elif mc_dist in ["U", "UNIFORM", "uniform"]:
                        u = sqrt(3) * s
                        # w = np.random.uniform(-u + m, u + m, mc_samples) 
                        w = abs(np.random.uniform(-u + m, u + m, mc_samples))
                    parents = list(self.graph.predecessors(t))
                    if not parents:
                        L[t] = w 
                        continue
                    pmatrix = []
                    for p in parents:
                        try:
                            m, s = self.graph[p][t]['weight'].mu, self.graph[p][t]['weight'].sd
                            if mc_dist in ["N", "NORMAL", "normal"]: 
                                # e = np.random.normal(m, s, mc_samples)
                                e = abs(np.random.normal(m, s, mc_samples))
                            elif mc_dist in ["G", "GAMMA", "gamma"]:
                                v = self.graph[p][t]['weight'].var
                                sh, sc = (m * m)/v, v/m
                                e = np.random.gamma(sh, sc, mc_samples)
                            elif mc_dist in ["U", "UNIFORM", "uniform"]:
                                u = sqrt(3) * s
                                # e = np.random.uniform(-u + m, u + m, mc_samples)  
                                e = abs(np.random.uniform(-u + m, u + m, mc_samples))
                            pmatrix.append(np.add(L[p], e))
                        except AttributeError:
                            pmatrix.append(L[p])
                    st = np.amax(pmatrix, axis=0)
                    L[t] = np.add(w, st) 
                return L[self.top_sort[-1]] if not full else L  
            else:
                E = [] if not full else {}
                mx_samples = mem_limit//self.size
                runs = mc_samples//mx_samples
                extra = mc_samples % mx_samples
                for _ in range(runs):
                    if full:
                        L = self.longest_path(method="MC", mc_samples=mx_samples, mc_dist=mc_dist, full=True)
                        if len(E) == 0:
                            E = L
                        else:
                            for t in self.top_sort:
                                E[t] += L[t] # TODO: check if this throws an error because L[t] is an np array.
                    else:   
                        E += list(self.longest_path(method="MC", mc_samples=mx_samples, mc_dist=mc_dist))
                if full:
                    L = self.longest_path(method="MC", mc_samples=extra, mc_dist=mc_dist, full=True)
                    for t in self.top_sort:
                        E[t] += L[t] # TODO: check if this throws an error because L[t] is an np array.
                else:                    
                    E += list(self.longest_path(method="MC", mc_samples=extra, mc_dist=mc_dist))
                return E 
            
    def get_upward_ranks(self, method="S", mc_dist="NORMAL", mc_samples=1000):
        """
        
        Parameters
        ----------
        method : TYPE, optional
            DESCRIPTION. The default is "S".
        mc_dist : TYPE, optional
            DESCRIPTION. The default is "NORMAL".
        mc_samples : TYPE, optional
            DESCRIPTION. The default is 1000.

        Returns
        -------
        None.
        
        Not the most efficient way of doing this since the entire graph is copied, but the overhead is typically low compared
        to the cost of the longest path algorithms so this isn't a huge issue.
        Intended to be used for "averaged" graphs. 
        """    
        
        R = SDAG(self.graph.reverse())
        return R.longest_path(method=method, mc_dist=mc_dist, mc_samples=mc_samples, full=True)       
               
class TDAG:
    """Represents a graph with stochastic node and edge weights."""
    def __init__(self, graph):
        """
        Graph is a NetworkX digraph with {Processor ID : RV} node and edge weights. Usually output by functions elsewhere...
        TODO: create a "workers" attribute since often useful when weights have been set?
        """
        self.graph = graph
        self.top_sort = list(nx.topological_sort(self.graph))    # Often saves time.  
        self.size = len(self.top_sort)
        
    def set_weights(self, n_processors=4, rtask=0.5, rmach=0.5, mu=1.0, V=0.5, muccr=1.0, mucov=0.1):
        """
        Used for setting randomized weights for DAGs.
        TODO: double check.
        """
                
        # Do corrections.
        n1 = 1 + (rtask - 2*rtask*rmach + rmach)*V**2 - rtask*rmach
        n2 = (rtask - rmach)**2 * V**4 + 2*(rtask*(rmach-1)**2 + rmach*(rtask-1)**2)*V**2 + (rtask*rmach - 1)**2
        vnoise = sqrt((n1 - sqrt(n2))/(2*rtask*rmach*(V**2 + 1)))
        vtask = 1/sqrt((1/rmach - 1)/vnoise**2 - 1)
        vmach = 1/sqrt((1/rtask - 1)/vnoise**2 - 1)
        # Generate the costs.
        task_sizes = np.random.gamma(1/vtask**2, vtask**2, size=self.size)
        machine_powers = np.random.gamma(1/vmach**2, vmach**2, size=n_processors)            
        costs = np.random.gamma(1/vnoise**2, vnoise**2, size=(self.size, n_processors))
        costs *= task_sizes[:, None]
        costs *= machine_powers
        costs *= mu
        task_covs = np.random.gamma(100, mucov*0.01, size=(self.size, n_processors)) # CoV of CoVs assumed to be 0.1 (confusing!)
        for i, t in enumerate(self.top_sort):             
            self.graph.nodes[t]['weight'] = {j : RV(mu, (mu * task_covs[i][j])**2) for j, mu in enumerate(costs[i])} 
        
        B = np.random.gamma(1/V**2, V**2, size=(n_processors, n_processors))
        B_rep_sum = sum(np.reciprocal(B[np.triu_indices(B.shape[0], k=1)])) 
        # Set the actual communication costs.
        ccrs = np.random.gamma(1/V**2, muccr*(V**2), size=self.size)
        edge_covs = np.random.gamma(100, mucov*0.01, size=(self.graph.number_of_edges(), n_processors, n_processors)) 
        for i, e in enumerate(self.graph.edges):
            t, s = e
            expected_compute = np.mean(list(val.mu for val in self.graph.nodes[t]['weight'].values()))
            D = (ccrs[t] * expected_compute*n_processors**2) / (2 * B_rep_sum) 
            self.graph[t][s]['weight'] = {}
            for w in range(n_processors):
                for p in range(w + 1, n_processors):
                    m = D/B[w][p]
                    sd = m * edge_covs[i][w][p]
                    self.graph[t][s]['weight'][(w, p)] = RV(m, sd) 
                    
    def adjust_cov(self, new_cov):
        """
        TODO.

        Parameters
        ----------
        new_cov : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        return
                                      
                        
    def minimal_serial_time(self, mc_samples=1000):
        """
        Extension of minimal serial time for scalar graphs. 
        Uses MC sampling to estimate the MST distribution, assuming that times are normally distributed (fairly
        reasonable as they are typically the sum of 100 or so task costs).  
        Assumes they are all independent (reasonable?)
        """
        workers = list(self.graph.nodes[self.top_sort[0]]['weight'])
        realizations = []
        for w in workers:
            dist = sum(self.graph.nodes[t]['weight'][w] for t in self.top_sort) 
            r = np.random.normal(dist.mu, dist.sd, mc_samples)
            realizations.append(r)
        return list(np.amin(realizations, axis=0))
    
    def get_scalar_graph(self, scal_func=lambda r : r.mu):
        """Return an ADAG object..."""
        
        # Copy the topology.
        A = self.graph.__class__()
        A.add_nodes_from(self.graph)
        A.add_edges_from(self.graph.edges)
        
        # Set the weights.
        for t in self.top_sort:
            # Set node weights.
            A.nodes[t]['weight'] = {k : scal_func(v) for k, v in self.graph.nodes[t]['weight'].items()}
            # Set edge weights.
            for s in self.graph.successors(t):
                A[t][s]['weight'] = {k : scal_func(v) for k, v in self.graph[t][s]['weight'].items()}
                
        # Return ADAG object.      
        return ADAG(A)
    
    def get_averaged_graph(self, avg_type="NORMAL"):
        """Return a graph with averaged weights..."""
        
        # Copy the topology.
        A = self.graph.__class__()
        A.add_nodes_from(self.graph)
        A.add_edges_from(self.graph.edges)
        
        # TODO: are these needed since changes?
        # mean = lambda r : 0.0 if (type(r) == float or type(r) == int) else r.mu
        # var = lambda r : 0.0 if (type(r) == float or type(r) == int) else r.var
                
        # Stochastic averages.
        if avg_type in ["N", "NORMAL", "CLT"]:
            L = len(self.graph.nodes[self.top_sort[0]]['weight'])
            L2 = L*L
            for t in self.top_sort:
                m = sum(r.mu for r in self.graph.nodes[t]['weight'].values()) 
                v = sum(r.var for r in self.graph.nodes[t]['weight'].values())
                A.nodes[t]['weight'] = RV(m, v)/L
                for s in self.graph.successors(t):
                    # m1 = 2*sum(mean(r) for r in self.graph[t][s]['weight'].values())
                    # v1 = 2*sum(var(r) for r in self.graph[t][s]['weight'].values())
                    m1 = 2*sum(r.mu for r in self.graph[t][s]['weight'].values())
                    v1 = 2*sum(r.var for r in self.graph[t][s]['weight'].values())
                    A[t][s]['weight'] = RV(m1, v1)/L2
            return SDAG(A)
        
        raise ValueError("Invalid stochastic average type!")        
    
    def comm_cost(self, parent, child, source, dest):
        if source == dest:
            return 0.0
        elif source < dest:
            return self.graph[parent][child]['weight'][(source, dest)]
        else:
            return self.graph[parent][child]['weight'][(dest, source)] # symmetric.
    
    def schedule_to_graph(self, schedule, where_scheduled=None):
        """
        Convert (fullahead) schedule to a "disjunctive" graph with stochastic weights whose longest path gives the makespan.
        
        Parameters
        ----------
        schedule : DICT
            {worker : [(task_id, est_start_time, est_finish_time), ...]}.
            Note est_start_time and est_finish_time may be scalar or RVs.
        where : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        
        if where_scheduled is None:
            where_scheduled = {}
            for w, load in schedule.items():
                for t in list(s[0] for s in load):
                    where_scheduled[t] = w 
        
        # Construct and return the schedule graph.
        S = self.graph.__class__()
        S.add_nodes_from(self.graph)
        S.add_edges_from(self.graph.edges)
        # Set the weights.
        for t in self.top_sort:
            w = where_scheduled[t]
            S.nodes[t]['weight'] = self.graph.nodes[t]['weight'][w] 
            for s in self.graph.successors(t):
                w1 = where_scheduled[s]
                S[t][s]['weight'] = self.comm_cost(t, s, w, w1)
            # Add disjunctive edge if necessary.
            idx = list(r[0] for r in schedule[w]).index(t)
            if idx > 0:
                d = schedule[w][idx - 1][0]
                if not S.has_edge(d, t):
                    S.add_edge(d, t)
                    S[d][t]['weight'] = 0.0
        return SDAG(S) 
    
    def priority_scheduling(self, 
                        priorities, 
                        prio_function, 
                        selection_function,
                        insertion=False, 
                        eval_method="MC",
                        eval_dist="N",
                        eval_samples=1000):
        """
        TODO. Insertion. Create ERV class?
        Quite a few differences from the ADAG method:
            1. Priorities are now assumed to be RVs/empirical RVs, so prio_function is needed to scalarize them (but in future might
               want to do something else instead). 
            2. Similarly for selection function.
        """
        
        # Get list of workers - often useful.
        workers = self.graph.nodes[self.top_sort[0]]['weight'].keys()
        
        # Create the schedule graph.
        S = self.graph.__class__()       
        
        # Simulate and find schedule.  
        where, last = {}, {}    # Helpers.
        ready_tasks = [self.top_sort[0]] 
        while ready_tasks:
            task = max(ready_tasks, key=prio_function)      # TODO: what if min?
            S.add_node(task)
            parents = list(self.graph.predecessors(task))
            for p in parents:
                S.add_edge(p, task)
            worker_makespans = {}
            for w in workers:
                
                # Set schedule node weights. 
                S.nodes[task]['weight'] = self.graph.nodes[task]['weight'][w]
                for p in parents:
                    S[p][task]['weight'] = self.comm_cost(p, task, where[p], w)
                    
                # Add the transitive edge if necessary. TODO: insertion?
                remove = False
                try:
                    L = last[w]
                    if not S.has_edge(L, task):
                        S.add_edge(L, task)
                        S[L][task]['weight'] = 0.0
                        remove = True
                except KeyError:
                    pass
                
                # Add artificial exit node if necessary. TODO: don't like this at all. And very slow.
                exit_tasks = [t for t in S if not len(list(S.successors(t)))]
                if len(exit_tasks) > 1:
                    S.add_node("X")
                    S.nodes["X"]['weight'] = RV(0.0, 0.0) # don't like.
                    for e in exit_tasks:
                        S.add_edge(e, "X")
                        S[e]["X"]['weight'] = 0.0 
                        
                # Compute longest path using specified method.
                # worker_makespans[w] = SDAG(S).longest_path(method=eval_method, mc_dist=eval_dist, mc_samples=eval_samples) # TODO.
                P = SDAG(S)
                if eval_method in ["MC", "mc"]:
                    worker_dist = P.longest_path(method="MC", mc_dist=eval_dist, mc_samples=eval_samples) # TODO: ERV class. 
                    m = sum(worker_dist)/len(worker_dist)
                    v = np.var(worker_dist)
                    worker_makespans[w] = RV(m, v)
                else:
                    worker_makespans[w] = P.longest_path(method=eval_method)
                
                # Clean up - remove edge etc. TODO: need to set node weight to zero?
                if remove:
                    S.remove_edge(L, task)
                if len(exit_tasks) > 1:
                    S.remove_node("X")
                    
            # Select the "best" worker according to the specified method.
            # TODO: used this syntax rather than e.g., min(mkspans, key=sel_function) to allow for more complex possibilities.
            chosen_worker = selection_function(worker_makespans)    
            # "Schedule" the task on chosen worker.
            where[task] = chosen_worker
            S.nodes[task]['weight'] = self.graph.nodes[task]['weight'][chosen_worker]
            # Same for edges.
            for p in parents:
                S[p][task]['weight'] = self.comm_cost(p, task, where[p], chosen_worker)
            try:
                L = last[chosen_worker] # TODO: insertion.
                if not S.has_edge(L, task):
                    S.add_edge(L, task)
                    S[L][task]['weight'] = 0.0
            except KeyError:
                pass
            last[chosen_worker] = task
            
            # Remove current task from ready set and add those now available for scheduling.
            ready_tasks.remove(task)
            for c in self.graph.successors(task):
                if all(p in where for p in self.graph.predecessors(c)):
                    ready_tasks.append(c) 
       
        return SDAG(S)    
        
        
# =============================================================================
# DETERMINISTIC HEURISTICS.
# =============================================================================

def HEFT(G, weighted=False):
    """
    HEFT scheduling heuristic.
    TODO: assumes G is an ADAG. If TDAG, convert to scalar then apply?
    """
    # Compute upward ranks.
    U = G.get_upward_ranks(weighted=weighted)
    # Simulate to get the schedule and return it.
    return G.priority_scheduling(priorities=U, policy="EFT")

def PEFT(G):
    """
    HEFT scheduling heuristic.
    TODO: assumes G is an ADAG. If TDAG, convert to scalar then apply?
    """
    # Compute optimistic cost table.
    OCT = G.optimistic_cost_table(include_current=False)
    # Compute the ranks.
    ranks = {t : sum(OCT[t].values())/len(OCT[t].values()) for t in G.top_sort}     
    # Get schedule.
    return G.priority_scheduling(priorities=ranks, policy="PEFT", lookahead_table=OCT) 

# =============================================================================
# STOCHASTIC HEURISTICS.
# =============================================================================

def SSTAR(T, det_heuristic, scal_func=lambda r : r.mu, scalar_graph=None):
    """
    Converts a TDAG S to a "scalarized" ADAG object using scal_func, then applies det_heuristic to it. 
    When avg_type == "MEAN" and det_heuristic == HEFT, this is just HEFT applied to the stochastic graph.
    When avg_type == "SHEFT" and det_heuristic == HEFT, this is the Stochastic HEFT (SHEFT) heuristic.        
    'A stochastic scheduling algorithm for precedence constrained tasks on Grid',
    Tang, Li, Liao, Fang, Wu (2011).
    However, this function can take any other deterministic heuristic instead.

    Parameters
    ----------
    S : TYPE
        DESCRIPTION.
    det_heuristic : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # Convert to an "averaged" graph with scalar weights (if necessary).
    if scalar_graph is None:
        scalar_graph = T.get_scalar_graph(scal_func=scal_func)
    # Apply the chosen heuristic.
    P, where = det_heuristic(scalar_graph) 
    # Convert fullahead schedule to its (stochastic) disjunctive graph and return.
    return T.schedule_to_graph(schedule=P, where_scheduled=where)
    

def SDLS(T, X=0.9, return_graph=True, insertion=None):
    """
    TODO: Insertion doesn't seem to make much of a difference but evaluate this more thoroughly.
    TODO: Still may be too slow for large DAGs. Obviously copying graph etc is not optimal but those kind of things aren't the
    real bottlenecks.
    
    Returns
    -------
    None.

    """
    
    mean = lambda r : 0.0 if (type(r) == float or type(r) == int) else r.mu # TODO: Hmmm... Only needed for insertion.
    
    # Get the list of workers - useful throughout.
    workers = list(T.graph.nodes[T.top_sort[0]]['weight'])
    
    # Convert to stochastic averaged graph.
    A = T.get_averaged_graph(avg_type="NORMAL")    
    # Get upward ranks via Sculli's method.
    B = A.get_upward_ranks(method="S") 
    
    # Compute the schedule.
    ready_tasks = [T.top_sort[0]]    # Assumes single entry task.   
    SDL, FT, where = {}, {}, {}     # FT and where make computing the SDL values easier. 
    schedule = {w : [] for w in workers}   # Loads are ordered
    while len(ready_tasks): 
        for task in ready_tasks:
            # Get the average task weight (used for computing the SDL values).
            avg_weight = A.graph.nodes[task]['weight']
            # Find all parents - useful for next part.
            parents = list(T.graph.predecessors(task)) 
            # Compute SDL value of the task on all workers.
            for w in workers:
                # Compute delta.
                delta = avg_weight - T.graph.nodes[task]['weight'][w]    
                
                # Compute earliest start time. 
                if not parents: # Single entry task.
                    EST, idx = 0.0, 0
                else:
                    p = parents[0]
                    # Compute DRT - start time without considering processor contention.
                    drt = FT[p] + T.comm_cost(p, task, where[p], w) 
                    for p in parents[1:]:
                        q = FT[p] + T.comm_cost(p, task, where[p], w) 
                        drt = clark(drt, q, rho=0)
                    # Find the earliest time task can be scheduled on the worker.
                    if not schedule[w]: # No tasks scheduled on worker. 
                        EST, idx = drt, 0
                    else:
                        if insertion is None:
                            EST, idx = clark(drt, schedule[w][-1][2], rho=0), len(schedule[w])
                        elif insertion in ["M", "MEAN"]: # mean instead of .mu - EST can be scalar...
                            task_cost = T.graph.nodes[task]['weight'][w].mu
                            found, prev_finish_time = False, 0.0
                            for i, t in enumerate(schedule[w]):
                                if mean(t[1]) < drt.mu:
                                    prev_finish_time = t[2]
                                    continue
                                try:
                                    poss_start_time = clark(prev_finish_time, drt) 
                                except AttributeError:
                                    poss_start_time = drt
                                poss_finish_time = poss_start_time + task_cost
                                if poss_finish_time.mu <= t[1].mu:
                                    found = True
                                    EST, idx = poss_start_time, i                           
                                    break
                                prev_finish_time = t[2]    
                            # No valid gap found.
                            if not found:
                                EST, idx = clark(drt, schedule[w][-1][2], rho=0), len(schedule[w])
                                                     
                # Compute SDL. (EST included as second argument to avoid having to recalculate it later but a bit ugly.)
                SDL[(task, w)] = (B[task] - EST + delta, EST, idx)
            
        # Select the maximum pair. 
        chosen_task, chosen_worker = max(it.product(ready_tasks, workers), 
                                          key=lambda pr : NormalDist(SDL[pr][0].mu, SDL[pr][0].sd).inv_cdf(X))
        # Comment above and uncomment below for Python versions < 3.8.
        # chosen_task, chosen_worker = max(it.product(ready_tasks, workers), key=lambda pr : norm.ppf(X, SDL[pr][0].mu, SDL[pr][0].sd))
                
        # Schedule the chosen task on the chosen worker. 
        where[chosen_task] = chosen_worker
        FT[chosen_task] = SDL[(chosen_task, chosen_worker)][1] + T.graph.nodes[chosen_task]['weight'][chosen_worker]
        schedule[chosen_worker].insert(SDL[(chosen_task, chosen_worker)][2], (chosen_task, SDL[(chosen_task, chosen_worker)][1], FT[chosen_task]))        
        
        # Remove current task from ready set and add those now available for scheduling.
        ready_tasks.remove(chosen_task)
        for c in T.graph.successors(chosen_task):
            if all(p in where for p in T.graph.predecessors(c)):
                ready_tasks.append(c)   
                
    # If specified, convert schedule to graph and return it.
    if return_graph:
        return T.schedule_to_graph(schedule, where_scheduled=where)
    # Else return schedule only.
    return schedule  

def rob_selection(est_makespans, alpha=45):
    """
    Helper function for RobHEFT.

    Parameters
    ----------
    points : TYPE
        DESCRIPTION.
    a : TYPE, optional
        DESCRIPTION. The default is 45.

    Returns
    -------
    None.

    """     
    # Filter the dominated workers. This might be more expensive than it's worth given the size of the sets under consideration.  
    sorted_workers = list(sorted(est_makespans, key=lambda p : est_makespans[p].mu)) # Ascending order of expected value.
    dominated = [False] * len(sorted_workers)           
    for i, w in enumerate(sorted_workers):
        if dominated[i]:
            continue
        for j, q in enumerate(sorted_workers[:i]):   
            if dominated[j]:
                continue
            if est_makespans[q].sd < est_makespans[w].sd:
                dominated[i] = True 
    nondominated = {w : est_makespans[w] for i, w in enumerate(sorted_workers) if not dominated[i]}
        
    # Find max mean and standard deviation for normalization.
    mxm = max(m.mu for m in nondominated.values())
    mxs = max(m.sd for m in nondominated.values())
    
    # Line segment runs from (0, 0) to (1, tan(alpha)) - but need to convert to radians first.
    line_end_pt = tan(radians(alpha)) 
    dist = lambda w : abs(line_end_pt * (nondominated[w].mu/mxm) - (nondominated[w].sd/mxs)) / sqrt(1 + line_end_pt**2)
    return min(nondominated, key=dist)    

def RobHEFT(T, alpha=45, method="C", mc_dist="N", mc_samples=1000):
    """
    RobHEFT (HEFT with robustness) heuristic.
    'Evaluation and optimization of the robustness of DAG schedules in heterogeneous environments,'
    Canon and Jeannot (2010). 
    TODO: this is a deliberately fairly slow implementation that places clarity/re-use of existing code above speed. May write a 
    faster version if it's ever necessary...
    TODO: not 100% convinced this actually works properly.
    """
    
    # Compute priorities.
    A = T.get_averaged_graph(avg_type="NORMAL") 
    R = SDAG(A.graph.reverse())
    ranks = R.CPM(variance=True, full=True) # TODO: check this still works.
    # Get maximums for later normalization. TODO: look at this.
    mx_mu = ranks[T.top_sort[0]].mu
    mx_sd = max(ranks[t].sd for t in T.top_sort)
    prio_function = lambda t : alpha*(ranks[t].mu/mx_mu) + (90-alpha)*(ranks[t].sd/mx_sd)   
    sel_function = partial(rob_selection, alpha=alpha)
    return T.priority_scheduling(priorities=ranks, 
                             prio_function=prio_function,
                             selection_function=sel_function,
                             eval_method=method, 
                             eval_dist=mc_dist, 
                             eval_samples=mc_samples)        
    
# def MCS(S, 
#         production_heuristic=HEFT, 
#         production_steps=100, 
#         threshold=0.02, 
#         prod_dist="N",
#         return_all=False,
#         eval_method="C",
#         mc_samples=1000,
#         criterion="MU",
#         c=1.0,
#         return_mkspan=True):
#     """ 
#     Monte Carlo Scheduling (MCS).
#     'Stochastic DAG scheduling using a Monte Carlo approach,'
#     Zheng and Sakellariou (2013).
#     Slightly slower than below. 
#     """        
    
#     if prod_dist in ["N", "n", "NORMAL", "normal"]:
#         real_func = lambda r : abs(random.gauss(r.mu, r.sd))
#     elif prod_dist in ["G", "g", "GAMMA", "gamma"]:
#         real_func = lambda r : random.gammavariate(alpha=(r.mu**2/r.var), beta=r.var/r.mu) # TODO: check these parameters.
#     elif prod_dist in ["U", "u", "UNIFORM", "uniform"]:
#         real_func = lambda r : abs(random.uniform(r.mu - sqrt(3)*r.sd, r.mu + sqrt(3)*r.sd)) 
            
#     # Create list of schedules.
#     L = []
    
#     # Get the standard static schedule (i.e., with mean values). 
#     avg_graph = S.get_scalar_graph() 
#     mean_static_schedule, where = production_heuristic(avg_graph) 
#     # Get schedule graph (easier to evaluate makespan).
#     omega_mean = S.schedule_to_graph(schedule=mean_static_schedule, where_scheduled=where)
#     # Add schedule graph to L.
#     L.append(omega_mean)
#     # Compute initial qualification check.
#     min_cpm = omega_mean.CPM()        
    
#     # Production steps.
#     for _ in range(production_steps):
#         # Generate a scalarized graph. 
#         G = S.get_scalar_graph(scal_func=real_func)
#         # Compute schedule.
#         candidate, where = production_heuristic(G)
#         # Convert schedule to graph for evaluation. 
#         omega = S.schedule_to_graph(schedule=candidate, where_scheduled=where)
#         # TODO. Want a quick check here that omega has not been seen before but usually more expensive. 
#         # (Usual suspects like set conversion don't work with schedule graphs.)
#         # Calculate CPM bound on expected value of omega makespan and compare.
#         omega_cpm = omega.CPM() 
#         if omega_cpm < min_cpm * (1 + threshold): 
#             L.append(omega)
#         # Update qualification check.
#         min_cpm = min(min_cpm, omega_cpm)
#     if return_all:
#         return L
    
#     # Evaluate schedule makespans.
#     if eval_method in ["MC", "mc", "MONTE CARLO", "Monte Carlo", "monte carlo"]:
#         makespans = {}
#         for pi in L:
#             dist = pi.longest_path(method=eval_method, mc_dist=prod_dist, mc_samples=mc_samples)
#             mu = np.mean(dist)
#             var = np.var(dist)
#             makespans[pi] = RV(mu, var)
#     else:
#         makespans = {pi : pi.longest_path(method=eval_method) for pi in L} 
    
#     # Choose the best schedule according to the specified criterion.
#     if criterion in ["MU", "mu", "MEAN", "mean", "M", "m"]:
#         pi_star = min(L, key=lambda pi : makespans[pi].mu)
#     elif criterion in ["SD", "sd", "SIGMA", "sigma"]:
#         pi_star = min(L, key=lambda pi : makespans[pi].sd)
#     elif criterion in ["UCB", "ucb"]:
#         pi_star = min(L, key=lambda pi : makespans[pi].mu + c*makespans[pi].sd)
#     if return_mkspan:
#         return pi_star, makespans[pi_star]
#     return pi_star

def MCS(S, 
        production_heuristic=HEFT, 
        production_steps=100, 
        threshold=0.02, 
        prod_dist="N",
        return_all=False,
        eval_method="C",
        mc_samples=1000,
        criterion="MU",
        c=1.0,
        return_mkspan=True):
    """ 
    Monte Carlo Scheduling (MCS).
    'Stochastic DAG scheduling using a Monte Carlo approach,'
    Zheng and Sakellariou (2013).
    This is a slightly faster version than above - although still pretty slow.
    """    
        
    realizations = {}
    for t in S.top_sort:
        for key, val in S.graph.nodes[t]['weight'].items():
            mu = val.mu
            var = val.var
            sd = val.sd
            if prod_dist in ["N", "n", "NORMAL", "normal"]:
                realizations[(t, key)] = abs(np.random.normal(mu, sd, production_steps))
            elif prod_dist in ["G", "g", "GAMMA", "gamma"]:
                realizations[(t, key)] = np.random.gamma(mu**2/var, var/mu, production_steps)
            elif prod_dist in ["U", "u", "UNIFORM", "uniform"]:
                realizations[(t, key)] = abs(np.random.uniform(mu - sqrt(3)*sd, mu + sqrt(3)*sd, production_steps))    
    for u, v in S.graph.edges:
        for key, val in S.graph[u][v]['weight'].items():
            mu = val.mu
            var = val.var
            sd = val.sd
            if prod_dist in ["N", "n", "NORMAL", "normal"]:
                realizations[((u, v), key)] = abs(np.random.normal(mu, sd, production_steps))
            elif prod_dist in ["G", "g", "GAMMA", "gamma"]:
                realizations[((u, v), key)] = np.random.gamma(mu**2/var, var/mu, production_steps)
            elif prod_dist in ["U", "u", "UNIFORM", "uniform"]:
                realizations[((u, v), key)] = abs(np.random.uniform(mu - sqrt(3)*sd, mu + sqrt(3)*sd, production_steps))
            
        
    # Create list of schedules.
    L = []
    
    # Get the standard static schedule (i.e., with mean values). 
    avg_graph = S.get_scalar_graph() 
    mean_static_schedule, where = production_heuristic(avg_graph) 
    # Get schedule graph (easier to evaluate makespan).
    omega_mean = S.schedule_to_graph(schedule=mean_static_schedule, where_scheduled=where)
    # Add schedule graph to L.
    L.append(omega_mean)
    # Compute initial qualification check.
    min_cpm = omega_mean.CPM()    

    # Copy the topology.
    A = S.graph.__class__()
    A.add_nodes_from(S.graph)
    A.add_edges_from(S.graph.edges)
    G = ADAG(A)    
    
    # Production steps.
    for i in range(production_steps):
        # Generate a scalarized graph. 
        for t in G.top_sort:
            G.graph.nodes[t]['weight'] = {k : realizations[(t, k)][i] for k, v in S.graph.nodes[t]['weight'].items()}
        for u, v in G.graph.edges:
            G.graph[u][v]['weight'] = {k : realizations[((u, v), k)][i] for k, val in S.graph[u][v]['weight'].items()}
        # Compute schedule.
        candidate, where = production_heuristic(G)
        # Convert schedule to graph for evaluation. 
        omega = S.schedule_to_graph(schedule=candidate, where_scheduled=where)
        # TODO. Want a quick check here that omega has not been seen before but usually more expensive. 
        # (Usual suspects like set conversion don't work with schedule graphs.)
        # Calculate CPM bound on expected value of omega makespan and compare.
        omega_cpm = omega.CPM() 
        if omega_cpm < min_cpm * (1 + threshold): 
            L.append(omega)
        # Update qualification check.
        min_cpm = min(min_cpm, omega_cpm)
    if return_all:
        return L
    
    # Evaluate schedule makespans.
    if eval_method in ["MC", "mc", "MONTE CARLO", "Monte Carlo", "monte carlo"]:
        makespans = {}
        for pi in L:
            dist = pi.longest_path(method=eval_method, mc_dist=prod_dist, mc_samples=mc_samples)
            mu = np.mean(dist)
            var = np.var(dist)
            makespans[pi] = RV(mu, var)
    else:
        makespans = {pi : pi.longest_path(method=eval_method) for pi in L} 
        
    # Choose the best schedule according to the specified criterion.
    if criterion in ["MU", "mu", "MEAN", "mean", "M", "m"]:
        pi_star = min(L, key=lambda pi : makespans[pi].mu)
    elif criterion in ["SD", "sd", "SIGMA", "sigma"]:
        pi_star = min(L, key=lambda pi : makespans[pi].sd)
    elif criterion in ["UCB", "ucb"]:
        pi_star = min(L, key=lambda pi : makespans[pi].mu + c*makespans[pi].sd)
    if return_mkspan:
        return pi_star, makespans[pi_star]
    return pi_star
        
            
# =============================================================================
# GENERAL FUNCTIONS.
# =============================================================================   
    
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
        
        

        
                
