#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main source file for simulation code.
"""

import networkx as nx
import numpy as np
from math import sqrt
from statistics import median, harmonic_mean, geometric_mean, pstdev
from timeit import default_timer as timer

class DAG:
    """
    Represents a task graph.
    """
    def __init__(self, graph):
        """
        Initialize with topology. 

        Parameters
        ----------
        graph : Networkx DiGraph
            DAG topology.

        Returns
        -------
        None.
        """
        self.graph = graph
        self.top_sort = list(nx.topological_sort(self.graph))    # Often saves time.  
        self.size = len(self.top_sort)
        
        # Set once weights are set.
        self.weighted = False
        self.workers = None 
        self.nworkers = None
        self.comm_costs = None # If all comm_costs are the same (e.g., Cholesky since data the same for all tasks.)
        
    def set_cholesky_weights(self, 
                             nprocessors, 
                             ccr,
                             vproc=1.0, 
                             vrel=1.0,
                             vband=1.0):
        """
        Set weights for Cholesky graphs.

        Parameters
        ----------
        nprocessors : INT
            The number of processors.
        ccr : FLOAT
            The target CCR.
        vproc : FLOAT, optional
            CoV of processor speeds. The default is 1.0.
        vrel : FLOAT, optional.
            CoV of relatedness. The default is 1.0.
        vband : FLOAT, optional.
            CoV of link bandwidths. The default is 1.0.

        Returns
        -------
        None.
        """
        
        # Set DAG as weighted.
        self.weighted = True
        self.workers = list(range(nprocessors))
        self.nworkers = nprocessors
        
        # Set the communication costs.
        processor_powers = np.random.gamma(1/vproc**2, vproc**2, size=nprocessors)     
        task_sizes = {"G" : 6, "P" : 1, "S" : 3, "T" : 3} 
        proc_task_times = {}
        noise = np.random.gamma(1/vrel**2, vrel**2, size=(nprocessors, 4)) 
        for w in range(nprocessors):
            for j, task_type in enumerate(["G", "P", "S", "T"]):
                theory = task_sizes[task_type] / processor_powers[w]
                proc_task_times[(w, task_type)] = theory * noise[w][j]
        for t in self.top_sort:
            task_type = t[0]
            self.graph.nodes[t]['weight'] = np.fromiter((proc_task_times[(w, task_type)] for w in self.workers), float)
            
        # Set the communication costs.
        expected_total_compute = sum(self.task_average(t) for t in self.top_sort)
        expected_comm = ccr * expected_total_compute   
        edge_mean = expected_comm/self.graph.number_of_edges()  
        mean_comm_cost = (edge_mean * self.nworkers)/(self.nworkers - 1)        
        comm_costs = np.random.gamma(1/vband**2, mean_comm_cost*vband**2, size=(self.nworkers, self.nworkers))
        self.comm_costs = {}
        for w in range(self.nworkers):
            for p in range(w + 1, self.nworkers):
                self.comm_costs[(w, p)] = comm_costs[w][p]
                
    def set_random_weights(self, 
                           nprocessors, 
                           comp_method="CNB",
                           comp_params=None,
                           vband=1.0,
                           muccr=0.1,
                           vccr=1.0):
        """
        Set weights according to specified methods for generating random costs.

        Parameters
        ----------
        nprocessors : INT
            The number of processors.
        comp_method : STRING, optional
            Method used to set computation costs. The default is "CNB".
        comp_params : ITERABLE, optional
            Parameters needed for method of setting computation costs. The default is None.
        vband : FLOAT, optional
            Coefficient of variation of bandwidths. The default is 1.0.
        muccr : FLOAT, optional
            Mean CCR. The default is 0.1.
        vccr : FLOAT, optional
            Coefficient of variation for gamma distribution CCRs are sampled from. The default is 1.0.

        Returns
        -------
        None.
        """
        
        # Set DAG as weighted. Do this after in case of failure?
        self.weighted = True
        self.workers = list(range(nprocessors))
        self.nworkers = nprocessors
        
        # Set computation costs.
        if comp_method in ["rb", "RB"]:
            wdag, beta = comp_params # Will throw an error if wrong size...
            wbar = np.random.uniform(0, 2*wdag, size=self.size)
            noise = np.random.uniform(1-beta/2, 1+beta/2, size=(self.size, self.nworkers))            
            costs = noise * wbar[:, None]
            for i, t in enumerate(self.top_sort):
                self.graph.nodes[t]['weight'] = costs[i]                
            
        elif comp_method in ["cvb", "CVB"]:
            vtask, vmach, mutask = comp_params
            alpha_task, alpha_mach = 1/vtask**2, 1/vmach**2
            beta_task = mutask/alpha_task
            beta_mach = np.random.gamma(alpha_task, beta_task, size=self.size)/alpha_mach
            for i, t in enumerate(self.top_sort):
                self.graph.nodes[t]['weight'] = np.random.gamma(alpha_mach, beta_mach[i], size=self.nworkers)          
                
        elif comp_method in ["nb", "NB"]:
            vtask, vmach, vnoise = comp_params
            task_sizes = np.random.gamma(1/vtask**2, vtask**2, size=self.size)
            machine_powers = np.random.gamma(1/vmach**2, vmach**2, size=self.nworkers)            
            costs = np.random.gamma(1/vnoise**2, vnoise**2, size=(self.size, self.nworkers))
            costs *= task_sizes[:, None]
            costs *= machine_powers
            for i, t in enumerate(self.top_sort):
                self.graph.nodes[t]['weight'] = costs[i]             
            
        elif comp_method in ["cnb", "CNB"]:
            rtask, rmach, mu, V = comp_params 
            # Do corrections.
            n1 = 1 + (rtask - 2*rtask*rmach + rmach)*V**2 - rtask*rmach
            n2 = (rtask - rmach)**2 * V**4 + 2*(rtask*(rmach-1)**2 + rmach*(rtask-1)**2)*V**2 + (rtask*rmach - 1)**2
            vnoise = sqrt((n1 - sqrt(n2))/(2*rtask*rmach*(V**2 + 1)))
            vtask = 1/sqrt((1/rmach - 1)/vnoise**2 - 1)
            vmach = 1/sqrt((1/rtask - 1)/vnoise**2 - 1)
            # Generate the costs.
            task_sizes = np.random.gamma(1/vtask**2, vtask**2, size=self.size)
            machine_powers = np.random.gamma(1/vmach**2, vmach**2, size=self.nworkers)            
            costs = np.random.gamma(1/vnoise**2, vnoise**2, size=(self.size, self.nworkers))
            costs *= task_sizes[:, None]
            costs *= machine_powers
            costs *= mu
            for i, t in enumerate(self.top_sort):
                self.graph.nodes[t]['weight'] = costs[i]  
                
        # Set the communication costs.
        # TODO: too slow.
        # Bandwidths.
        B = np.random.gamma(1/vband**2, vband**2, size=(self.nworkers, self.nworkers))
        B_rep_sum = sum(np.reciprocal(B[np.triu_indices(B.shape[0], k=1)])) 
        # Set the actual communication costs.
        ccrs = np.random.gamma(1/vccr**2, muccr*vccr**2, size=self.size)     
        for t in self.top_sort[:-1]: # Omit final task...
            expected_compute = self.task_average(t)
            # Choose data amount such that expected comm is met.
            D = (ccrs[t] * expected_compute*self.nworkers**2) / (2 * B_rep_sum) 
            for s in self.graph.successors(t):
                self.graph[t][s]['weight'] = {}
                for w in range(self.nworkers):
                    for p in range(w + 1, self.nworkers):
                        self.graph[t][s]['weight'][(w, p)] = D/B[w][p]                
        
                        
    def set_example_weights(self, nprocessors, max_cost=10):
        """
        Set weights randomly for small example graphs. Not used anywhere.

        Parameters
        ----------
        nprocessors : INT
            The number of processors.
        
        max_cost : INT/FLOAT
            The maximum cost.

        Returns
        -------
        None.
        """
        
        # Set DAG as weighted. 
        self.weighted = True
        self.workers = list(range(nprocessors))
        self.nworkers = nprocessors
        
        task_costs = np.random.randint(low=1, high=max_cost, size=(self.size, self.nworkers))
        for i, t in enumerate(self.top_sort):
            self.graph.nodes[t]['weight'] = task_costs[i] 
        
        for t in self.top_sort[:-1]:
            for s in self.graph.successors(t):
                self.graph[t][s]['weight'] = {}
                for w in range(self.nworkers):
                    for p in range(w + 1, self.nworkers):
                        self.graph[t][s]['weight'][(w, p)] = np.random.randint(low=1, high=max_cost)
        
        
    def comm_cost(self, parent, child, source, dest):
        """
        Get the communication/edge cost between parent and child when they are scheduled on source and dest (respectively).

        Parameters
        ----------
        parent : INT/STRING
            ID of parent task.
        child : INT/STRING
            ID of child task.
        source : INT/STRING
            ID of processor parent is scheduled on.
        dest : INT/STRING
            ID of processor child is scheduled on.

        Returns
        -------
        FLOAT
            The communication cost.
        
        Notes
        -------
        1. Assumed to be symmetric.
        """
        if source == dest:
            return 0.0
        elif source < dest:
            if self.comm_costs is not None:
                return self.comm_costs[(source, dest)]
            return self.graph[parent][child]['weight'][(source, dest)]
            # return self.graph[parent][child]['weight'] / self.bandwidths[(source, dest)]
        else:
            if self.comm_costs is not None:
                return self.comm_costs[(dest, source)]
            return self.graph[parent][child]['weight'][(dest, source)] # symmetric.  
            # return self.graph[parent][child]['weight'] / self.bandwidths[(dest, source)]
        
    def task_average(self, task, avg_type="M"):
        """
        Calculate average task weight. 
        """

        data = self.graph.nodes[task]['weight']       
        
        if avg_type in ["M", "m"]:
            return sum(data)/len(data)
        elif avg_type in ["MD", "md"]:
            return median(data)
        elif avg_type in ["B", "b", "SB", "sb"]:
            return min(data)
        elif avg_type in ["W", "w", "SW", "sw"]:
            return max(data)
        elif avg_type in ["HM", "hm", "SHM", "shm"]:
            return harmonic_mean(data) 
        elif avg_type in ["GM", "gm", "SGM", "sgm"]:
            return geometric_mean(data) 
        elif avg_type in ["R", "r"]:
            return max(data) / min(data)
        elif avg_type in ["D", "d"]:
            return max(data) - min(data)
        elif avg_type in ["NC", "nc"]:
            x, n = max(data), min(data)
            return (x - n) / (x/n) 
        elif avg_type in ["SD", "sd"]:
            return pstdev(data)
        elif avg_type in ["UCB", "ucb"]:
            m = sum(data)/len(data)
            sd = pstdev(data, xbar=m)
            return m - sd if m > sd else m - (sd/m)
        raise ValueError('Unrecognized avg_type!') 
        
    def edge_average(self, parent=None, child=None, avg_type="M"):
        """
        Calculate edge average (separate function needed to above).

        Parameters
        ----------
        parent : INT/STRING, optional
            ID of parent task. Needed for some average types. The default is None.
        child : INT/STRING, optional
            ID of child task. Needed for some average types. The default is None.
        avg_type : STRING, optional
            The type of average to use. The default is "M".

        Returns
        -------
        FLOAT
            The average edge cost.
        """                
        
        if self.comm_costs is not None:
            tri = self.comm_costs.values()
        else:
            assert (parent is not None and child is not None), 'No parent and child tasks specified!'
            tri = self.graph[parent][child]['weight'].values()
        
        if avg_type in ["SB", "sb", "SHM", "shm", "SGM", "sgm", "R", "r", "NC", "nc"]:
            return 0.0
        elif avg_type in ["M", "m"]:
            return 2 * sum(tri) / self.nworkers**2 
        elif avg_type in ["MD", "md"]:
            data = 2 * list(tri) + self.nworkers * [0.0]            
            return median(data)
        elif avg_type in ["B", "b"]:
            assert (parent is not None and child is not None), 'Need to specify parent and child for B avg_type!'
            wp = min(self.workers, key=lambda w:self.graph.nodes[parent]['weight'][w])
            wc = min(self.workers, key=lambda w:self.graph.nodes[child]['weight'][w])
            return self.comm_cost(parent, child, wp, wc)
        elif avg_type in ["SW", "sw", "D", "d"]:
            return max(tri)
        elif avg_type in ["W", "w"]:
            assert (parent is not None and child is not None), 'Need to specify parent and child for W avg_type!'
            wp = max(self.workers, key=lambda w:self.graph.nodes[parent]['weight'][w])
            wc = max(self.workers, key=lambda w:self.graph.nodes[child]['weight'][w])
            return self.comm_cost(parent, child, wp, wc)
        elif avg_type in ["HM", "hm"]: 
            assert (parent is not None and child is not None), 'Need to specify parent and child for HM avg_type!'
            s1 = sum(1/v for v in self.graph.nodes[parent]['weight'])
            s2 = sum(1/v for v in self.graph.nodes[child]['weight'])
            cbar = 0.0
            D = self.comm_costs if self.comm_costs is not None else self.graph[parent][child]['weight']
            for k, v in D.items():
                t_w = self.graph.nodes[parent]['weight'][k[0]]
                c_w = self.graph.nodes[child]['weight'][k[1]]             
                cbar += v/(t_w * c_w) 
            cbar *= 2 
            cbar /= (s1 * s2)
            return cbar 
        elif avg_type in ["GM", "gm"]:
            data = 2 * list(v + 1 for v in tri) + len(self.workers) * [1.0] 
            return geometric_mean(data) 
        elif avg_type in ["SD", "sd"]:
            data = 2 * list(tri) + self.nworkers * [0.0] 
            return pstdev(data)
        elif avg_type in ["UCB", "ucb"]:
            data = 2 * list(tri) + self.nworkers * [0.0] 
            m = sum(data)/len(data)
            sd = pstdev(data, xbar=m)
            return m - sd if m > sd else m - (sd/m)
        raise ValueError('Unrecognized avg_type!')  
                
    def get_upward_ranks(self, avg_type="M"): 
        """
        Compute upward ranks for all tasks according to the input average type.

        Parameters
        ---------- 
        avg_type : STRING, optional
            Type of average to use. The default is "M".

        Returns
        -------
        ranks : DICT
            {Task ID : rank}.
        """
        
        ranks = {}
        backward_traversal = list(reversed(self.top_sort))
                
        # If all edge averages the same...
        if self.comm_costs is not None and avg_type not in ["B", "b", "W", "w", "HM", "hm"]:
            edge_average = self.edge_average(avg_type=avg_type)
            for t in backward_traversal:
                ranks[t] = self.task_average(task=t, avg_type=avg_type)                
                try:
                    ranks[t] += (edge_average + max(ranks[s] for s in self.graph.successors(t)))
                except ValueError:
                    pass   
            return ranks         
        # Otherwise...
        for t in backward_traversal:
            ranks[t] = self.task_average(task=t, avg_type=avg_type)            
            try:
                ranks[t] += max(self.edge_average(parent=t, child=s, avg_type=avg_type) + ranks[s] for s in self.graph.successors(t))
            except ValueError:
                pass   
        return ranks 
    
    def get_downward_ranks(self, avg_type="M"): 
        """
        Compute downward ranks for all tasks according to the input average type.

        Parameters
        ---------- 
        avg_type : STRING, optional
            Type of average to use. The default is "M".

        Returns
        -------
        ranks : DICT
            {Task ID : rank}.
        """
        ranks = {}
        if self.comm_costs is not None and avg_type not in ["B", "b", "W", "w", "HM", "hm"]:
            edge_average = self.edge_average(avg_type=avg_type)
            for t in self.top_sort:
                ranks[t] = 0.0 
                try:
                    ranks[t] += edge_average + max(self.task_average(task=p, avg_type=avg_type) + ranks[p] for p in self.graph.predecessors(t))
                except ValueError:
                    pass 
        # Otherwise...
        for t in self.top_sort:
            ranks[t] = 0.0 
            try:
                ranks[t] += max(self.task_average(task=p, avg_type=avg_type) 
                                + self.edge_average(parent=p, child=t, avg_type=avg_type) 
                                + ranks[p] for p in self.graph.predecessors(t))
            except ValueError:
                pass   
        return ranks
    
    def optimistic_critical_path(self, pessimistic=False, return_path=False):
        """
        Modified version of Optimistic Cost Table defined in PEFT heuristic (without the edge average).
        """
        
        if return_path:
            path_info = {}
                
        C = {} 
        backward_traversal = list(reversed(self.top_sort))
        for task in backward_traversal:
            C[task] = {}
            if return_path:
                path_info[task] = {}
            for w in self.workers:
                C[task][w] = self.graph.nodes[task]['weight'][w] 
                child_values = {}
                for child in self.graph.successors(task):
                    worker_values = {v : C[child][v] + self.comm_cost(task, child, w, v) for v in self.workers}
                    chosen = max(worker_values, key=worker_values.get) if pessimistic else min(worker_values, key=worker_values.get)
                    child_values[child] = (chosen, worker_values[chosen])
                if not child_values: 
                    continue
                max_child = max(child_values, key=lambda s : child_values[s][1])
                C[task][w] += child_values[max_child][1]
                if return_path:
                    path_info[task][w] = (max_child, child_values[max_child][0])
        
        # Identify the path if specified.                
        if return_path:
            cp = self.top_sort[0]
            critical_worker = max(self.workers, key=C[task].get) if pessimistic else min(self.workers, key=C[task].get)
            path = [cp] # path=[(cp, critical_worker)]
            while True:
                cp, critical_worker = path_info[cp][critical_worker]
                path.append(cp) #path.append((cp, critical_worker))
                if cp == self.top_sort[-1]:
                    break
            return C, path        
        return C
    
    def makespan_lower_bound(self):
        """
        Computes a lower bound on the schedule makespan.

        Returns
        -------
        FLOAT
            The lower bound on the makespan.
        """  
        path_bound = min(self.optimistic_critical_path()[self.top_sort[0]].values())        
        min_work = sum(min(self.graph.nodes[t]['weight']) for t in self.top_sort)
        work_bound = min_work / self.nworkers        
        return max(path_bound, work_bound) 
    
    def minimal_serial_time(self):
        """
        Returns the minimal serial time (MST).

        Returns
        -------
        FLOAT
            The minimal serial time.
        """  
        
        minimal_worker_times = sum(self.graph.nodes[t]['weight'] for t in self.top_sort) # Assumes weights are numpy arrays.
        return min(minimal_worker_times)
        
    def ccr(self, avg_type="M"):
        """
        Returns the communication-to-computation ratio (CCR) of the DAG, as defined by Equation 2.2.

        Parameters
        ----------
        avg_type : STRING, optional
            The average type to use (e.g., arithmetic mean). The default is "M".

        Returns
        -------
        FLOAT
            The CCR.
        """ 
        
        exp_total_compute = sum(self.task_average(t, avg_type=avg_type) for t in self.top_sort)
        if self.comm_costs is not None and avg_type not in ["HM", "hm"]:
            exp_total_comm = self.graph.number_of_edges() * self.edge_average(avg_type=avg_type) 
        else:
            exp_total_comm = sum(self.edge_average(parent, child, avg_type=avg_type) for parent, child in self.graph.edges)
        return exp_total_comm/exp_total_compute
    
    def monte_carlo(self, realizations, pmf="A", times=False):
        """
        Monte Carlo method. 

        Parameters
        ----------
        realizations : INT
            The number of samples/realizations of the costs.
        pmf : STRING, optional
            The pmfs to use to sample the costs. Options are "A" or "H". The default is "A".
        times : BOOL, optional
            If True, return timing data. The default is False.

        Returns
        -------
        None.
        """
        
        if times:
            timing_data = {}
            start = timer()
        
        # Generate all realizations.
        R = {}
        for t in self.top_sort:
            if pmf in ["H", "h"]:   
                s = sum(1/v for v in self.graph.nodes[t]['weight'])
                R[t] = np.random.choice(self.graph.nodes[t]['weight'], size=realizations, p=[(1/v)/s for v in self.graph.nodes[t]['weight']])
            else:
                R[t] = np.random.choice(self.graph.nodes[t]['weight'], size=realizations)
        
        # Edges.
        if pmf in ["A", "a"]:
            probs = [2/self.nworkers**2] * ((self.nworkers*(self.nworkers - 1))//2) + [1/self.nworkers]
        for t, s in self.graph.edges: 
            if pmf in ["A", "a"]:
                data = list(self.graph[t][s]['weight'].values()) if self.comm_costs is None else list(self.comm_costs.values()) 
                data += [0.0]
            else:
                costs = self.comm_costs if self.comm_costs is not None else self.graph[t][s]['weight']     
                data, probs = [], []
                h = sum(1/v for v in self.graph.nodes[t]['weight'])*sum(1/v for v in self.graph.nodes[s]['weight'])
                for k, v in costs.items():
                    data.append(v)
                    t1 = self.graph.nodes[t]['weight'][k[0]]
                    s1 = self.graph.nodes[s]['weight'][k[1]]  
                    t2 = self.graph.nodes[t]['weight'][k[1]]
                    s2 = self.graph.nodes[s]['weight'][k[0]]                     
                    pr = 1/(t1*s1) + 1/(t2*s2)
                    probs.append(pr)
                # Now the zeroes...
                data.append(0.0)
                pr = sum(1/(self.graph.nodes[t]['weight'][w]*self.graph.nodes[s]['weight'][w]) for w in self.workers)
                probs.append(pr)
                # Divide through to normalize...
                probs = [p/h for p in probs]
            # Set the realizations for each edge. (Re-done even when self.comm_costs is not None and pmf == "M").
            R[(t, s)] = np.random.choice(data, size=realizations, p=probs)    

        if times:
            elapsed = timer() - start
            timing_data["REAL"] = elapsed  
            start = timer()
                    
        # Longest path distributions downward of all tasks.
        L = {}
        backward_traversal = list(reversed(self.top_sort))
        for t in backward_traversal:
            children = list(self.graph.successors(t))
            if not children:
                L[t] = R[t] 
                continue
            pmatrix = [np.add(L[s], R[(t, s)]) for s in children]
            st = np.amax(pmatrix, axis=0)
            L[t] = np.add(R[t], st) 
        
        if times:
            elapsed = timer() - start
            timing_data["LONG"] = elapsed 
            start = timer()            
        
        # # Count the number of times each longest path occurred - and criticalities.
        path_counts = {}
        critical_counts = {t : 0 for t in self.top_sort}
        for r in range(realizations):
            cp = self.top_sort[0]
            critical_counts[cp] += 1
            path = [cp]
            while cp != self.top_sort[-1]:
                for child in self.graph.successors(cp):
                    if abs(L[child][r] + R[(cp, child)][r] + R[cp][r] - L[cp][r]) < 1e-6: 
                        cp = child
                        critical_counts[cp] += 1
                        path.append(cp)
                        break
            tpath = tuple(path)
            try:
                path_counts[tpath] += 1 
            except KeyError:
                path_counts[tpath] = 1
        criticalities = {t : critical_counts[t]/realizations for t in self.top_sort}
                
        if times:
            elapsed = timer() - start
            timing_data["CRT"] = elapsed 
            return L, path_counts, criticalities, timing_data
        
        return L, path_counts, criticalities
                              
   
# =============================================================================
# HEURISTICS.
# =============================================================================

def priority_scheduling(G,
                    priorities, 
                    critical_assignment=None,
                    return_schedule=False):
    """
    Simulates the scheduling of the task graph according to the task priorities.

    Parameters
    ----------
    G : DAG
        Task DAG.
    priorities : DICT
        Task priorities, {task ID : priority}.  
    critical_assignment : DICT, optional
        An assignment of tasks to processors (or just processor types), {task ID : processor or processor type}. The default is None.
    return_schedule : BOOL, optional
        If True, return the schedule. The default is False.

    Returns
    -------
    mkspan : FLOAT
        The schedule makespan.
    schedule : DICT, optional
        If return_schedule == True. {Worker ID : [(task, start time, finish time), ...], ...}.
    """
        
    # Build schedule. Keep track of finish times and where tasks are scheduled.
    schedule = {w : [] for w in G.workers}
    finish_times, where = {}, {}
    
    # Initialize ready tasks - assumes single entry task.
    ready_tasks = [G.top_sort[0]] 
    # Start the simulation.
    while len(ready_tasks): 
        
        # Get the highest priority task.
        task = max(ready_tasks, key=priorities.get)
        
        # Find parents.
        parents = list(G.graph.predecessors(task))
        
        # Simulate scheduling on each worker.
        worker_finish_times = {}
        for w in G.workers:
            if critical_assignment is not None and (task in critical_assignment) and critical_assignment[task] != w:
                continue               
            
            task_cost = G.graph.nodes[task]['weight'][w]  
            
            # Find the data-ready time.       
            drt = 0.0 if not parents else max(finish_times[p] + G.comm_cost(p, task, where[p], w) for p in parents) 
            
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
                    
        # Find the best worker.
        best_worker = min(worker_finish_times, key=lambda w:worker_finish_times[w][1])        
        # Schedule the task on best_worker and update the helper dicts.
        where[task] = best_worker            
        st, ft, idx = worker_finish_times[best_worker]
        finish_times[task] = ft            
        if not schedule[best_worker] or idx < 0:             
            schedule[best_worker].append((task, st, ft))  
        else: 
            schedule[best_worker].insert(idx, (task, st, ft)) 
            
        # Update ready tasks.
        ready_tasks.remove(task)
        for c in G.graph.successors(task):
            if all(p in where for p in G.graph.predecessors(c)):
                ready_tasks.append(c)      
    
    # Return makespan/schedule.
    mkspan = finish_times[G.top_sort[-1]] # Assumes single exit task.
    if return_schedule:
        return mkspan, schedule    
    return mkspan

def heft(G, avg_type="M", return_schedule=False):
    """
    Heterogeneous Earliest Finish Time (HEFT).

    Parameters
    ----------
    G : DAG
        Task DAG.
    avg_type : STRING, optional
        Type of average to use. The default is "M" (arithmetic mean).   
    return_schedule : BOOL, optional
        If True, return the schedule. The default is False.

    Returns
    -------
    mkspan : FLOAT
        The schedule makespan.
    schedule : DICT, optional
        If return_schedule == True. {Worker ID : [(task, start time, finish time), ...], ...}.
    """
    # Compute upward ranks.
    U = G.get_upward_ranks(avg_type=avg_type)
    # Simulate to get the schedule and return it.
    return priority_scheduling(G, priorities=U, return_schedule=return_schedule)

def cpop(G, priorities, critical_path, return_schedule=False):
    """
    Critical Path on a Processor (CPOP).
    Note always used in contexts with path and priorities already computed.

    Parameters
    ----------
    G : DAG
        Task DAG.
    priorities : DICT
        Task priorities, {task ID : priority}.  
    critical_path : LIST
        Tasks on the critical path.
    return_schedule : BOOL, optional
        If True, return the schedule. The default is False.

    Returns
    -------
    mkspan : FLOAT
        The schedule makespan.
    schedule : DICT, optional
        If return_schedule == True. {Worker ID : [(task, start time, finish time), ...], ...}.
    """     
        
    # Decide where to schedule critical tasks.
    worker_serial_time = lambda w : sum(G.graph.nodes[cp]['weight'][w] for cp in critical_path)
    critical_worker = min(G.workers, key=worker_serial_time)
    alpha = {cp : critical_worker for cp in critical_path}        
    # Simulate to get the schedule and return it.
    return priority_scheduling(G, priorities=priorities, critical_assignment=alpha, return_schedule=return_schedule)