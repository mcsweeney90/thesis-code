#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes and functions for accelerated scheduling simulation model. 
"""

import networkx as nx
from random import uniform, gammavariate as gamma 
from statistics import median
from math import sqrt
from copy import deepcopy

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
        
    def set_cholesky_weights(self, timings, nb=1024):
        """
        Set weights for Cholesky DAGs. 

        Parameters
        ----------
        timings : DICT
            See chol_graphs directory.
        nb : INT, optional
            Tile size. The default is 1024.

        Returns
        -------
        None.

        """              
        runs = len(timings[nb]["G"]["c"]) # Assumes they're all the same length.        
        for task in self.top_sort:
            task_type = task[0] 
            c, g = sum(timings[nb][task_type]["c"])/runs, sum(timings[nb][task_type]["g"])/runs
            self.graph.nodes[task]['weight'] = {"c" : c, "g" : g}
            # Edges.
            for child in self.graph.successors(task):
                child_type = child[0]
                d = sum(timings[nb][child_type]["d"])/runs 
                self.graph[task][child]['weight'] = d
        
    def set_random_weights(self, r, s, gpu_mean=1.0, gpu_cov=1.0, cpu_mean=15.0, cpu_cov=1.0, ccr=1.0):
        """
        Set randomly generated weights (for STG graphs).

        Parameters
        ----------
        r : INT
            Number of CPUs.
        s : INT
            Number of GPUs.
        gpu_mean : FLOAT, optional
            Mean GPU task execution time. The default is 1.0.
        gpu_cov : FLOAT, optional
            Coefficient of variation of GPU execution time distribution. The default is 1.0.
        cpu_mean : FLOAT, optional
            Mean CPU task execution time. The default is 15.0.
        cpu_cov : TYPE, optional
            Coefficient of variation of CPU execution time distribution. The default is 1.0.
        ccr : FLOAT, optional
            Target computation-to-communication ratio of DAG. Defined as sum of all mean edge weights over sum of all mean task weights. 
            The default is 1.0.

        Returns
        -------
        None.

        """
        
        # Determine shape and scale parameters for the CPU and GPU distributions.
        gpu_shape, gpu_scale = 1/gpu_cov**2, gpu_mean * gpu_cov**2  
        cpu_shape, cpu_scale = 1/cpu_cov**2, cpu_mean * cpu_cov**2  
        # Calculate the mean "granularity".
        mean_granularity = (self.size * ccr) / self.graph.number_of_edges() 
        
        # Set the weights.
        for task in self.top_sort:
            c, g = gamma(alpha=cpu_shape, beta=cpu_scale), gamma(alpha=gpu_shape, beta=gpu_scale)
            self.graph.nodes[task]['weight'] = {"c" : c, "g" : g}
            avg_comp = average(c, g, r=r, s=s, avg_type="M")           
            
            # Set communication costs.
            gran = uniform(0, 2) * mean_granularity            
            d = (gran*avg_comp*(r + s)**2) / (s * (2*r + s - 1))
            for child in self.graph.successors(task):
                self.graph[task][child]['weight'] = d                
            
    
    def edge_average(self, parent, child, r=1, s=1, avg_type="M"):
        """
        Calculates the average weight of edge from parent to child.

        Parameters
        ----------
        parent : INT/STRING
            ID of parent task.
        child : INT/STRING
            ID of child task.
        r : INT, optional
            Number of CPUs. The default is 1.
        s : INT, optional
            Number of GPUs. The default is 1.
        avg_type : STRING, optional
            Type of average to use. The default is "M".

        Raises
        ------
        ValueError
            If avg_type not recognized.

        Returns
        -------
        FLOAT
            The average weight of the edge.
            
        Notes
        ----------
        Couldn't just use average function (see below) because of how edge weights are stored. Also more efficient.         
        
        """
        
        d = self.graph[parent][child]['weight']
        
        if avg_type in ["M", "m"]:
            q = r + s
            return (d * s * (2*r + s - 1)) / q**2 
        elif avg_type in ["MD", "md"]:
            return median((r**2 + s) * [0.0] + s * (2*r + s - 1) * [d])
        elif avg_type in ["B", "b"]:
            ci, gi = self.graph.nodes[parent]['weight']["c"], self.graph.nodes[parent]['weight']["g"]
            ck, gk = self.graph.nodes[child]['weight']["c"], self.graph.nodes[child]['weight']["g"]
            return 0.0 if (ci <= gi and ck <= gk) or (gi <= ci and gk <= ck) else d
        elif avg_type in ["SB", "sb"]:
            return 0.0
        elif avg_type in ["W", "w"]:
            ci, gi = self.graph.nodes[parent]['weight']["c"], self.graph.nodes[parent]['weight']["g"]
            ck, gk = self.graph.nodes[child]['weight']["c"], self.graph.nodes[child]['weight']["g"]
            return 0.0 if (ci <= gi and ck <= gk) else d
        elif avg_type in ["SW", "sw"]:
            return d
        elif avg_type in ["HM", "hm"]: 
            ci, gi = self.graph.nodes[parent]['weight']["c"], self.graph.nodes[parent]['weight']["g"]
            ck, gk = self.graph.nodes[child]['weight']["c"], self.graph.nodes[child]['weight']["g"]
            num = d * r * s * (ci*gk + ck*gi) + d*ci*ck*s*(s - 1)
            denom = (r*gi + s*ci) * (r*gk + s*ck)
            return num/denom
        elif avg_type in ["SHM", "shm"]:
            return 0.0 
        elif avg_type in ["GM", "gm"]:
            return pow(d + 1, (s*(2*r + s - 1))/(r + s))
        elif avg_type in ["SGM", "sgm"]:
            return 0.0
        elif avg_type in ["R", "r", "D", "d", "NC", "nc"]:
            return 0.0
        elif avg_type in ["SD", "sd"]:
            q = r + s
            m = (d * s * (2*r + s - 1)) / q**2
            return (sqrt(s*(2*r + s - 1)*(d - m)**2 + (r**2 + s)*m*m)) / q
        elif avg_type in ["UCB", "ucb"]: # Didn't include this in the end.
            q = r + s
            m = (d * s * (2*r + s - 1)) / q**2
            sd = (sqrt(s*(2*r + s - 1)*(d - m)**2 + (r**2 + s)*m*m)) / q
            return m - sd if m > sd else m - (sd/m)
        raise ValueError('Unrecognized avg_type!')
        
    def get_upward_ranks(self, r, s, avg_type="M"):
        """
        Compute upward ranks for all tasks according to the input average type.

        Parameters
        ----------
        r : INT, optional
            Number of CPUs. 
        s : INT, optional
            Number of GPUs. 
        avg_type : STRING, optional
            Type of average to use. The default is "M".

        Returns
        -------
        ranks : DICT
            {Task ID : rank}.

        """
        ranks = {}
        backward_traversal = list(reversed(self.top_sort))
        for t in backward_traversal:
            c, g = self.graph.nodes[t]['weight']["c"], self.graph.nodes[t]['weight']["g"]
            ranks[t] = average(c, g, r, s, avg_type=avg_type)
            try:
                ranks[t] += max(self.edge_average(t, c, r, s, avg_type=avg_type) + ranks[c] for c in self.graph.successors(t))
            except ValueError:
                pass   
        return ranks  
    
    def get_downward_ranks(self, r, s, avg_type="M"):
        """
        Compute downward ranks for all tasks according to the input average type.

        Parameters
        ----------
        r : INT, 
            Number of CPUs. 
        s : INT,
            Number of GPUs. 
        avg_type : STRING, optional
            Type of average to use. The default is "M".

        Returns
        -------
        ranks : DICT
            {Task ID : rank}.

        """
        ranks = {}
        for t in self.top_sort:
            ranks[t] = 0.0 
            try:
                ranks[t] += max(average(self.graph.nodes[p]['weight']["c"], self.graph.nodes[p]['weight']["g"], r, s, avg_type=avg_type) 
                                + self.edge_average(p, t, r, s, avg_type=avg_type) + ranks[p] for p in self.graph.predecessors(t))
            except ValueError:
                pass   
        return ranks              
    
    def orig_optimistic_cost_table(self, r, s):
        """
        Optimistic cost table used in Predict Earlier Finish Time (PEFT) heuristic by Arabnejad and [citation].
        Original version that uses average weights for edge.

        Parameters
        ----------
        r : INT, 
            Number of CPUs. 
        s : INT, 
            Number of GPUs. 

        Returns
        -------
        OCT : DICT
            {Task ID : {Worker ID : optimistic cost, ...}, ...}.

        """
        
        worker_types = ["c", "g"]
        delta = lambda x, y : 0.0 if x == y else 1.0         
        OCT = {} 
        backward_traversal = list(reversed(self.top_sort))
        for task in backward_traversal:
            OCT[task] = {}
            for w in worker_types:
                OCT[task][w] = 0.0
                child_values = []
                for child in self.graph.successors(task):
                    action_values = [OCT[child][v] + delta(w, v) * self.edge_average(task, child, r, s, avg_type="M")
                                     + self.graph.nodes[child]['weight'][v] for v in worker_types]
                    child_values.append(min(action_values))   
                OCT[task][w] += max(child_values) if len(child_values) else 0.0 
        return OCT
    
    def optimistic_cost_table(self, include_current=False):
        """
        Optimistic cost table used in Predict Earlier Finish Time (PEFT) heuristic.
        Alterative version that uses actual edge cost rather than average.

        Parameters
        ----------
        include_current : BOOL, optional
            Include current task weight when calculating optimistic future costs (True for priorities, False for selection). The default is False.

        Returns
        -------
        OCT : DICT
            {Task ID : {Worker ID : optimistic cost, ...}, ...}.
            
        Notes
        -------
        Unlike above, r and s not required since don't use average values for edge weight.

        """
        
        worker_types = ["c", "g"]
        delta = lambda x, y : 0.0 if x == y else 1.0 
        
        OCT = {} 
        backward_traversal = list(reversed(self.top_sort))
        for task in backward_traversal:
            OCT[task] = {}
            for w in worker_types:
                OCT[task][w] = self.graph.nodes[task]['weight'][w] if include_current else 0.0
                child_values = []
                for child in self.graph.successors(task):
                    if include_current:
                        action_values = [OCT[child][v] + delta(w, v) * self.graph[task][child]['weight'] for v in worker_types]
                    else:
                        action_values = [OCT[child][v] + delta(w, v) * self.graph[task][child]['weight'] 
                                         + self.graph.nodes[child]['weight'][v] for v in worker_types]
                    child_values.append(min(action_values))   
                OCT[task][w] += max(child_values) if len(child_values) else 0.0 
        return OCT
    
    def get_critical_path(self, cp_type="AVG", avg_type="M", r=1, s=1, all_critical_tasks=False):
        """
        Identifies and returns critical path.

        Parameters
        ----------
        cp_type : STRING, optional
            Specifies the method used - either 'AVG' (averaging the weights) or 'OPT' (optimistic critical path). The default is "AVG".
        r : INT, optional
            Number of CPUs. The default is 1.
        s : INT, optional
            Number of GPUs. The default is 1.
        avg_type : STRING, optional
            Type of average to use if cp_type == "AVG". The default is "M" (arithmetic mean).
        all_critical_tasks : BOOL, optional
            If True and there are multiple critical paths, returns all of the critical tasks. The default is False.

        Returns
        -------
        critical_path : LIST
            List of critical tasks.

        """
        
        if cp_type in ["avg", "AVG"]:
            U = self.get_upward_ranks(r, s, avg_type=avg_type)
            D = self.get_downward_ranks(r, s, avg_type=avg_type)
            ranks = {t : U[t] + D[t] for t in self.top_sort}   
                        
            # Identify a single critical path (unless all_critical_tasks) - randomly if there are multiple...
            cp_length = ranks[self.top_sort[0]] # Single entry/exit task.
            if all_critical_tasks:
                return list(t for t in self.top_sort if abs(ranks[t] - cp_length) < 1e-6)
            ct = self.top_sort[0]
            critical_path = [ct]
            while True:
                children = list(self.graph.successors(ct))
                if not children:
                    break
                for child in children:
                    if abs(ranks[child] - cp_length) < 1e-6:
                        critical_path.append(child)
                        ct = child
                        break
            return critical_path
        
        elif cp_type in ["opt", "OPT"]:
            optimistic_costs = self.optimistic_cost_table(include_current=True)   
            delta = lambda x, y : 0.0 if x == y else 1.0 
            cp = self.top_sort[0]
            if optimistic_costs[cp]["c"] < optimistic_costs[cp]["g"]:
                target = optimistic_costs[cp]["c"]
                proc_type = "c"
            else:
                target = optimistic_costs[cp]["g"]
                proc_type = "g"
            critical_path = [cp]      
            while True:
                children = list(self.graph.successors(cp))
                if not children:
                    break
                for child in children:
                    if abs(self.graph.nodes[cp]['weight'][proc_type] + optimistic_costs[child]["c"] 
                           + delta(proc_type, "c") * self.graph[cp][child]['weight'] - target) < 1e-6:
                        target = optimistic_costs[child]["c"]
                        proc_type = "c"
                        critical_path.append(child)
                        cp = child
                        break
                    elif abs(self.graph.nodes[cp]['weight'][proc_type] +optimistic_costs[child]["g"] 
                             + delta(proc_type, "g") * self.graph[cp][child]['weight'] - target) < 1e-6:
                        target = optimistic_costs[child]["g"]
                        proc_type = "g"
                        critical_path.append(child)
                        cp = child
                        break
            return critical_path
            
    def makespan_lower_bound(self, q):
        """
        Computes a lower bound on the schedule makespan.

        Parameters
        ----------
        q : INT
            Total number of processors.

        Returns
        -------
        FLOAT
            The lower bound on the makespan.

        """      
        path_bound = min(self.optimistic_cost_table(include_current=True)[self.top_sort[0]].values()) 
        min_work = sum(min(self.graph.nodes[t]['weight'].values()) for t in self.top_sort)
        work_bound = min_work / q
        return max(path_bound, work_bound) 
    
    def minimal_serial_time(self):
        """
        Returns the minimal serial time (MST).

        Returns
        -------
        FLOAT
            The minimal serial time.

        """       
        cpu_time = sum(self.graph.nodes[t]['weight']["c"] for t in self.top_sort)
        gpu_time = sum(self.graph.nodes[t]['weight']["g"] for t in self.top_sort)
        return min(cpu_time, gpu_time)
    
    def ccr(self, r, s, avg_type="M"):
        """
        Returns the communication-to-computation ratio (CCR) of the DAG, as defined by Equation 2.2.

        Parameters
        ----------
        r : INT
            Number of CPUs.
        s : INT
            Number of GPUs.
        avg_type : STRING, optional
            The average type to use (e.g., arithmetic mean). The default is "M".

        Returns
        -------
        FLOAT
            The CCR.

        """                
        exp_total_compute = sum(average(c=self.graph.nodes[t]['weight']["c"], g=self.graph.nodes[t]['weight']["g"], r=r, s=s, avg_type=avg_type) 
                                for t in self.top_sort)
        exp_total_comm = sum(self.edge_average(parent, child, r, s, avg_type=avg_type) for parent, child in self.graph.edges)
        return exp_total_comm/exp_total_compute
        

# =============================================================================
# HELPER FUNCTIONS.
# =============================================================================

def average(c, g, r=1, s=1, avg_type="M"):
    """
    Quick average calculator when we have a set with r duplicates of c and s duplicates of g.
    (Used for task averages.)

    Parameters
    ----------
    c : INT/FLOAT
        CPU execution time.
    g : INT/FLOAT
        GPU execution time.
    r : INT, optional
        Number of CPUs/multiplicity of c. Not needed for all average types. The default is 1.
    s : INT, optional
        Number of GPUs/multiplicity of g. Not needed for all average types. The default is 1.
    avg_type : STRING, optional
        The average type to use (e.g., arithmetic mean). The default is "M".

    Raises
    ------
    ValueError
        Unrecognized avg_type.

    Returns
    -------
    FLOAT
        The average value.

    """
    
    if avg_type in ["M", "m"]:
        return (r * c + s * g) / (r + s)
    elif avg_type in ["MD", "md"]:
        return median(r * [c] + s * [g])
    elif avg_type in ["B", "b", "SB", "sb"]:
        return min(c, g)
    elif avg_type in ["W", "w", "SW", "sw"]:
        return max(c, g)
    elif avg_type in ["HM", "hm", "SHM", "shm"]:
        q = r + s
        return (q * c * g) / (r * g + s * c)
        # return harmonic_mean(r * [c] + s * [g]) # Hard coding is faster.
    elif avg_type in ["GM", "gm", "SGM", "sgm"]:
        return pow(pow(c, r) * pow(g, s), 1/(r + s))
        # return geometric_mean(r * [c] + s * [g]) 
    elif avg_type in ["R", "r"]:
        return max(c, g) / min(c, g)
    elif avg_type in ["D", "d"]:
        return max(c, g) - min(c, g)
    elif avg_type in ["NC", "nc"]:
        x, n = max(c, g), min(c, g)
        return (x - n) / (x/n) 
    elif avg_type in ["SD", "sd"]:
        m = (r * c + s * g) / (r + s)
        return sqrt((r*(c - m)**2 + s*(g - m)**2)/(r + s))
    elif avg_type in ["UCB", "ucb"]:
        m = (r * c + s * g) / (r + s)
        sd = sqrt((r*(c - m)**2 + s*(g - m)**2)/(r + s))
        return m - sd if m > sd else m - (sd/m)
    raise ValueError('Unrecognized avg_type!')    

# =============================================================================
# HEURISTICS.
# =============================================================================

def priority_scheduling(G, r, s,
                    priorities, 
                    sel_policy="EFT", 
                    lookahead_table=None,
                    assignment=None,
                    cross_thresh=0.3,
                    return_schedule=False):
    """
    Simulates the scheduling of the task graph according to the task priorities.

    Parameters
    ----------
    G : DAG
        Task DAG.
    r : INT
        Number of CPUs. 
    s : INT
        Number of GPUs.
    priorities : DICT
        Task priorities, {task ID : priority}.   
    sel_policy : STRING, optional
        Processor selection rule. Default is "EFT" (earliest finish time).
    lookahead_table : DICT, optional
        Used if sel_policy == "PEFT".
    assignment : DICT, optional
        An assignment of tasks to processors (or just processor types), {task ID : processor or processor type}. Used if sel_policy == "AMT".
    cross_thresh : FLOAT, optional
        Cross threshold parameter used in HEFT-NC heuristic. Used if sel_policy == "NC". The default is 0.3.
    return_schedule : BOOL, optional
        If True, return the schedule. The default is False.

    Returns
    -------
    mkspan : FLOAT
        The schedule makespan.
    schedule : DICT, optional
        If return_schedule == True. {Worker ID : [(task, start time, finish time), ...], ...}.

    """
    
    delta = lambda source, dest: 0.0 if (source == dest) or (source < r and dest < r) else 1.0
    if sel_policy in ["BL", "FL"]:
        last = lambda L : L[-1][2] if len(L) else 0.0
    
    # Create list of workers. Convention followed below is that 0,.., (r - 1) are CPUs and r,..., (r + s - 1) the GPUs.
    workers = list(range(r + s))
    
    # Build schedule. Keep track of finish times and where tasks are scheduled.
    schedule = {w : [] for w in workers}
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
        for w in workers:
            if sel_policy == "AMT" and (task in assignment) and isinstance(assignment[task], int) and assignment[task] != w:
                continue
            elif sel_policy == "AMT" and (task in assignment) and assignment[task] == "c" and w == r:
                break
            elif sel_policy == "AMT" and (task in assignment) and assignment[task] == "g" and w < r:
                continue                
            
            task_cost = G.graph.nodes[task]['weight']["c"] if w < r else G.graph.nodes[task]['weight']["g"] 
            
            # Find the data-ready time.       
            drt = 0.0 if not parents else max(finish_times[p] + delta(where[p], w)*G.graph[p][task]['weight'] for p in parents)
            
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
        if sel_policy == "EFT":
            best_worker = min(workers, key=lambda w:worker_finish_times[w][1])
        elif sel_policy == "AMT":
            best_worker = min(worker_finish_times, key=lambda w:worker_finish_times[w][1])
        elif sel_policy == "NC": 
            best_cpu = min(workers[:r], key=lambda w:worker_finish_times[w][1])
            best_gpu = min(workers[r:], key=lambda w:worker_finish_times[w][1])
            weight_min = "c" if G.graph.nodes[task]['weight']["c"] < G.graph.nodes[task]['weight']["g"] else "g"
            finish_min = "c" if worker_finish_times[best_cpu][1] < worker_finish_times[best_gpu][1] else "g"
            if weight_min == finish_min:
                best_worker = best_cpu if weight_min == "c" else best_gpu
            else:
                if finish_min == "c":
                    fa, fb = worker_finish_times[best_cpu][1], worker_finish_times[best_gpu][1]
                else:
                    fa, fb = worker_finish_times[best_gpu][1], worker_finish_times[best_cpu][1]
                N = (fb - fa) / (fb/fa)
                wbar = average(c=G.graph.nodes[task]['weight']["c"], g=G.graph.nodes[task]['weight']["g"], r=r, s=s, avg_type="NC")
                if wbar/N < cross_thresh:
                    best_worker = best_cpu if finish_min == "c" else best_gpu
                else:
                    best_worker = best_cpu if weight_min == "c" else best_gpu
             
        elif sel_policy == "PEFT":
            # Compare best CPU and GPU workers.
            best_cpu = min(workers[:r], key=lambda w:worker_finish_times[w][1])
            best_gpu = min(workers[r:], key=lambda w:worker_finish_times[w][1])
            if worker_finish_times[best_cpu][1] + lookahead_table[task]["c"] < worker_finish_times[best_gpu][1] + lookahead_table[task]["g"]:
                best_worker = best_cpu
            else:
                best_worker = best_gpu 
        
        elif sel_policy == "GCP":
            # Compare best CPU and GPU workers.
            best_cpu = min(workers[:r], key=lambda w:worker_finish_times[w][1])
            best_gpu = min(workers[r:], key=lambda w:worker_finish_times[w][1]) 
            if worker_finish_times[best_cpu][1] < worker_finish_times[best_gpu][1]:
                best_worker = best_cpu
            elif task == G.top_sort[-1]: # TODO: ugly...
                best_worker = best_gpu
            else:       
                gpu_saving = worker_finish_times[best_cpu][1] - worker_finish_times[best_gpu][1]                
                exp_comm_penalty = ((r-1)/(r+s))*max(G.graph[task][child]['weight'] for child in G.graph.successors(task))
                if gpu_saving < exp_comm_penalty:
                    best_worker = best_cpu
                else:
                    best_worker = best_gpu
        
        elif sel_policy == "HAL":
            # Compare best CPU and GPU workers.
            best_cpu = min(workers[:r], key=lambda w:worker_finish_times[w][1])
            best_gpu = min(workers[r:], key=lambda w:worker_finish_times[w][1]) 
            
            # Calculate expected penalties.
            xc, xg = 0.0, 0.0
            for child in G.graph.successors(task):
                c, g = G.graph.nodes[child]['weight']["c"], G.graph.nodes[child]['weight']["g"]
                d = G.graph[task][child]['weight']
                sxc = (r + s)*c*(d + g)
                sxc /= (r*(d + g) + s*c)
                xc = max(xc, sxc)
                # Same for GPU.
                sxg = (r + s) * g * (c + d) * (g + d)
                sxg /= (r*g*g + d**2 + (r + s)*d*g + s*g*c + c*d)
                xg = max(xg, sxg)
            
            # Now compare...
            if worker_finish_times[best_cpu][1] + xc < worker_finish_times[best_gpu][1] + xg:
                best_worker = best_cpu
            else:
                best_worker = best_gpu
                
        elif sel_policy in ["FL", "BL"]: # TODO: slow. 
            children = list(sorted(G.graph.successors(task), key=priorities.get))
            if sel_policy == "FL":
                poss_workers = workers
            else:
                best_cpu = min(workers[:r], key=lambda w:worker_finish_times[w][1])
                best_gpu = min(workers[r:], key=lambda w:worker_finish_times[w][1])
                poss_workers = [best_cpu, best_gpu]
            
            lk_mkspans = {}
            for bw in poss_workers:
                # Make copy of schedule and finish time dicts.
                pschedule = deepcopy(schedule)
                pfinish_times = deepcopy(finish_times)
                pwhere = deepcopy(where)
                # Add to the schedule copy.
                st, ft, idx = worker_finish_times[bw]
                pwhere[task] = bw
                pfinish_times[task] = ft  
                if not pschedule[bw] or idx < 0:             
                    pschedule[bw].append((task, st, ft))  
                else: 
                    pschedule[bw].insert(idx, (task, st, ft)) 
                # Consider each child.
                for child in children:
                    cworker_finish_times = {}#
                    valid_parents = list(p for p in G.graph.predecessors(child) if p in pwhere)
                    for w in workers:              
                        
                        child_cost = G.graph.nodes[child]['weight']["c"] if w < r else G.graph.nodes[child]['weight']["g"] 
                        
                        # Find the data-ready time.   
                        cdrt = max(pfinish_times[vp] + delta(pwhere[vp], w)*G.graph[vp][child]['weight'] for vp in valid_parents) 
                        
                        # Find time worker can actually execute the task (assumes insertion). 
                        if not pschedule[w]:
                            cworker_finish_times[w] = (cdrt, cdrt + child_cost, 0)
                        else:
                            found, prev_finish_time = False, 0.0
                            for i, t in enumerate(pschedule[w]):
                                if t[1] < cdrt:
                                    prev_finish_time = t[2]
                                    continue
                                poss_start_time = max(prev_finish_time, cdrt) 
                                poss_finish_time = poss_start_time + child_cost
                                if poss_finish_time <= t[1]:
                                    found = True
                                    cworker_finish_times[w] = (poss_start_time, poss_finish_time, i)                            
                                    break
                                prev_finish_time = t[2]    
                            # No valid gap found.
                            if not found:
                                st = max(pschedule[w][-1][2], cdrt)
                                cworker_finish_times[w] = (st, st + child_cost, -1) 
                    
                    bcw = min(workers, key=lambda w:cworker_finish_times[w][1])
                    pwhere[child] = bcw           
                    cst, cft, cidx = cworker_finish_times[bcw]
                    pfinish_times[child] = cft            
                    if not pschedule[bcw] or cidx < 0:             
                        pschedule[bcw].append((child, cst, cft))  
                    else: 
                        pschedule[bcw].insert(idx, (child, cst, cft)) 
                # After all children have been scheduled, get new makespan.
                lk_mkspans[bw] = max(last(load) for load in pschedule.values())
            # Choose the best worker.
            best_worker = min(lk_mkspans, key=lk_mkspans.get)
        
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

def heft(G, r, s, avg_type="M", sel_policy="EFT", return_schedule=False):
    """
    Heterogeneous Earliest Finish Time (HEFT).

    Parameters
    ----------
    G : DAG
        Task DAG.
    r : INT
        Number of CPUs. 
    s : INT
        Number of GPUs.
    avg_type : STRING, optional
        Type of average to use. The default is "M" (arithmetic mean).    
    sel_policy : STRING, optional
        Processor selection rule. Default is "EFT" (earliest finish time).
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
    U = G.get_upward_ranks(r, s, avg_type=avg_type)
    # Simulate to get the schedule and return it.
    if return_schedule:
        return priority_scheduling(G, r, s, priorities=U, sel_policy=sel_policy, return_schedule=True)
    return priority_scheduling(G, r, s, priorities=U, sel_policy=sel_policy)

def peft(G, r, s, original=False, avg_type="M", return_schedule=False):
    """
    Predict Earliest Finish Time (PEFT).

    Parameters
    ----------
    G : DAG
        Task DAG.
    r : INT
        Number of CPUs. 
    s : INT
        Number of GPUs.
    original : BOOL, optional
        Type of average to use (only needed for . The default is False. 
    avg_type : STRING, optional
        Type of average to use for task priorities (and OCT if original == True). The default is "M" (arithmetic mean).    
    return_schedule : BOOL, optional
        If True, return the schedule. The default is False.

    Returns
    -------
    mkspan : FLOAT
        The schedule makespan.
    schedule : DICT, optional
        If return_schedule == True. {Worker ID : [(task, start time, finish time), ...], ...}.

    """
    
    # Compute optimistic cost table and ranks.
    if original:
        OCT = G.orig_optimistic_cost_table(r, s)
        ranks = {t : average(OCT[t]["c"], OCT[t]["g"], r=r, s=s, avg_type="M") for t in G.top_sort}
    else:
        OCT = G.optimistic_cost_table() 
        ranks = {t : average(OCT[t]["c"] + G.graph.nodes[t]['weight']["c"], OCT[t]["g"] + G.graph.nodes[t]['weight']["g"], 
                             r=r, s=s, avg_type=avg_type) for t in G.top_sort} 
                
    if return_schedule:
        return priority_scheduling(G, r, s, priorities=ranks, sel_policy="PEFT", lookahead_table=OCT, return_schedule=True)
    return priority_scheduling(G, r, s, priorities=ranks, sel_policy="PEFT", lookahead_table=OCT)
    
def cpop(G, r, s, avg_type="M", return_schedule=False):
    """
    Critical Path on a Processor (CPOP).

    Parameters
    ----------
    G : DAG
        Task DAG.
    r : INT
        Number of CPUs. 
    s : INT
        Number of GPUs.
    avg_type : STRING, optional
        Type of average to use to identify critical path. The default is "M" (arithmetic mean).  
    return_schedule : BOOL, optional
        If True, return the schedule. The default is False.

    Returns
    -------
    mkspan : FLOAT
        The schedule makespan.
    schedule : DICT, optional
        If return_schedule == True. {Worker ID : [(task, start time, finish time), ...], ...}.

    """
    
    # Compute upward and downward ranks.
    U = G.get_upward_ranks(r, s, avg_type=avg_type)
    D = G.get_downward_ranks(r, s, avg_type=avg_type)
    ranks = {t : U[t] + D[t] for t in G.top_sort}
    
    # Identify critical path.
    critical_path = G.get_critical_path(cp_type="AVG", avg_type=avg_type, r=r, s=s) 
    # Compute the assignment.
    total_cpu_cost = sum(G.graph.nodes[ct]['weight']["c"] for ct in critical_path)
    total_gpu_cost = sum(G.graph.nodes[ct]['weight']["g"] for ct in critical_path)
    if total_cpu_cost < total_gpu_cost:
        alpha = {ct : "c" for ct in critical_path}
    else:
        alpha = {ct : r for ct in critical_path} # Entry task on CP so any choice will do.             
    return priority_scheduling(G, r, s, priorities=ranks, sel_policy="AMT", assignment=alpha, return_schedule=return_schedule)     

def summarize_schedule(schedule, r, s):
    """
    Print the number of tasks assigned to CPUs and GPUs.

    Parameters
    ----------
    schedule : DICT
        A complete schedule.

    Returns
    -------
    None.

    """
    ncpu_tasks = sum(len(load) for worker, load in schedule.items() if worker < r)
    ngpu_tasks = sum(len(load) for worker, load in schedule.items() if worker >= r)    
    ntasks = ncpu_tasks + ngpu_tasks    
    print("Number of tasks on CPU : {} / {} ({}%)".format(ncpu_tasks, ntasks, 100*ncpu_tasks/ntasks))
    print("Number of tasks on GPU : {} / {} ({}%)".format(ngpu_tasks, ntasks, 100*ngpu_tasks/ntasks))
    
    
    
        

        
                
