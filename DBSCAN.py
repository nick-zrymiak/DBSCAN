import pandas as pd
from math import sqrt
import numpy as np

class Queue:
    def __init__(self, queue=[]):
        self.queue = queue
    
    def enqueue(self, element):
        self.queue.insert(0, element)
    
    def dequeue(self):
        return self.queue.pop(-1)
        
    def is_empty(self):
        return len(self.queue) == 0
        
class Graph:
    def __init__(self, graph={}):
        self.graph = graph
    
    def add_edge(self, from_node, to_nodes):
        if not isinstance(to_nodes, list):
            to_nodes = [to_nodes]
        
        if from_node not in self.graph:
            self.graph[from_node] = to_nodes
        else:
            for n in to_nodes:
                self.graph[from_node].append(n)
        
    def BFS(self, src):
        q = Queue()
        q.enqueue(src)
        visited = [src]
         
        while not q.is_empty():
            v = q.dequeue()
             
            for n in self.graph[v]:
                if n not in visited:
                    q.enqueue(n)
                    visited.append(n)
                    
        return visited

def is_density_reachable(g, eps, min_pts, from_q, to_p):
    # is_core_object condition as it is part of the official density reachable definition
    # is_core_object condition can be commented for this specific implementation of DBSCAN to speed up runtime 
    if not is_core_object(g, min_pts, from_q):
        return False
    if to_p in g.BFS(from_q):
        return True
    return False

def euclidean_distance(u, v):
    dist = 0
    for i in range(len(u)):
        dist += (u[i] - v[i])**2
    
    return sqrt(dist)

def eps_neighbourhood(df, eps, cur_obj_i, dist_matrix) -> list:
    df_prime = df.drop(cur_obj_i)
    
    neighbours = []
    for i, _ in df_prime.iterrows():
        if dist_matrix[cur_obj_i][i] <= eps:
            neighbours.append(i)
        
    return neighbours

def is_core_object(g, min_pts, i):
    return len(g.graph[i]) >= min_pts

def init_dist_matrix(df):
    dist_matrix = np.array([])
    df_prime = df
     
    for i, o in df.iterrows():
        row_dists = np.array([])
        
        for _ in range(i):
            row_dists = np.append(row_dists, 0)
            
        for _, o_prime in df_prime.iterrows():
            dist = euclidean_distance(o, o_prime)
            row_dists = np.append(row_dists, dist)
            
        df_prime = df_prime.iloc[1:]
        dist_matrix = np.append(dist_matrix, row_dists)
     
    dist_matrix = np.reshape(dist_matrix, [len(df), -1])
    dist_matrix = dist_matrix + dist_matrix.T
    dist_df = pd.DataFrame(dist_matrix, index=df.index, columns=df.index)
    
    return dist_df

def init_directed_graph(df, eps):
    g = Graph()
    dist_matrix = init_dist_matrix(df)
    
    for i, _ in df.iterrows():
        neighbours = eps_neighbourhood(df, eps, i, dist_matrix)
        g.add_edge(i, neighbours)
    
    return g

def assign_objects(df, eps, min_pts, g, NOISE, cluster_id, j, df_prime):
    df['cluster_label'][j] = cluster_id
    for i, o_prime in df_prime.iterrows():
        if is_density_reachable(g, eps, min_pts, j, i):
            if o_prime['cluster_label'] == NOISE:
                df['cluster_label'][i] = cluster_id
            else:
                new_ids = [df['cluster_label'][i]]
                if isinstance(df['cluster_label'][i], tuple):
                    new_ids = list(df['cluster_label'][i])
                new_ids.append(cluster_id)
                new_ids = tuple(new_ids)
                df['cluster_label'][i] = new_ids

def DBSCAN(df, eps, min_pts) -> pd.DataFrame:
    g = init_directed_graph(df, eps)
    NOISE = -1
    df['cluster_label'] = NOISE
    df = df.astype({'cluster_label': 'object'})
    cluster_id = 0
    
    for j, o in df.iterrows():
        o = df.iloc[j]
        df_prime = df.drop(j)
        if o['cluster_label'] == NOISE:
            if is_core_object(g, min_pts, j):
                assign_objects(df, eps, min_pts, g, NOISE, cluster_id, j, df_prime)
                cluster_id += 1
                
    return df

def generate_hyperparams(df):
    k = 2*len(df.columns)-1
    min_pts = k+1
    
#   Chose epsilon by plotting distances and identifying distance where slope decreases the fastest
#   The distribution starts to flatten at around distance = .4
    eps = .4
    
    return eps, min_pts

# Details can be found in power_consumption.ipynb
def process(df):
    df = df[df['Date'].apply(lambda x: x.endswith('/1/2007'))]
    df = df.replace('?', float('NaN'))
    df.drop(['Date', 'Time'], axis=1, inplace=True) 
    df = df.apply(pd.to_numeric)
    missing = df[df.isnull().any(axis=1)].index
    
    for m in missing:
        prev_row = df.iloc[m-1]
        next_row = df.iloc[m+1]

        for i, _ in enumerate(prev_row):
            df.iloc[m][i] = (prev_row[i] + next_row[i]) / 2
    
    for col in df:
        df[col] = (df[col] - df[col].mean()) / df[col].std(ddof=0)
    
    return df

if __name__ == '__main__':
    df = pd.read_csv('./household2007.csv')
    df = process(df)
    eps, min_pts = generate_hyperparams(df)
    df = DBSCAN(df, eps, min_pts)
    print(df)
