import numpy as np
import networkx as nx # handle tree
import matplotlib as mpl
import matplotlib.pyplot as plt

class RapidlyExploringRandomTreesStarClass(object):
    """
        Rapidly-Exploring Random Trees (RRT) Class
    """
    def __init__(self,name,point_min=np.array([-1,-1]),point_max=np.array([+1,+1]),
                 steer_len_max=0.1,norm_ord=2,search_radius=0.3):
        """
            Initialize RRT object
        """
        self.name          = name
        self.point_min     = point_min
        self.point_max     = point_max
        self.point_root    = None
        self.point_goal    = None
        self.dim           = len(self.point_min)
        self.steer_len_max = steer_len_max # maximum steer length
        self.norm_ord      = norm_ord      # norm order (2, inf, ... )
        self.search_radius = search_radius
        self.tree          = None
        
    def reset(self):
        """
            Reset
        """
        if self.tree is not None:
            self.tree.clear()
        
    def init_tree(self,point_root=None,point_goal=None):
        """
            Initialize Tree
        """
        self.reset()
        self.tree = tree = nx.DiGraph(name=self.name) # directed graph
        self.tree.add_node(0)
        if point_root is None:
            self.point_root = np.zeros(self.dim)
        else:
            self.point_root = point_root
        self.tree.update(nodes=[(0,{'point':self.point_root,'cost':0.0})])
        if point_goal is not None:
            self.point_goal = point_goal
        
    def set_goal(self,point_goal):
        """
            Set Goal
        """
        self.point_goal = point_goal
        
    def add_node(self,point=None,cost=None,node_parent=None):
        """
            Add node to tree
        """
        node_new = self.get_n_node()
        self.tree.add_node(node_new)
        if point is not None:
            self.tree.update(
                nodes=[(node_new,{'point':point})]
            )
        if cost is not None:
            self.tree.update(
                nodes=[(node_new,{'cost':cost})]
            )
        if node_parent is not None:
            self.tree.add_edge(node_parent,node_new)
        return node_new
            
    def get_n_node(self):
        """
            Get number of nodes
        """
        return self.tree.number_of_nodes()
    
    def get_nodes(self):
        """ 
            Get tree nodes
        """
        return self.tree.nodes
    
    def get_node_info(self,node):
        """
            Get tree node information
        """
        return self.tree.nodes[node]
    
    def update_node_info(self,node,point=None,cost=None):
        """
            Update node information
        """
        if point is not None:
            self.tree.nodes[node]['point'] = point
        if cost is not None:
            self.tree.nodes[node]['cost'] = cost
    
    def get_edges(self):
        """ 
            Get tree edges
        """
        return self.tree.edges
        
    def get_node_nearest(self,point):
        """
            Get nearest node
        """
        distances = [
            self.get_dist_to_node(node=node,point=point)
            for node in self.tree.nodes
        ]
        node_nearest = np.argmin(distances)
        return node_nearest
    
    def get_node_point(self,node):
        """
            Get node point
        """
        return self.tree.nodes[node]['point']
    
    def get_node_cost(self,node):
        """
            Get node cost
        """
        return self.tree.nodes[node]['cost']
    
    def get_node_point_and_cost(self,node):
        """
            Get node point and cost
        """
        point = self.get_node_point(node)
        cost = self.get_node_cost(node)
        return point,cost
    
    def get_dist(self,point1,point2):
        """
            Get distance
        """
        return np.linalg.norm(point1-point2,ord=self.norm_ord)
    
    def get_dist_to_node(self,node,point):
        """
            Get distance from node to point
        """
        return self.get_dist(self.tree.nodes[node]['point'],point)
    
    def get_dist_to_goal(self):
        """
            Get distance from tree to goal
        """
        node_nearest = self.get_node_nearest(self.point_goal)
        dist_goal = self.get_dist_to_node(node=node_nearest,point=self.point_goal)
        return dist_goal
    
    def get_node_goal(self,eps=1e-6):
        """
            Get goal node
        """
        node_nearest = self.get_node_nearest(self.point_goal)
        dist_goal = self.get_dist_to_node(node=node_nearest,point=self.point_goal)
        if dist_goal < eps:
            node_goal = node_nearest
        else:
            node_goal = None
        return node_goal
    
    def get_node_parent(self,node):
        """
            Get parent node
        """
        node_parent = [node for node in self.tree.predecessors(node)][0]
        return node_parent
    
    def get_path_to_goal(self):
        """
            Get path to goal
        """
        node_goal = self.get_node_goal()
        if node_goal is None: # RRT not finished yet
            path_to_goal = None
            node_list = []
            return path_to_goal,node_list
        path_list = [self.point_goal]
        node_list = [node_goal]
        parent_node = [node for node in self.tree.predecessors(node_goal)][0]
        while parent_node:
            path_list.append(self.tree.nodes[parent_node]['point'])
            node_list.append(parent_node)
            parent_node = [node for node in self.tree.predecessors(parent_node)][0]
        path_list.append(self.point_root)
        node_list.append(0)
        path_list.reverse() # start from root, end with goal
        node_list.reverse()
        path_to_goal = np.array(path_list) # [L x D]

        # Update cost information of 'path_node_list'
        cost_sum = 0
        for idx,node in enumerate(node_list):
            point_curr = self.get_node_info(node)['point']
            if idx > 0:
                node_parent = self.get_node_parent(node)
                point_parent = self.get_node_info(node_parent)['point']
                cost_sum = cost_sum + self.get_dist(point_curr,point_parent)
                # Update cost
                self.update_node_info(node,point=None,cost=cost_sum)

        return path_to_goal,node_list
    
    def sample_point(self):
        """
            Sample point
        """
        point_range = self.point_max-self.point_min
        point_rand = self.point_min+point_range*np.random.rand(self.dim)
        return point_rand
    
    def steer(self,node_nearest,point_sample):
        """
            Steer
        """
        # Find the nearest point in the tree
        point_nearest = self.get_node_point(node=node_nearest)
        
        vector = point_sample - point_nearest
        length = np.linalg.norm(vector)
        if length == 0:
            # If the tree already contains 'point_sample', skip this turn
            point_steer,cost_steer = None,None
        else:
            stepsize = min(self.steer_len_max,length)
            point_steer = point_nearest + vector/np.linalg.norm(vector)*stepsize
            cost_nearest = self.get_node_cost(node=node_nearest)
            cost_steer = cost_nearest + \
                self.get_dist_to_node(node=node_nearest,point=point_steer)
        return point_steer,cost_steer
    
    def get_nodes_near(self,point,search_radius=None):
        """
            Get the list of nodes near 'point' w.r.t given 'search_radius'
        """
        # Get distances of all nodes to 'node' 
        distances = [
            self.get_dist(self.get_node_point(node),point)
            for node in self.get_nodes()
        ]
        # Accumulate the list of near nodes thresholded by 'search_radius'
        if search_radius is None:
            search_radius = self.search_radius
        nodes_near = []
        for node,dist in enumerate(distances):
            if dist <= search_radius:
                nodes_near.append(node)
        return nodes_near

    def replace_node_parent(self,node,node_parent_new):
        """
            Rewire 'node' from 'node_parent_curr' to 'node_parent_new'
        """
        # Remove current parent
        node_parent = self.get_node_parent(node)
        self.tree.remove_edge(node_parent,node)
        # Connect new parent
        self.tree.add_edge(node_parent_new,node)
    
    def plot_tree(self,figsize=(6,6),nodesize=50,arrowsize=10,linewidth=1,
                  nodecolor='w',edgecolor='k',xlim=(-1,+1),ylim=(-1,+1),
                  title_str=None,titlefs=10,SKIP_PLT_SHOW=False):
        """
            Plot tree
        """
        if self.dim == 2:
            pos = {node:self.tree.nodes[node]['point'] for node in self.tree.nodes}
        else:
            pos = nx.spring_layout(self.tree,seed=0)
        plt.figure(figsize=figsize)
        ax = plt.axes()
        nx.draw_networkx_nodes(
            self.tree,pos=pos,node_size=nodesize,node_color=nodecolor,
            linewidths=linewidth,edgecolors=edgecolor,ax=ax)
        nx.draw_networkx_edges(
            self.tree,pos=pos,node_size=nodesize,edge_color=edgecolor,
            width=linewidth,arrowstyle="->",arrowsize=arrowsize,ax=ax)
        ax.tick_params(left=True,bottom=True,labelleft=True,labelbottom=True)
        ax.set(xlim=xlim,ylim=ylim)
        if title_str is None:
            title_str = "Tree of [%s]"%(self.name)
        plt.title(title_str,fontsize=titlefs)
        if not SKIP_PLT_SHOW:
            plt.show()

    def plot_tree_custom(self,
                         figsize=(6,6),xlim=(-1.01,1.01),ylim=(-1.01,1.01),
                         nodems=3,nodemec='k',nodemfc='w',nodemew=1/2,
                         edgergba=[0,0,0,0.2],edgelw=1/2,
                         obsrgba=[0.2,0.2,0.2,0.5],
                         startrgb=[1,0,0],startms=8,startmew=2,startfs=10,
                         goalrgb=[0,0,1],goalms=8,goalmew=2,goalfs=10,
                         pathrgba=[1,0,1,0.5],pathlw=5,pathtextfs=8,
                         obs_list=[],
                         textfs=8,titlestr=None,titlefs=12,
                         PLOT_FULL_TEXT=False
                         ):
        """
            Plot tree without using networkx package
        """
        plt.figure(figsize=figsize)
        ax = plt.axes()
        ax.tick_params(left=True,bottom=True,labelleft=True,labelbottom=True)
        ax.set(xlim=xlim,ylim=ylim)
        # Get node positions
        pos = np.array([self.get_node_point(node) for node in self.get_nodes()])
        edgelist = list(self.get_edges())
        edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in edgelist])
        edge_collection = mpl.collections.LineCollection(
            edge_pos,colors=edgergba[:3],linewidths=edgelw,alpha=edgergba[3])
        ax.add_collection(edge_collection)
        plt.plot(pos[:,0],pos[:,1],'o',ms=nodems,mfc=nodemfc,mec=nodemec,mew=nodemew)
        path_to_goal,path_node_list = self.get_path_to_goal()
        if path_to_goal is not None:
            plt.plot(path_to_goal[:,0],path_to_goal[:,1],'o-',
                    color=pathrgba,lw=pathlw,mec='k',mfc='none')
        for node_idx in range(self.get_n_node()):
            node = self.get_nodes()[node_idx]
            if PLOT_FULL_TEXT:
                plt.text(node['point'][0],node['point'][1],'  [%d] %.2f'%(node_idx,node['cost']),
                        color='k',fontsize=textfs,va='center')
        # Root to goal path
        for node_idx in path_node_list:
            node = self.get_nodes()[node_idx]
            plt.text(node['point'][0],node['point'][1],'  [%d] %.2f'%(node_idx,node['cost']),
                    color='k',fontsize=pathtextfs,va='center')
        for obs in obs_list:
            plt.fill(*obs.exterior.xy,fc=obsrgba,ec='none')
        plt.plot(self.point_root[0],self.point_root[1],'o',
                mfc='none',mec=startrgb,ms=startms,mew=startmew)
        plt.text(self.point_root[0],self.point_root[1],'  Start',
                color=startrgb,fontsize=startfs,va='center')
        plt.plot(self.point_goal[0],self.point_goal[1],'o',
                mfc='none',mec=goalrgb,ms=goalms,mew=goalmew)
        plt.text(self.point_goal[0],self.point_goal[1],'  Goal',
                color=goalrgb,fontsize=goalfs,va='center')
        plt.xticks(fontsize=8); plt.yticks(fontsize=8)
        if titlestr is None:
            titlestr = '%s'%(self.name)
        plt.title(titlestr,fontsize=titlefs)
        plt.show()