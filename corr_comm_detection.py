import numpy as np
import numpy.matlib
import configcorr as cg #for correlation matrix null model
from scipy.sparse import spdiags
import pandas as pd
import matplotlib.pyplot as plt
from netneurotools import cluster #for consensus clustering
import matlab.engine #for GenLouvain

def cov_to_corr(matrix):
    """function to transform a covariance matrix into a correlation matrix"""
    v = np.sqrt(np.diag(matrix))
    outer_v = np.outer(v, v)
    corr_mat = matrix / outer_v
    corr_mat[matrix == 0] = 0
    return corr_mat

def con_corr_func(p,L,tolerance=1e-5,transform_to_corr_mat=True):
    """
    function to generate configuration model correlation matrix
    from empirical correlation or covariance matrix, using configcorr package
    input: p: empirical correlation or covariance matrix or list of matrices
          L: number of samples from estimated covariance matrix, for null model
          tolerance: to judge whether the null model algorithm has converged. In the paper it is 1e-5
          transform_to_corr_mat: if True, transform the input covariance matrix to the correlation matrix
                                 before running the gradient descent method
                                 if False, treat input as correlation matrix
    output: con_cov: estimated covariance matrix, Sigma in paper (single matrix, or list of matrices)
            con_corr: configuration model correlation matrix, rho^con in paper (single matrix, or list of matrices)
            Sigma_con: configuation model covariance matrix, Sigma_con in paper (single matrix, or list of matrices)
    """
    #convert input empirical correlation or covariance matrix (or matrices) into numpy array
    p = np.array(p)

    #single layer input (p is 2-dimensional array)
    if p.ndim == 2:
        N = len(p) #number of variables
        mean_vec = np.zeros(N)
        con_cov = cg.max_ent_config_dmcc(p, tolerance, transform_to_corr_mat) #estimated covariance matrix (Sigma)
        X = np.random.multivariate_normal(mean_vec,con_cov,size=L).T #sample from Sigma
        Sigma_con = (1/L)*np.matmul(X,X.T)
        con_corr = cov_to_corr(Sigma_con) #configuration model correlation matrix (rho^con)
        return con_cov, con_corr, Sigma_con

    #multilayer input (p is 3-dimensional array)
    if p.ndim == 3:
        T = len(p) #number of layers
        N = len(p[0]) #number of variables
        con_cov = [] #estimated covariance matrices (Sigma)
        con_corr = [] #configuration model correlation matrices (rho^con)
        Sigma_con = [] #configuation model covariance matrices (Sigma^con)
        for i in range(T):
            mean_vec = np.zeros(N)
            con_cov1 = cg.max_ent_config_dmcc(p[i], tolerance, transform_to_corr_mat)
            X = np.random.multivariate_normal(mean_vec,con_cov1,size=L).T #sample from Sigma
            Sigma_con1 = (1/L)*np.matmul(X,X.T)
            con_corr1 = cov_to_corr(Sigma_con1)
            con_cov.append(con_cov1)
            con_corr.append(con_corr1)
            Sigma_con.append(Sigma_con1)
        return con_cov, con_corr, Sigma_con


def multicorrcat(p,con_corr,gamma=1,omega=1,transform_to_corr_mat=True):
    """
    function (inspired by multicat.m) to output a flattened modularity matrix
    input: p: empirical correlation or covariance matrix or list of matrices
           con_corr: configuration model correlation matrix or matrices (output from con_corr function)
           transform_to_corr_mat: if True, transform the input covariance matrix to the correlation matrix
                                  before running the gradient descent method (default=True)
                                  if False, treat input as correlation matrix
           gamma: intralayer resolution parameter (can input np.array of length T, of gamma for each layer)
           omega: interlayer coupling strength
           (can input a np.array, with dimensions number of layer pairs x number of nodes in each layer,
            of omega values.
            NOTE: the layer pair associated with each row of omega must be in this order (1st row to last row):
            (layer i, layer j): (0,1),(0,2),(0,3),...,(0,T-1),(1,2),(1,3),...,(1,T-1),...(T-2,T-1))
    output: B: NxN modularity matrix for single layer network,
              or [NxT]x[NxT] flattened modularity matrix for the
              multilayer network with uniform categorical coupling
    """
    #convert input empirical correlation matrix (or matrices) into numpy array
    p = np.array(p)

    #single layer input (p is 2-dimensional array)
    if p.ndim == 2:
        N = len(p[0]) #number of nodes

        #generate empirical correlation matrix, if covariance matrix is given
        if transform_to_corr_mat == True:
            rho = cov_to_corr(p) #rho is empirical correlation matrix (notation from Naoki's paper)
        else:
            rho = p

        k = np.sum(rho,axis=0)
        twom = np.sum(k)
        #modularity matrix
        B = rho - gamma*con_corr

        return B

    #multilayer input (p is 3-dimensional array)
    if p.ndim == 3:
        T=len(p) #number of layers
        N=len(p[0][0]) #number of nodes in each layer

        #check if gamma is a scalar
        if isinstance(gamma, int) or isinstance(gamma, float):
            gamma = np.repeat(gamma,T)

        #generate empricial correlation matrices, if covariance matrices are given
        rhos = []
        for i in range(T):
            if transform_to_corr_mat == True:
                v = np.sqrt(np.diag(p[i]))
                outer_v = np.outer(v, v)
                rho = p[i] / outer_v #rho is empirical correlation matrix (notation from Naoki's paper)
                rho[p[i] == 0] = 0
            else:
                rho = p[i] #when input matrix is correlation matrix
            rhos.append(rho)

        #initialize modularity matrix B
        B = np.zeros((N*T, N*T))

        C_norm = 0
        for s in range(T):
            mm = np.sum(np.sum(rhos[s], axis=0)) #node strengths in slice s
            C_norm += mm

            B[s*N:(s+1)*N,s*N:(s+1)*N] = rhos[s] - gamma[s]*con_corr[s]

        all2all = N*np.concatenate((np.arange(-T+1,0),np.arange(1,T)))

        #interlayer coupling
        #check if omega is a scalar
        if isinstance(omega, int) or isinstance(omega, float):
            B = B + omega*spdiags(np.ones((2*T-2,N*T)), all2all, N*T, N*T).toarray()
            C_norm += (N*T*(T-1)*omega)
        #if omega is not a scalar, it should be a 2D array of omega values with
        #dimensions (number of layer pairs x number of nodes in each layer)
        #NOTE: the layer pair associated with each row of omega must be in this order (1st row to last row):
        #(layer i, layer j): (0,1),(0,2),(0,3),...,(0,T-1),(1,2),(1,3),...,(1,T-1),...(T-2,T-1)
        else:
            #find layer pair indices needed to access omegas for the diagonals
            inds = []
            ind = [T-2]
            inds.append(ind)
            diff = T-1
            for i in range(T-2):
                ind = [j-1 for j in ind]
                ind.append(ind[-1]+diff)
                inds.append(ind)
                diff -= 1

            #fill in lower diagonals with proper omegas
            lower_diags = []
            mult = T-1
            for i in range(T-1):
                diag = np.concatenate((omega[inds[i]].flatten(),np.zeros(mult*N)))
                mult -= 1
                lower_diags.append(diag)

            #fill in upper diagonals with proper omegas
            upper_diags = []
            for i in range(T-1):
                diag = np.concatenate((np.zeros((i+1)*N),omega[inds[-(i+1)]].flatten()))
                upper_diags.append(diag)

            diags = lower_diags + upper_diags

            #add omegas on the proper diagonals of the modularity matrix B
            B = B + spdiags(diags, all2all, N*T, N*T).toarray()
            C_norm += 2*np.sum(omega)

        #return flattened modularity matrix
        return B

def it_genlouvain_corr_consensus(p,B,genlouvain_file_location,runs=200):
    """
    function to run iterated genlouvain MMM on correlation (or covariance) matrices for each layer
    input: p: empirical correlation or covariance matrix or list of matrices
           B: [NxT]x[NxT] flattened modularity tensor for the
              multilayer network with uniform categorical coupling, output from multicorrcat function
           genlouvain_file_location: string with path to where the file iterated_genlouvain.m is located on your computer
           runs: number of times to run genlouvain
    output: final_part_array: 2D numpy array describing the partition found by iterated genlouvain,
                              rows represent genes, columns represent layers, entries represent community assignment
    """
    #convert input empirical correlation matrix (or matrices) into numpy array
    p = np.array(p)

    #single layer input (p is 2-dimensional array)
    if p.ndim == 2:
        #number of nodes
        N = len(p[0])
        #single layer
        T = 1

    #multilayer input (p is 3-dimensional array)
    if p.ndim == 3:
        #number of nodes in each layer
        N = len(p[0])
        #number of layers
        T = len(p)

    #convert numpy array to matlab matrix
    Bmat = matlab.double(B)

    eng = matlab.engine.start_matlab()
    #change to the directory where the matlab file iterated_genlouvain.m is located to run MMM
    eng.cd(str(genlouvain_file_location))

    #run genlouvain runs times to get runs partitions
    partitions = []
    for i in range(runs):
        [partition,Q,nit] = eng.iterated_genlouvain(Bmat,nargout=3)
        part_array = np.asarray(partition)
        part_array_flat = part_array.flatten().astype(int)
        partitions.append(part_array_flat)

    #find consensus clustering
    consensus = cluster.find_consensus(np.column_stack(partitions), seed=1234) #1D numpy array of consensus partition

    #reshape partition so row represents the gene and column represents the layer
    part_reshape = np.reshape(consensus,(N,T),order='F')

    #re-number communities in descending order of the number of nodes
    new_part_array = np.copy(part_reshape)
    unique, counts = np.unique(part_reshape, return_counts=True)
    num_nodes_comm = dict(zip(unique, counts))
    comms_large_to_small = sorted(num_nodes_comm, key=num_nodes_comm.get, reverse=True)
    new_comm_nums = np.array(list(range(1,len(comms_large_to_small)+1)))
    new_part = np.empty(part_reshape.max() + 1, dtype=new_comm_nums.dtype)
    new_part[comms_large_to_small] = new_comm_nums
    final_part_array = new_part[new_part_array]

    return final_part_array

def corr_partition_info(p,final_part_array):
    """
    function to obtain information about partition needed for significance calculations
    input: p: empirical covariance matrix or matrices
           final_part_array: 2D numpy array, output from it_genlouvain_consensus, row=node, column=layer
    output: nodes_by_comm_by_layer: 3D list of nodes in each community by layer
            nodes_notin_by_comm_by_layer: 3D list of nodes NOT IN each community by layer
            tot_intra_by_comm: 1D list of total intralayer strength within in each community
    """
    #convert input empirical covariance matrix (or matrices) into numpy array
    p = np.array(p)

    #single layer input (p is 2-dimensional array)
    if p.ndim == 2:
        layer_ids = [0]

    #multilayer input (p is 3-dimensional array)
    if p.ndim == 3:
        #list of layer ids
        layer_ids = list(range(len(p)))

    #number of communities
    num_comms = np.max(final_part_array)
    #node indicies (same for each layer because same number of nodes)
    node_inds = np.arange(len(final_part_array))

    nodes_by_comm_by_layer = []
    nodes_notin_by_comm_by_layer = []
    for i in range(1,num_comms+1):
        nodes_by_comm = []
        nodes_notin_by_comm = []
        #2D array of indicies of partition_arr that are in community i
        node_inds_arr = np.argwhere(final_part_array==i)
        for j in layer_ids:
            #1D array of node indicies of nodes in community i in layer j
            nodes = node_inds_arr[np.where(node_inds_arr[:,1]==j)][:,0].tolist()
            #1D array of node indicies of nodes NOT IN community i in layer j
            nodes_notin = list(set(node_inds)-set(nodes))
            nodes_by_comm.append(nodes)
            nodes_notin_by_comm.append(nodes_notin)
        nodes_by_comm_by_layer.append(nodes_by_comm)
        nodes_notin_by_comm_by_layer.append(nodes_notin_by_comm)

    #total intralayer strength within in each community
    tot_intra_by_comm = []
    for c in range(num_comms):
        strength = 0
        for m in layer_ids:
            for i in range(len(nodes_by_comm_by_layer[c][m])):
                for j in range(i):
                    node1 = nodes_by_comm_by_layer[c][m][i]
                    node2 = nodes_by_comm_by_layer[c][m][j]
                    if p.ndim == 2:
                        strength += p[node1][node2]
                    if p.ndim == 3:
                        strength += p[m][node1][node2]
        tot_intra_by_comm.append(strength)

    return nodes_by_comm_by_layer, nodes_notin_by_comm_by_layer, tot_intra_by_comm

def corr_intra_z(con_cov,nodes_by_comm_by_layer,tot_intra_by_comm,L):
    """
    function to calculate significance (Z score for total intralayer weight) of each community
    input: con_cov: estimated covariance matrix, Sigma in paper (single matrix, or list of matrices)
           nodes_by_comm_by_layer: 3D list of nodes in each community by layer
           tot_intra_by_comm: 1D list of total intralayer strength within in each community
           L: number of samples from estimated covariance matrix, for null model
    output: zscores: list of Z scores for total intralayer strength of each community detected
    """

    #convert input empirical covariance matrix (or matrices) into numpy array
    con_cov = np.array(con_cov)

    #single layer input (con_cov is 2-dimensional array)
    if con_cov.ndim == 2:
        layer_ids = [0]

    #multilayer input (con_cov is 3-dimensional array)
    if con_cov.ndim == 3:
        #list of layer ids
        layer_ids = list(range(len(con_cov)))

    #number of communities
    num_comms = len(nodes_by_comm_by_layer)

    E_by_comm = []
    for c in range(num_comms):
        strength = 0
        for m in layer_ids:
            for i in range(len(nodes_by_comm_by_layer[c][m])):
                for j in range(i):
                    nodei = nodes_by_comm_by_layer[c][m][i]
                    nodej = nodes_by_comm_by_layer[c][m][j]
                    if con_cov.ndim == 2:
                        strength += con_cov[nodei][nodej]
                    if con_cov.ndim == 3:
                        strength += con_cov[m][nodei][nodej]
        E_by_comm.append(strength)

    Var_by_comm = []
    for c in range(num_comms):
        strength = 0
        for a in layer_ids:
            for i in range(len(nodes_by_comm_by_layer[c][a])):
                for j in range(i):
                    for k in range(len(nodes_by_comm_by_layer[c][a])):
                        for r in range(k):
                            nodei = nodes_by_comm_by_layer[c][a][i]
                            nodej = nodes_by_comm_by_layer[c][a][j]
                            nodek = nodes_by_comm_by_layer[c][a][k]
                            noder = nodes_by_comm_by_layer[c][a][r]
                            if con_cov.ndim == 2:
                                strength += ((1/L)*((con_cov[nodei][nodek]*con_cov[nodej][noder])+ \
                                                (con_cov[nodei][noder]*con_cov[nodej][nodek])))
                            if con_cov.ndim == 3:
                                strength += ((1/L)*((con_cov[a][nodei][nodek]*con_cov[a][nodej][noder])+ \
                                                (con_cov[a][nodei][noder]*con_cov[a][nodej][nodek])))
        Var_by_comm.append(strength)

    zscores = []
    for i in range(num_comms):
        numerator = (tot_intra_by_comm[i]-E_by_comm[i])
        var = Var_by_comm[i]
        if var != 0:
            zscore = numerator/np.sqrt(var)
        else:
            zscore = 'N/A' #set Z score equal to N/A if the variance is equal to zero
        zscores.append(zscore)

    return zscores

def main(p,L,genlouvain_file_location,tolerance=1e-5,runs=200,gamma=1,omega=1):
    """
    main function that performs community detection with covariance matrix or matrices input
    and outputs the partition array and the Z score for each community
    input: p: single covariance matrix, or list/3D array of covariance matrices
           L: number of samples from estimated covariance matrix, for null model
           genlouvain_file_location: string with path to where the file iterated_genlouvain.m is located on your computer
           tolerance: tolerance: to judge whether the null model algorithm has converged. In the paper it is 1e-5
           runs: number of runs of iterated genlouvain
           gamma: resolution parameter
                  can be a scalar or a np.array of length=number of layers (a gamma for each layer)
           omega: interlayer coupling strength parameter
                  can be a scalar or a np.array with dimensions number of layer pairs x number of nodes in each layer
                  NOTE: the layer pair associated with each row of omega must be in this order (1st row to last row):
                  (layer i, layer j): (0,1),(0,2),(0,3),...,(0,T-1),(1,2),(1,3),...,(1,T-1),...(T-2,T-1)
    output: final_part_array: partition of the network.
                            2D numpy array describing the partition found by iterated genlouvain,
                            dimensions are number of nodes in each layer x number of layers
                            rows represent variables, columns represent layers,
                            entries represent community assignment of that node
            zscores: list of z-scores for total intralayer strength of each community detected
    """

    con_cov, con_corr, Sigma_con = con_corr_func(p,L,tolerance,transform_to_corr_mat=True)
    B = multicorrcat(p,con_corr,gamma=gamma,omega=omega,transform_to_corr_mat=True)
    final_part_array = it_genlouvain_corr_consensus(p,B,genlouvain_file_location,runs=runs)
    nodes_by_comm_by_layer, nodes_notin_by_comm_by_layer, tot_intra_by_comm = corr_partition_info(p,final_part_array)
    zscores = corr_intra_z(con_cov,nodes_by_comm_by_layer,tot_intra_by_comm,L)

    return final_part_array, zscores
