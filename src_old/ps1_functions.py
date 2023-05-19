# ps1_functions.py
# Skeleton file by Chris Harshaw, Yale University, Fall 2017
# Adapted by Jay Stanley, Yale University, Fall 2018
# Adapted by Scott Gigante, Yale University, Fall 2019
# CPSC 553 -- Problem Set 1
#
# This script contains uncompleted functions for implementing diffusion maps.
#
# NOTE: please keep the variable names that I have put here, as it makes grading easier.

# import required libraries
import numpy as np
import scipy.spatial  
import codecs, json

##############################
# Predefined functions
##############################

def load_json_files(file_path):
    '''
    Loads data from a json file

    Inputs:
        file_path   the path of the .json file that you want to read in

    Outputs:
        json_data    this is a numpy array if data is numeric, it's a list if it's a string

    '''

    #  load data from json file
    with codecs.open(file_path, 'r', encoding='utf-8') as handle:
        json_data = json.loads(handle.read())

    # if a string, then returns list of strings
    if not isinstance(json_data[0], str):
        # otherwise, it's assumed to be numeric and returns numpy array
        json_data = np.array(json_data)

    return json_data


##############################
# Skeleton code (fill these in)
##############################


def compute_distances(X):
    '''
    Constructs a distance matrix from data set, assumes Euclidean distance

    Inputs:
        X       a numpy array of size n x p holding the data set (n observations, p features)

    Outputs:
        D       a numpy array of size n x n containing the euclidean distances between points

	'''

    D = scipy.spatial.distance_matrix(X,X) 

    # return distance matrix
    return D


def compute_affinity_matrix(D, kernel_type, sigma=None, k=None):
    '''
    Construct an affinity matrix from a distance matrix via gaussian kernel.

    Inputs:
        D               a numpy array of size n x n containing the distances between points
        kernel_type     a string, either "gaussian" or "adaptive".
                            If kernel_type = "gaussian", then sigma must be a positive number
                            If kernel_type = "adaptive", then k must be a positive integer
        sigma           the non-adaptive gaussian kernel parameter
        k               the adaptive kernel parameter

    Outputs:
        W       a numpy array of size n x n that is the affinity matrix

    '''

    if kernel_type == "gaussian":

        W = np.exp(-(D*D)/(sigma*sigma))



    elif kernel_type == "adaptive":

    	#first define the vector of adaptive sigmas for each point

    	D_sort = np.sort(D, axis = 1)

    	#kth column of sorted array should be distance to kth nearest neighbor for each point

    	sigma_nnk = D_sort[:,k]

    	#Dividing each row by its sigma

    	D_sigma = D/(sigma_nnk[:,None])

    	#Squaring Each Entry

    	D_sigma_sq = D_sigma * D_sigma

    	#Computing Gaussian Part

    	g1 = np.exp(-D_sigma_sq)

    	#Now we transpose so that we get in the same position the distance divided by the sigma of the other point


    	g2 = g1.T

    	#Now we add the two and divide by two to complete the procedure


    	W = 1/2*(g1 + g2)

    # return the affinity matrix
    return W


def diff_map_info(W):
    '''
    Construct the information necessary to easily construct diffusion map for any t

    Inputs:
        W           a numpy array of size n x n containing the affinities between points

    Outputs:

        diff_vec    a numpy array of size n x n-1 containing the n-1 nontrivial eigenvectors of Markov matrix as columns
        diff_eig    a numpy array of size n-1 containing the n-1 nontrivial eigenvalues of Markov matrix

        We assume the convention that the coordinates in the diffusion vectors are in descending order
        according to eigenvalues.
    '''

    row_sums = np.sum(W, axis = 1)

    entries = row_sums ** (-0.5)

    #Create normalizing matrix D^(-0.5)

    D_half = np.diag(entries)

    #globals()['D_half'] = D_half

    #Carry out Matrix Multiplication to get M_s

    P1 = np.matmul(D_half,W)


    M_s = np.matmul(P1, D_half)


    #Get non-trivial eigenvalues and eigenvectors of M_s

    all_vals, all_vecs = np.linalg.eigh(M_s)

    #globals()['M_s'] = M_s


    # return the info for diffusion maps

    #We know all non-diagonal entries for W are > 0 and so when we left and right multiply by D_half we should get a matrix with all values > 0. 
    #Hence highest eigenvalue is unique by Perron-Frobenius. We are told the largest is 1, hence we only need to remove the last eigenvalue and last eigenvector. 

    vals = all_vals[:-1]

    vecs = all_vecs[:,:-1]

    #Now since Numpy Documentation tells me eigh returns values and corresponding vectors in ascending order, we need to flip

    diff_eig = np.flip(vals)

    vecs = np.flip(vecs, axis = 1)

    #Converting to Eigenvalues of Markvov Matrix

    new_vecs = np.matmul(D_half, vecs)

    #Now compute norm of each column

    vec_norms = np.sqrt(np.sum(new_vecs**2, axis = 0))

    #Normalize each column

    diff_vec = new_vecs/vec_norms[None,:]

    return diff_vec, diff_eig


def get_diff_map(diff_vec, diff_eig, t):
    '''
    Construct a diffusion map at t from eigenvalues and eigenvectors of Markov matrix

    Inputs:
        diff_vec    a numpy array of size n x n-1 containing the n-1 nontrivial eigenvectors of Markov matrix as columns
        diff_eig    a numpy array of size n-1 containing the n-1 nontrivial eigenvalues of Markov matrix
        t           diffusion time parameter t

    Outputs:
        diff_map    a numpy array of size n x n-1, the diffusion map defined for t


    '''

    #Raising eigenvalues to power of t

    scales = diff_eig ** (t)

    #Multiplying each normalized vector by its powered eigenvalue

    diff_map = scales[None,:] * diff_vec


    return diff_map

def get_top_five_channels(data, diff_map):

	#Getting Pearson correlation for DM with each channel

    dm_1 = diff_map[:,0]

    dm_2 = diff_map[:,1]

    dm_3 = diff_map[:,2]

    num_channels = np.shape(data)[1]

    #Creating list of correlations

    corrs_1 = np.zeros(num_channels)
    corrs_2 = np.zeros(num_channels)
    corrs_3 = np.zeros(num_channels)

    for i in range(0, num_channels):

	    corrs_1[i] = np.abs(np.corrcoef(dm_1, data[:,i])[0,1])

	    corrs_2[i] = np.abs(np.corrcoef(dm_2, data[:,i])[0,1])

	    corrs_3[i] =np.abs(np.corrcoef(dm_3, data[:,i])[0,1])

    #Getting top 5 channels with highest correlation using argsort 
    #Adding 1 because indices start from 0 but we like to count coordinates or channels from 1
    #Flipping so that they appear in descending order

    dm_1_highest_channels = np.flip(1 + np.argsort(corrs_1)[-5:])

    dm_2_highest_channels = np.flip(1 + np.argsort(corrs_2)[-5:])

    dm_3_highest_channels = np.flip(1 + np.argsort(corrs_3)[-5:])

    return dm_1_highest_channels, dm_2_highest_channels, dm_3_highest_channels

    


