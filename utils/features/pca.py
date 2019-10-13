import time
import numpy as np

def pca_naive(X, K):
    """
    PCA -- naive version

    Inputs:
    - X: (float) A numpy array of shape (N, D) where N is the number of samples,
         D is the number of features
    - K: (int) indicates the number of features you are going to keep after
         dimensionality reduction

    Returns a tuple of:
    - P: (float) A numpy array of shape (K, D), representing the top K
         principal components
    - T: (float) A numpy vector of length K, showing the score of each
         component vector
    """

    ###############################################
    # TODO: Implement PCA by extracting        #
    # eigenvector.You may need to sort the      #
    # eigenvalues to get the top K of them.     #
    ###############################################
    ###############################################
    #          START OF YOUR CODE         #
    ###############################################

    N=X.shape[0]
    D=X.shape[1]

    cov=(X.T.dot(X))/(N-1)

    eigenvalue,eigenvector=np.linalg.eig(cov)
    eigen_index=eigenvalue.argsort()[::-1]
    eigenvalue = eigenvalue[eigen_index]
    eigenvector = eigenvector[:,eigen_index]

    eigenvector_K = eigenvector[:,0:K]
    P = eigenvector_K.T
    
    #keep eigenvalue
    T=eigenvalue[0:K]


    ###############################################
    #           END OF YOUR CODE         #
    ###############################################
    
    return (P, T)
