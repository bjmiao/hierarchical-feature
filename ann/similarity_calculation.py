import numpy as np

def get_rdm(matrix):
    rdm = 1 - np.corrcoef(matrix)
    return rdm

def correlate_rdms(rdm1, rdm2):
    """
    Correlate off-diagonal elements of two RDM's
    Args:
    rdm1 (np.ndarray): S x S representational dissimilarity matrix
    rdm2 (np.ndarray): S x S representational dissimilarity matrix to
      correlate with rdm1

    Returns:
    float: correlation coefficient between the off-diagonal elements
      of rdm1 and rdm2
    """
    # Extract off-diagonal elements of each RDM
    ioffdiag = np.triu_indices(rdm1.shape[0], k=1)  # indices of off-diagonal elements
    rdm1_offdiag = rdm1[ioffdiag]
    rdm2_offdiag = rdm2[ioffdiag]
    corr_coef = np.corrcoef(rdm1_offdiag, rdm2_offdiag)[0,1]
    return corr_coef