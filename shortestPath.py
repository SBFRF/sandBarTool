import numpy as np
from matplotlib import pyplot as plt

def shortestPath(inputArray, Left2RightPathDirection=False):
    """ Solve for the second pass of accumulated evidence across each "slice" A and
    find the lowest cost path through the resulting accumlated evidence (Y)
    following Step 2 of the method in references
    
    
    Args:
        inputArray: D x V evidence matrix from doEvidence.rows
        Left2RightPathDirection(bool): function assumes top to bottom path direction, mark True to
            make path traverse left to right.

    Returns:
        Y
        ind
        
    References
        Sun, 2002, "Fast Stereo Matching Using Rectangular Sub-regioning
        and 3D Maximum-Surface Techniques," International
        Journal of Computer Vision 47, 99-117.
        
        Method was originally coded by L. Clarke.
        M. Palmsten 11/2009
        translated to Python Spicer Bak 12/19
        
    """
    padWindow = 2      # pads ouput array for ease, split across both sides
    ##############################################################################################
    # first Check for NaN's
    if np.isnan(inputArray).any():
      raise TypeError('input array must not have any NaN.');
    elif np.ma.array(inputArray).mask.any():
        raise TypeError('input array must not have any masked values');
    # Transpose graph if path direction is left to right.
    if Left2RightPathDirection is True:
        tG = inputArray.T
    else:
        tG = inputArray

    # Pad sides of graph with inf to simplify neighbor search near graph boundaries.
    G = np.pad(tG, [(0, 0), (int(padWindow/2), int(padWindow/2)) ],
               mode='constant', constant_values=np.inf)
    rows, cols = np.shape(G)

    #  Forward pass
    #  Y is length of shortest path from graph top to node G(i,j)
    #  Shortest path in first row is just itself.
    Y, k = np.zeros_like(G), np.zeros_like(G)
    Y[0,:] = G[0,:]
    Y[:,0] = np.ones_like(Y[:,0]) * np.inf
    Y[:,-1] = np.ones_like(Y[:,-1]) * np.inf
    step = np.array([0, -1, 1])

    for ii in range(1, rows):
      for jj in range(0, cols-1):
        prev = Y[ii-1, jj+step]                        # Sub array of upper nearest neighbors.
        # Find shortest path neighbor
        # zmin = min(find(prev == min(prev)));                                                          # matlab
        zmin = np.argwhere(prev == min(prev)).squeeze().min()  #np.argwhere(prev == min(prev)).squeeze()
        Y[ii, jj] = G[ii, jj] + prev[zmin]             # Shortest path length to this node.
        k[ii, jj] = step[zmin]                         # Index to shortest path neighbor.

    # Array for indices of shortest path.
    ind = np.zeros(rows)
    
    # Backtracking
    ind[-1] = np.argwhere(Y[-1,:] == min(Y[-1,:])).squeeze().max()

    for ii in range(rows-1, 0, -1):
        ind[ii-1] = ind[ii] + k[ii, int(ind[ii])]

    Y = Y[:, int(padWindow/2):int(-padWindow/2)]
    ind -= 1   # subtract to indices

    if Left2RightPathDirection is True:
        ind = ind.T
        Y = Y.T

    return Y, ind

# generate example image to create path for
# inputArray = np.ones((100, 60))*10
# for i in range(0, 100):
#     for ii in [10,9,8,7,6,5,4,3,2,1,0]:
#         inputArray[slice(i, i+ii), slice(i, i+ii)] = ii
# inputArray *= 1
# plt.pcolormesh(inputArray); plt.colorbar(); plt.show()