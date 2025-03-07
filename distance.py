import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def distance(mat, origin, L, direction):
    """
    mat: matrix of average fires
    origin: origin of the fire
    L: threshold, if cell's value is larger than L we count it
    direction: specifies whether which of the eight directions to check

    Checks along the specified direction for cells whose average value is greater than L. Returns the distance
    from origin to the point in the specified direction where the mean value is L by linear interpolation.
    """
    if direction not in ['N', 'S', 'E', 'W', 'NW', 'NE', 'SW', 'SE']:
        raise ValueError("direction must be a cardinal or intercardinal direction")
    
    # define which direction to move in
    # note that cells are denoted by [row, column]
    movementDict = {'N': np.array([-1,0]), # move vertically by -1, no horizontal movement
                    'S': np.array([1,0]), # move vertically by 1, no horizontal movement
                    'W': np.array([0,-1]), # no vertical movement, move horizontally by -1
                    'E': np.array([0,1]), # no vertical movement, move horizontally by 1
                    'NW': np.array([-1,-1]), # move vertically by -1, move horizontally by -1
                    'NE': np.array([-1,1]), #move vertically by -1, move horizontally by 1
                    'SW': np.array([1,-1]), # move vertically by 1, move horizontally by -1
                    'SE': np.array([1,1])} # move vertically by 1, move horizontally by 1
    
    # movement will increment position based on specified direction
    movement = movementDict[direction]

    # start from the origin
    currentCell = np.array(origin)
    
    # mean value of mat at position currentCell
    meanCurrent = mat[currentCell[0], currentCell[1]]


    # create figure
    #fig = plt.figure(figsize=(8,8))

    # check that current cell is within domain and mean value at currentCell >= L
    while ((currentCell>0).all() == True) and ((currentCell < (np.array(mat.shape) - 1)).all() == True) and (meanCurrent >= L):
        currentCell += movement # increment position
        meanPrev = meanCurrent # update previous mean
        meanCurrent = mat[currentCell[0], currentCell[1]] # update current mean
        
        distanceCurrent = la.norm(currentCell-np.array(origin)) # distance between origin and  currentCell
        distancePrev = la.norm((currentCell-movement)-np.array(origin)) # distance between origin and cell prior to currentCell
        
        # plot the points and connect with straight line
        #plt.plot([distancePrev, distanceCurrent], [meanPrev, meanCurrent], marker='o', color='k', markersize=10)

    # point slope form
    # y-y0 = m(x-x0)
    # y0 = meanCurrent
    # x0 = distanceCurrent
    # solve for x st y = L
    # L - y0 = m(x-x0)
    # (L - y0)/m = x-x0
    # x = x0 + (L-y0)/m

    m = (meanCurrent-meanPrev)/(distanceCurrent-distancePrev)
    d = distanceCurrent + (L-meanCurrent)/m

    # plot point corresponding to mean value of 0.5
    
    #plt.scatter(d,L, color='r', label=f'({round(d,3)}, {L})', zorder=2, linewidth=4)
    #plt.title(f'Mean Value vs Distance Along {direction}', fontsize=18, fontweight='bold')
    #plt.xlabel('Distance', fontsize=14)
    #plt.ylabel('Mean Value', fontsize=14)
    #plt.xticks(fontsize=12)
    #plt.yticks(fontsize=12)
    #plt.grid()
    #plt.legend(fontsize=12, bbox_to_anchor=(0.85,0.95))
    #plt.show()
    return d