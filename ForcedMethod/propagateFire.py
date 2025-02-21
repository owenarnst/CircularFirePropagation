import numpy as np

################################################################
# FUNCTION DEFINITIONS
def norm(origin, point):
    """
    origin: starting point of fire
    point: point on grid where fire reached

    Returns the Euclidean distance between origin and point
    """

    x0, y0 = origin
    x, y = point

    return np.sqrt((x-x0)**2 + (y-y0)**2)   


def getSurroundingCells(pos, size, radius):
    """
    searches grid and returns list of neighboring cells 

    pos: position of current cell
    size: size of the grid [n,m]
    radius: number of spaces vertically, horizontally, or diagonally we are willing to look
    """

    left = max(0, pos[0]-radius) # left endpoint of neighbor search
    right = min(size[1], pos[0]+radius+1) # right endpoint of neighbor search
    top = max(0,pos[1]-radius) # top endpoint of neighbor search
    bottom = min(size[0], pos[1]+radius+1) #bottom endpoint of neighbor search
    # add 1 to right and top endpoints to account for zero indexing
    
    width = right - left # total width of possible neighbors
    height = bottom - top # total height of possible neighbors
    numSurrounding = width*height - 1 # subtract 1 to account for cell [i,j]

    # initialize array to hold neighboring cells
    surroundingCells = np.zeros([numSurrounding, 2], dtype=int)

    # loop to add neighbors to array
    for i in range(left, right):
        for j in range(top, bottom):
            if i != pos[0] or j != pos[1]:
                surroundingCells[numSurrounding-1] = [i,j]
                numSurrounding -= 1

    return surroundingCells


def standardPropagation(mat, distances, time, p, size):
    """
    mat: matrix size (# of states x size)
    distances: unused by algorithm. Dummy variable needed so that avgProp can work on both propagation methods
    time: current timestate (propagating to time + 1)
    p: probability of fire spreading to adjacent cell
    size: size of grid

    Propagates one timestep forward given current state of the grid

    Returns input matrix, mat, with timestep (time +1) updated
    """

    mat[time+1,:,:] = mat[time,:,:] # copy current timestate to next timestate for propagation

    # check all cells in matrix
    # if cell is 0, then find neighboring cells and count how many are on fire 
    # calculate probability of fire for current cell then roll 
    # ignite based on roll
    for i in range(size[0]):
        for j in range(size[1]):
            if mat[time,i,j] == 0:
                neighbors = getSurroundingCells([i,j], size, 1)
                numSurroundingFires = 0
                for cell in neighbors:
                    if mat[time,cell[0], cell[1]] != 0:
                        numSurroundingFires += 1
                
                fireProb = 1- (1-p)**numSurroundingFires # 1- prob(no fire)

                if np.random.uniform(0,1) < fireProb:
                    mat[time+1,i,j] = 1
    return mat



def forcedPropagation(mat, distances, time, p, size):
    """
    mat: matrix size (# of states x size)
    time: current timestate (propagating to time + 1)
    p: probability of fire spreading to adjacent cell
    size: size of grid

    Propagates one timestep forward given current state of the grid ==> from time to time + 1

    Returns input matrix, mat, with timestep (time +1) updated
    """

    mat[time+1,:,:] = mat[time,:,:] # copy current timestate to next timestate for propagation

    # check all cells in matrix
    # if cell is 0, then find neighboring cells and count how many are on fire 
    # calculate probability of fire for current cell then roll 
    # ignite based on roll
    for i in range(size[0]):
        for j in range(size[1]):
            if distances[i,j] <= (time + 1) and mat[time,i,j] == 0:
                neighbors = getSurroundingCells([i,j], size, 1)
                numSurroundingFires = 0
                for cell in neighbors:
                    if mat[time,cell[0], cell[1]] != 0:
                        numSurroundingFires += 1
                
                fireProb = 1- (1-p)**numSurroundingFires # 1- prob(no fire)

                if np.random.uniform(0,1) < fireProb:
                    mat[time+1,i,j] = 1
    return mat


def runSimulation(func, size, origin, p, maxt):
    """
    func: propagation function
    origin: origin of the fire
    p: probability of fire propagating to adjacent cell
    size: size of grid [n,m]
    maxt: number of timesteps

    Runs maxt timesteps forward in time

    Returns matrix of size (maxt, size) containing the state of the grid at each of the initial state and the following maxt timesteps
    """

    # initialize current and future states
    # if n = 1 mod 2, light center
    # if n = 0 mod 2, light center 2x2 submatrix

    fires = np.zeros([maxt+1,size[0],size[1]], dtype=float) # maxt + 1 elements, each nxn
    fires[0, origin[0], origin[1]] = 1

    distance = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            distance[i,j] = norm(origin, [i,j])

    # propagate fire at each timestep
    # if at any point, all cells are on fire,
    for time in range(maxt):
        if (fires[time,:,:] != 0).all() == True:
            fires[time+1:,:,:] = np.ones_like(fires[time+1:,:,:])
            break
        func(fires, distance, time, p, size)
    return fires


def avgProp(propFunc, size, origin, p, maxt, maxr):
    """
    propFunc: Propagation method
    size: size of matrix
    origin: origin of fire
    p: vector of probabilities to test
    maxt: maximum number of timesteps
    maxr: maximum number of runs to test for each set of conditions

    Similar to testPropagation but takes vector of probabilities and counts number of successful runs

    Returns 4D tensor size len(p) x (maxt + 1) x size[0] x size[1] holding the average states of the fire for each probability and each timestep
    Returns percentage of runs that resulted in a circular fire for each probability tested
    """

    # Initialize empty matrices to hole average fire and successs rates for the propagation method used at each probability tested
    avgFire = np.zeros([len(p), maxt + 1, size[0], size[1]])

    for i in range(len(p)):
        for r in range(maxr): # run simulation maxr times
            fire = runSimulation(propFunc, size, origin, p[i], maxt)
            avgFire[i,:,:,:] += fire

        avgFire[i,:,:,:] /= maxr # average of fire at each timestep

    return avgFire
################################################################