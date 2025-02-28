import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

def getCardinalCells(pos, size, radius):
    """
    searches grid and returns list of neighboring cells 

    pos: position of current cell
    size: size of the grid [n,m]
    radius: number of spaces vertically and horizontally we are willing to look
    """

    left = max(0, pos[0]-radius) # left endpoint of neighbor search
    right = min(size[1], pos[0]+radius+1) # right endpoint of neighbor search
    top = max(0,pos[1]-radius) # top endpoint of neighbor search
    bottom = min(size[0], pos[1]+radius+1) #bottom endpoint of neighbor search
    # add 1 to right and top endpoints to account for zero indexing

    cardinalCells = []
    # check along the horizontal direction
    for i in range(left,right):
        if i != pos[0]:
            cardinalCells.append([i,pos[1]])

    # check along the vertical direction
    for j in range(top, bottom):
        if j != pos[1]:
            cardinalCells.append([pos[0],j])

    return cardinalCells


def getDiagonalCells(pos, size, radius):
    """
    searches grid and returns list of neighboring cells 

    pos: position of current cell
    size: size of the grid [n,m]
    radius: number of spaces diagonally we are willing to look
    """

    left = max(0, pos[0]-radius) # left endpoint of neighbor search
    right = min(size[1], pos[0]+radius+1) # right endpoint of neighbor search
    top = max(0,pos[1]-radius) # top endpoint of neighbor search
    bottom = min(size[0], pos[1]+radius+1) #bottom endpoint of neighbor search
    # add 1 to right and top endpoints to account for zero indexing

    diagonalCells = []

    for i in range(left, right):
        for j in range(top,bottom):
            if i != pos[0] and j != pos[1] and abs(i - pos[0]) == abs(j - pos[1]):
                diagonalCells.append([i,j])

    return diagonalCells


def forwardKernel(mat, time, p, size, alpha):
    """
    mat: matrix size (# of states x size)
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
            if mat[time,i,j] == 1:
                cardinalCells = getCardinalCells([i,j], size, 1)
                diagonalCells = getDiagonalCells([i,j], size, 1)
                for cell in cardinalCells:
                    if mat[time, cell[0], cell[1]] == 0:
                        rng = np.random.uniform(0,1)
                        if rng < p:
                            mat[time+1, cell[0], cell[1]] = 1
                for cell in diagonalCells:
                    if mat[time, cell[0], cell[1]] == 0:
                        rng = np.random.uniform(0,1)
                        if rng < alpha*p:
                            mat[time+1, cell[0], cell[1]] = 1
    return mat


def propagateFire(size, origin, p, maxt, alpha):
    """
    func: propagation function
    origin: origin of the fire
    p: probability of fire propagating to adjacent cell
    size: size of grid [n,m]
    maxt: number of timesteps
    alpha: scaling factor for diagonal cell probabilities

    Runs maxt timesteps forward in time

    Returns matrix of size (maxt, size) containing the state of the grid at each of the initial state and the following maxt timesteps
    """

    # initialize current and future states
    # if n = 1 mod 2, light center
    # if n = 0 mod 2, light center 2x2 submatrix

    fires = np.zeros([maxt+1,size[0],size[1]], dtype=float) # maxt + 1 elements, each nxn
    fires[0, origin[0], origin[1]] = 1

    # propagate fire at each timestep
    # if at any point, all cells are on fire,
    for time in range(maxt):
        forwardKernel(fires, time, p, size, alpha)
    return fires


def fireEnssemble(size, origin, p, maxt, maxr, alpha):
    """
    size: size of matrix
    origin: origin of fire
    p: vector of probabilities to test
    maxt: maximum number of timesteps
    maxr: maximum number of runs to test for each set of conditions
    alpha: function used to scale probabilities for kernel method

    Similar to testPropagation but takes vector of probabilities and counts number of successful runs

    Returns 4D tensor size len(p) x (maxt + 1) x size[0] x size[1] holding the average states of the fire for each probability and each timestep
    Returns percentage of runs that resulted in a circular fire for each probability tested
    """

    # Initialize empty matrices to hole average fire and successs rates for the propagation method used at each probability tested
    avgFire = np.zeros([maxt + 1, size[0], size[1]])

    #for i in range(len(p)):
    for r in range(maxr): # run simulation maxr times
        fire = propagateFire(size, origin, p, maxt, alpha)
        avgFire[:,:,:] += fire

    avgFire[:,:,:] /= maxr # average of fire at each timestep

    return avgFire

def distanceTraveled(mat, origin, L, direction):
    """
    mat: matrix of average fires
    origin: origin of the fire
    L: threshold, if cell's value is larger than L we count it
    direction: specifies whether to check diagonal or cardinal directions
               Must be 'cardinal' or 'diagonal

    Checks along either the cardinal or diagonal direction and returns the average distance traveled in the four directions
    (either cardinal or diagonal)
    """

    if direction not in ["cardinal", "diagonal"]:
        raise ValueError("direction must be either 'cardinal' or 'diagonal'")
    
    size = mat.shape
    
    if direction == 'cardinal':
        # initialize farthest cells travelled in the directions parallel to the vertical and horizontal axes
        upper = origin
        lower = origin
        left = origin
        right = origin

        # check that each cell is within the domain of the grid, then check that each cell's value is larger than L
        for i in range(min(mat.shape)):
            if upper[0] > 0 and mat[upper[0] - 1, upper[1]] >= L:
                upper += np.array([-1, 0])
            if lower[0] < size[0] - 1 and mat[lower[0] + 1, lower[1]] >= L:
                lower += np.array([1, 0])
            if left[1] > 0 and mat[left[0], left[1] - 1] >= L:
                left += np.array([0, -1])
            if right[1] < size[1] - 1 and mat[right[0], right[1] + 1] >= L:
                right += np.array([0, 1])

        upperDistance = la.norm(upper-np.array(origin))
        lowerDistance = la.norm(lower-np.array(origin))
        leftDistance = la.norm(left-np.array(origin))
        rightDistance = la.norm(right-np.array(origin))

        return np.mean([upperDistance, lowerDistance, leftDistance, rightDistance])
    
    else:
        # initialize farthest cells travelled in the diagonal directions at a 45 degree angle
        NW = origin
        NE = origin
        SW = origin
        SE = origin

        # check that each cell is within the domain of the grid, then check that each cell's value is larger than L
        for i in range(min(mat.shape)):
            if (NW[0] > 0) and (NW[1] > 0) and (mat[NW[0] -1, NW[1] - 1] >= L):
                NW += np.array([-1, -1])
            if (NE[0] > 0) and (NE[1] < size[1] - 1) and (mat[NE[0] - 1, NE[1] + 1] >= L):
                NE += np.array([-1, +1])
            if (SW[0] < size[0]) and (SW[1] > 0) and (mat[SW[0] + 1, SW[1] - 1] >= L):
                SW += np.array([1, -1])
            if (SE[0] < size[0] -1) and (SE[1] < size[1] - 1) and (mat[SE[0] + 1, SE[1] + 1] >= L):
                SE += np.array([1, 1])
        NW_distance = la.norm(NW-np.array(origin))
        NE_distance = la.norm(NE-np.array(origin))
        SW_distance = la.norm(SW-np.array(origin))
        SE_distance = la.norm(SE-np.array(origin))

        return np.mean([NW_distance, NE_distance, SW_distance, SE_distance])
    

def optimizeAlpha(size, origin, p, maxt, maxr, alpha0=0.5, delta0=0.5, iters=10):
    alpha = alpha0
    R = 0
    R_prev = R
    R_prev_prev = R_prev
    for n in range(1,iters+1):
        print(f'alpha: {alpha}')
        delta=delta0
        
        avgFire = fireEnssemble(size, origin, p, maxt, maxr, alpha)
        cardinalDistance = distanceTraveled(avgFire[-1,:,:], origin, L, direction='cardinal')
        diagonalDistance = distanceTraveled(avgFire[-1,:,:], origin, L, direction='diagonal')

        R = cardinalDistance/diagonalDistance
        print(f'unsigned relative error: {abs(1-R)}')
        print('\n')

        # if new R is equal to previous two recordings, we found suitable alpha given error is small enough
        if R == R_prev and R == R_prev_prev and abs(R-1)<0.05:
            return alpha, R

        # if error is small, we reduce the stepsize by dividing by the absolute value of order of magnitude
        errorMagnitude = math.log10(abs(R-1))
        if errorMagnitude <= -1:
            delta /= math.ceil(abs(errorMagnitude))

        if R > 1: # traveled more distance in cardinal direction, increase probability of diagonal propagation
            alpha += delta*(1/2)**n
        elif R < 1: # traveled more distance diagonally, decrease probability of diagonal propagation
            alpha -= delta*(1/2)**n
        else:
            break

        # update previous two recordings of R
        R_prev_prev = R_prev
        R_prev = R

    return alpha, R


if __name__ =="__main__":
    p = 0.8
    n = 23
    size=[n,n]
    maxt = 10
    origin = 2*[int(n/2)]
    maxr = 500
    L = 0.5

    # find best value for alpha
    alpha, R = optimizeAlpha(size, origin, p, maxt, maxr)

    # simulate fire with best alpha
    avgFire = fireEnssemble(size, origin, p, maxt, maxr, alpha)


    # animate simulations
    fire = avgFire[:,:,:]
    # Animation code provided by Kevin through Canvas, adjustements were made to match my code
    # create a figure
    fig = plt.figure()

    # create an Axes.Image object with imshow()
    image = plt.imshow(fire[0, :, :])


    # function for FuncAnimation to update the image
    def animate(t):
        image.set_data(fire[int(np.floor(t/4)), :, :])
        plt.title(f'Average of Kernel Method, p={p}, ' + r'$\alpha\approx$' + f' {str(round(alpha,2))}, t={int(np.floor(t/4))}', fontsize=18)
        return [image]

    # instantiate the FuncAnimation class 
    movie = animation.FuncAnimation(
        fig = fig, 
        func = animate, # animate function as defined above
        frames = 4*(maxt + 1), # number of timesteps + initial state
        interval = 200, 
        blit = False, # change blitting to False in order to set title depending on frame number
        # https://stackoverflow.com/questions/44594887/how-to-update-plot-title-with-matplotlib-using-animation
        repeat = False
    )

    # you can save the movie like this
    #movie.save(f'kernelSim{str(int(p*100))}.gif', writer='ffmpeg', fps = 10)

    # and show the movie like this
    plt.colorbar()
    plt.show()