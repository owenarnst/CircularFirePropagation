import numpy as np
import distance
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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



def forwardBasic(mat, size, p, time):
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

            # check for "off" cells
            if mat[time,i,j] == 0:
                neighbors = getSurroundingCells([i,j], size, 1)
                numSurroundingFires = 0
                
                for cell in neighbors:
                    if mat[time, cell[0], cell[1]] == 1:
                        numSurroundingFires += 1
                
                fireProb = 1 - (1-p)**numSurroundingFires
                rng = np.random.uniform(0,1)
                if rng < fireProb:
                    mat[time+1,i,j]=1

    return mat


def propagateBasic(size, origin, p, maxt):
    """
    origin: origin of the fire
    p: probability of fire propagating to adjacent cell
    size: size of grid [n,m]
    maxt: number of timesteps

    Runs maxt timesteps forward in time

    Returns matrix of size (maxt, size) containing the state of the grid at each of the initial state and the following maxt timesteps
    """

    # initialize current and future states
    fires = np.zeros([maxt+1,size[0],size[1]], dtype=float) # maxt + 1 elements, each nxn
    #for i in range(len(origin)):
    #    fires[0, origin[i,0], origin[i,1]] = 1
    fires[0, origin[0], origin[1]] = 1

    # propagate fire at each timestep
    # if at any point, all cells are on fire, break
    for time in range(maxt):
        if (fires[time,:,:] != 0).all() == True:
            fires[time+1:,:,:] = np.ones_like(fires[time+1:,:,:])
            break
        forwardBasic(fires, size, p, time)
    return fires


def basicEnssemble(size, origin, p, maxt, maxr):
    """
    size: size of matrix
    origin: origin of fire
    p: vector of probabilities to test
    maxt: maximum number of timesteps
    maxr: maximum number of runs to test for each set of conditions


    Returns 3D tensor size len(p) x (maxt + 1) x size[0] x size[1] holding the average states of the fire for each probability and each timestep
    Returns percentage of runs that resulted in a circular fire for each probability tested
    """

    # Initialize empty matrices to hole average fire and successs rates for the propagation method used at each probability tested
    avgFire = np.zeros([maxt + 1, size[0], size[1]])

    for r in range(maxr): # run simulation maxr times
        fire = propagateBasic(size, origin, p, maxt)
        avgFire[:,:,:] += fire

    avgFire[:,:,:] /= maxr # average of fire at each timestep

    return avgFire


if __name__ == "__main__":
    #probabilities = np.arange(0.1,1.1,0.1)
    probabilities = [0.9]
    n = 23
    size=[n,n]
    maxt = 10
    origin = 2*[int(n/2)]
    maxr = 1000
    L = 0.5


    ratios = []


    for p in probabilities:
        # simulate fire using multiple realizations
        avgFire = basicEnssemble(size, origin, p, maxt, maxr)

        
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
            plt.title(f'Average of Basic Method, p={p}, t={int(np.floor(t/4))}', fontsize=18)
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
        #movie.save(f'basicSim{str(int(p*100))}.gif', writer='ffmpeg', fps = 10)

        # and show the movie like this
        plt.colorbar()
        plt.show()

        cardinalDistances = []
        diagonalDistances = []
        ratios = []
        dN = distance.distance(avgFire[-1,:,:], origin, L, 'N')
        dS = distance.distance(avgFire[-1,:,:], origin, L, 'S')
        dW = distance.distance(avgFire[-1,:,:], origin, L, 'W')
        dE = distance.distance(avgFire[-1,:,:], origin, L, 'E')
        d1 = np.mean([dN, dS, dW, dE])

        dNW = distance.distance(avgFire[-1,:,:], origin, L, 'NW')
        dNE = distance.distance(avgFire[-1,:,:], origin, L, 'NE')
        dSW = distance.distance(avgFire[-1,:,:], origin, L, 'SW')
        dSE = distance.distance(avgFire[-1,:,:], origin, L, 'SE')
        d2 = np.mean([dNW, dNE, dSW, dSE])

        R = d1/d2
        error = abs(1-R)
        ratios.append(R)

        


        print(f'Method: Basic')
        print(f'p: {p}')
        print(f'R: {R}')
        print(f'Error: {error}')
        print('\n')
    
    plt.plot(probabilities, ratios)
    plt.title("Basic Method R vs. p")
    plt.xlabel('p')
    plt.ylabel('R')
    plt.grid()
    plt.show()