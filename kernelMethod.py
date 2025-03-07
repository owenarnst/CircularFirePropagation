import numpy as np
import distance
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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


def forwardKernel(mat, size, p, time, alpha):
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
            # check for cells that equal zero
            if mat[time,i,j] == 0:
                cardinalCells = getCardinalCells([i,j], size, 1)
                diagonalCells = getDiagonalCells([i,j], size, 1)

                # count of cardinal and diagonal fires
                numCardinalFires = 0
                numDiagonalFires = 0

                for cell in cardinalCells:
                    if mat[time, cell[0], cell[1]] == 1:
                        numCardinalFires += 1

                for cell in diagonalCells:
                    if mat[time, cell[0], cell[1]] == 1:
                        numDiagonalFires += 1
            
            fireProb = 1 - ((1-p)**numCardinalFires * (1-p*alpha)**numDiagonalFires)
            rng = np.random.uniform(0,1)

            if rng < fireProb:
                mat[time+1,i,j]=1

    return mat


def propagateKernel(size, origin, p, maxt, alpha):
    """
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
    #for cell in origin:
    #    fires[0,cell[0],cell[1]] = 1
    fires[0, origin[0], origin[1]] = 1

    # propagate fire at each timestep
    # if at any point, all cells are on fire,
    for time in range(maxt):
        forwardKernel(fires, size, p, time, alpha)
    return fires


def kernelEnssemble(size, origin, p, maxt, maxr, alpha):
    """
    size: size of matrix
    origin: origin of fire
    p: vector of probabilities to test
    maxt: maximum number of timesteps
    maxr: maximum number of runs to test for each set of conditions
    alpha: function used to scale probabilities for kernel method

    Similar to testPropagation but takes vector of probabilities and counts number of successful runs

    Returns 3D tensor size (maxt + 1) x size[0] x size[1] holding the average states of the fire for each probability and each timestep
    Returns percentage of runs that resulted in a circular fire for each probability tested
    """

    # Initialize empty matrices to hole average fire and successs rates for the propagation method used at each probability tested
    avgFire = np.zeros([maxt + 1, size[0], size[1]])

    #for i in range(len(p)):
    for r in range(maxr): # run simulation maxr times
        fire = propagateKernel(size, origin, p, maxt, alpha)
        avgFire[:,:,:] += fire

    avgFire[:,:,:] /= maxr # average of fire at each timestep

    return avgFire
    

def optimizeAlpha(size, origin, p, maxt, maxr, alpha0=0.5, epsilon = 0.001):
    alpha = alpha0
    delta=0.25

    alphaValues = [] #keep track of alpha values
    ratios = [] #keep track of distance ratios
    # will plot later


    while delta > epsilon:
        print(f'alpha: {alpha}')

        # simulate fire
        avgFire = kernelEnssemble(size, origin, p, maxt, maxr, alpha)


        # compute R
        dN = distance.distance(avgFire[-1,:,:], origin, 0.5, 'N')
        dS = distance.distance(avgFire[-1,:,:], origin, 0.5, 'S')
        dW = distance.distance(avgFire[-1,:,:], origin, 0.5, 'W')
        dE = distance.distance(avgFire[-1,:,:], origin, 0.5, 'E')
        d1 = np.mean([dN, dS, dW, dE])
        #print(d1)

        dNW = distance.distance(avgFire[-1,:,:], origin, 0.5, 'NW')
        dNE = distance.distance(avgFire[-1,:,:], origin, 0.5, 'NE')
        dSW = distance.distance(avgFire[-1,:,:], origin, 0.5, 'SW')
        dSE = distance.distance(avgFire[-1,:,:], origin, 0.5, 'SE')
        d2 = np.mean([dNW, dNE, dSW, dSE])
        #print(d2)
        
        R = d1/d2

        print(f'R: {R}')
        print('\n')

        alphaValues.append(alpha) #add alpha to list, plot later
        ratios.append(R) # add R to list, plot later

        if R > 1: # traveled more distance in cardinal direction, increase probability of diagonal propagation
            alpha += delta
        elif R < 1: # traveled more distance diagonally, decrease probability of diagonal propagation
            alpha -= delta
        else: # R = 1, perfect circle
            break

        delta *= 1/2 # update delta
        
        # stop when delta is smaller than epsilon (alpha has converged)
        if delta < epsilon:
            break


    # plot evolution of alpha and R
    """fig1 = plt.figure()
    plt.plot(a, label=r'$\alpha(p)$')
    plt.title(r'Evolution of $\alpha(p)$,' + f' p={round(p,1)}')
    plt.xlabel('Iteration')
    plt.ylabel(r'$\alpha$'+f'({round(p,1)})')
    plt.grid()
    #plt.show()
    
    fig2 = plt.figure()
    plt.plot(r, label=r'$R$')
    plt.title(r'Evolution of $R$,' + f' p={round(p,1)}')
    plt.xlabel('Iteration')
    plt.ylabel(r'$R$')
    plt.grid()
    plt.show()"""

    return alpha, R


if __name__ == "__main__":
    probabilities = np.arange(0.1,1.1,0.1)
    n = 25
    size=[n,n]
    maxt = 10
    origin = 2*[int(n/2)]
    maxr = 1000
    L = 0.5

    bestAlpha = [] # keep track of alpha for each p
    ratios = [] # keep track of ratios for given p and alpha

    for p in probabilities:
        print(f'p: {round(p,2)}')
        # find best value for alpha
        alpha, R = optimizeAlpha(size, origin, p, maxt, maxr)

        
        # simulate fire with best alpha
        avgFire = kernelEnssemble(size, origin, p, maxt, maxr, alpha)

        # commenting out animations
        """     
        # animate simulations
        fire = avgFire[:,:,:]
        # Animation code provided by Kevin through Canvas, adjustements were made to match my code
        # create a figure
        fig = plt.figure(figsize=(8,6))

        # create an Axes.Image object with imshow()
        image = plt.imshow(fire[0, :, :])


        # function for FuncAnimation to update the image
        def animate(t):
            image.set_data(fire[int(np.floor(t/4)), :, :])
            plt.title(f'Average of Kernel Method, p={round(p,1)}, ' + r'$\alpha\approx$' + f' {round(alpha,3)}, t={int(np.floor(t/4))}', fontsize=18)
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
        """

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

        bestAlpha.append(alpha) #add alpha to list
        ratios.append(R) # add R to list

        # print inputs and results
        print(f'Method: Kernel')
        print(f'p: {p}')
        print(f'alpha: {alpha}')
        print(f'R: {R}')
        print(f'Error: {error}')
        print('\n')
        


    data = np.zeros([len(probabilities), 3])
    data[:,0] = probabilities
    data[:,1] = ratios
    data[:,2] = bestAlpha

    np.savetxt('kernel_data.csv', data, delimiter=',')


    plt.plot(probabilities, bestAlpha, marker='o')
    plt.title(r'$\mathbf{\alpha}$ vs p', fontsize=18)
    plt.xlabel('p')
    plt.ylabel(r'$\alpha$')
    plt.grid()
    plt.show()

    plt.plot(probabilities, ratios, marker='o')
    plt.title(r'R for Kernel Method with Best $\mathbf{\alpha}$(p)', fontsize=18)
    plt.xlabel('p')
    plt.ylabel('R')
    plt.grid()
    plt.show()