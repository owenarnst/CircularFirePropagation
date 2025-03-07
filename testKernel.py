import kernelMethod as kernel
import numpy as np
import distance

def testKernel(size, origin, p, maxt, maxr, alpha):
    """
    propFunc: method used to propagate fire
    p: probability that fire spreads to an adjacent cell
    size: size of grid [n,m]
    maxr: total number of runs to average

    Propagates initial state one step forward in time and averages maxr runs

    Prints the average results to terminal

    Returns average matrix after at each timestep
    """

    # Initialize average matrix to zeros
    Mavg = np.zeros([size[0], size[1]],dtype=float)

    # Sum over maxr sample runs and normalize the matrix
    for r in range(maxr):
        # sampleRun is a (maxt+1) x n x n matrix holding the state of the nxn grid at each timestep
        sampleRun = kernel.propagateKernel(size, origin, p, maxt, alpha)
        Mavg += sampleRun[-1,:,:]
    Mavg /= maxr

    # Print initial condtion
    print(f'Kernel Method\tp={p}, alpha={alpha}\n')
    print(f'Initial Condition: {size[0]}x{size[1]} matrix with center {2-size[0]%2}x{2-size[1]%2} on fire')
    print(sampleRun[0,:,:], "\n")

    # Print average results
    print(f"Average of {maxr} results after {maxt} timesteps:")
    print(Mavg)
    
    return Mavg

if __name__ == "__main__":
    n = 5
    origin = [int(n/2),int(n/2)]
    p = 0.2
    maxt = 2
    maxr = 5000
    alpha = 0.5
    Mavg = testKernel([n,n], origin, p, maxt, maxr, alpha)
    print('\n')