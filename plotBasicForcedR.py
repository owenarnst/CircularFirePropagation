import basicMethod as basic
import forcedMethod as forced
import distance
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    probabilities = np.arange(0.1, 1.1, 0.1)
    n = 21
    size = [n,n]
    maxt = 10
    origin = 2*[int(n/2)]
    maxr = 1000
    L = 0.5
    
    basicR = []
    forcedR = []

    for p in probabilities:
        basicFire = basic.basicEnssemble(size, origin, p, maxt, maxr)
        forcedFire = forced.forcedEnssemble(size, origin, p, maxt, maxr)


        # compute R for basic method
        dN_basic = distance.distance(basicFire[-1,:,:], origin, L, 'N')
        dS_basic = distance.distance(basicFire[-1,:,:], origin, L, 'S')
        dW_basic = distance.distance(basicFire[-1,:,:], origin, L, 'W')
        dE_basic = distance.distance(basicFire[-1,:,:], origin, L, 'E')
        d1_basic = np.mean([dN_basic, dS_basic, dW_basic, dE_basic])

        dNW_basic = distance.distance(basicFire[-1,:,:], origin, L, "NW")
        dNE_basic = distance.distance(basicFire[-1,:,:], origin, L, "NE")
        dSW_basic = distance.distance(basicFire[-1,:,:], origin, L, "SW")
        dSE_basic = distance.distance(basicFire[-1,:,:], origin, L, "SE")
        d2_basic = np.mean([dNW_basic, dNE_basic, dSW_basic, dSE_basic])

        basicR.append(d1_basic/d2_basic) # add R to list for plotting


        # compute R for forced method
        dN_forced = distance.distance(forcedFire[-1,:,:], origin, L, 'N')
        dS_forced = distance.distance(forcedFire[-1,:,:], origin, L, 'S')
        dW_forced = distance.distance(forcedFire[-1,:,:], origin, L, 'W')
        dE_forced = distance.distance(forcedFire[-1,:,:], origin, L, 'E')
        d1_forced = np.mean([dN_forced, dS_forced, dW_forced, dE_forced])

        dNW_forced = distance.distance(forcedFire[-1,:,:], origin, L, 'NW')
        dNE_forced = distance.distance(forcedFire[-1,:,:], origin, L, 'NE')
        dSW_forced = distance.distance(forcedFire[-1,:,:], origin, L, 'SW')
        dSE_forced = distance.distance(forcedFire[-1,:,:], origin, L, 'SE')
        d2_forced = np.mean([dNW_forced, dNE_forced, dSW_forced, dSE_forced])

        forcedR.append(d1_forced/d2_forced) #add to list, will plot later

    plt.plot(probabilities, basicR, label="basic")
    plt.plot(probabilities, forcedR, label="forced")

    plt.title("Basic vs. Forced Method R Values")
    plt.xlabel('p')
    plt.ylabel('R')
    plt.legend()
    plt.grid()
    plt.show()