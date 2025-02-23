import propagateFire
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

################################################################################################################################
# chosen test parameters
#probabilities = [0.2, 0.5, 0.8] # probabilities to test
probabilities = [i/10 for i in range(1,11)]
n = 40 # size of nxn grid
maxt = 10 # number of timesteps
origin = [int(n/2), int(n/2)] # origin of fire
maxr = 100 # number of runs for each probability
# chosen test parameters
################################################################################################################################




################################################################################################################################
# run simulations
avg = propagateFire.avgProp(propagateFire.kernelPropagation, [n,n], origin, probabilities, maxt, maxr)
# run simulations
################################################################################################################################




################################################################################################################################
# animate simulations
for j in range(len(probabilities)):
    fire = avg[j,:,:,:]
    # Animation code provided by Kevin through Canvas, adjustements were made to match my code
    # create a figure
    fig = plt.figure()
    #plt.xlabel("X")
    #plt.ylabel("Y")
    plt.xticks([])
    plt.yticks([])

    # create an Axes.Image object with imshow()
    image = plt.imshow(fire[0, :, :])


    # function for FuncAnimation to update the image
    def animate(t):
        image.set_data(fire[int(np.floor(t/4)), :, :])
        plt.title(f'Kernel Method, p={probabilities[j]}, ' + r'$\alpha(p)\propto 2\sigma(p)-1$' + f', time={int(np.floor(t/4))}', fontsize=18)
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
    # saving as mp4 not working so I saved the file as a gif
    movie.save(f'kernelSigmoidSim{str(int(probabilities[j]*100))}.gif', writer='ffmpeg', fps = 10)

    # and show the movie like this
    plt.show()
# animate simulations
################################################################################################################################