import testPropagation
import propagateFire

p = 0.5
n = 5
origin = [int(n/2), int(n/2)]
maxr = 5000


propMethods = {"Standard": propagateFire.standardPropagation,
               "Forced": propagateFire.forcedPropagation}

# run the propagation test to see if values converge to their expected values
# Timestep propagation method is run for two timesteps to ensure that it works for two different scenarios of t
for methodName, func in propMethods.items():
    print(f'Propagation Method: {methodName}')
    testPropagation.testPropagation(func, [n,n], origin, p, 2, maxr)
    print("\n################\n")