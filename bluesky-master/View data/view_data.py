import numpy as np

success = np.load('../results/sim1Suc.npy')
collision = np.load('../results/sim1Col.npy')

final = np.array(list(zip(success, collision)))

np.savetxt('sim1.csv', final, delimiter=',', fmt='%.f')