import matplotlib.pyplot as plt
import numpy as np

training_number = np.log([50995*n for n in [1/16, 1/8, 1/4, 1/2, 1]])
training_error = np.log([0.0064, 0.0082, 0.0083, 0.0085, 0.0143])
test_error = np.log([0.0834, 0.0614, 0.0528, 0.0395, 0.0381])

plt.plot(training_number, training_error, label='training error')
plt.plot(training_number, test_error, label='test error')
plt.xlabel('log(number of examples)')
plt.ylabel('log(error)')
plt.legend()
plt.title('training and test error with different number of training examples')
plt.show()

