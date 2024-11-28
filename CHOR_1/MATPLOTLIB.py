import matplotlib.pyplot as plt
import numpy as np

# Generate a numpy array
x = np.arange(0, 6, 0.1)
y = np.sin(x)

# Draw
plt.plot(x, y)
plt.show()
