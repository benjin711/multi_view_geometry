import numpy as np
import matplotlib.pyplot as plt

# Function to plot a vector
def plot_vector(v, color='blue', start=[0, 0], label=None, alpha=0.5):
    plt.quiver(*start, *v, angles='xy', scale_units='xy', scale=1, color=color, label=label, alpha=alpha)

# Initial vectors
vector1 = np.array([np.cos(np.radians(45)), 1])  # Example vector
vector2 = np.array([np.cos(np.radians(120)), 1])  # Another example vector

# Transformation: Mirror vector2 on the x=y axis and then multiply y by -1
transformed_vector1 = np.array([vector2[1], vector2[0]])
transformed_vector2 = np.array([vector2[1], -vector2[0]])

# Plotting
plt.figure(figsize=(8, 8))
plot_vector(vector1, 'blue', label='Vector 1')
plot_vector(vector2, 'green', label='Vector 2')
plot_vector(transformed_vector1, 'green', label='Transformed Vector 1', alpha=0.75)
plot_vector(transformed_vector2, 'green', label='Transformed Vector 2', alpha=1.0)

# Setting plot limits and labels
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Vector Transformations')
plt.legend()

# Show plot
plt.show()
