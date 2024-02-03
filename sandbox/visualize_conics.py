import numpy as np
import matplotlib.pyplot as plt


def conic_to_matrix(A, B, C, D, E, F):
    """
    Constructs and prints a symmetric 3x3 matrix from the 6 parameters
    of a 2D conic section.
    """
    # Construct the matrix
    matrix = np.array([
        [A, B/2, D/2],
        [B/2, C, E/2],
        [D/2, E/2, F]
    ])
    
    # Print the matrix
    print("The symmetric 3x3 matrix representation of the conic is:\n", matrix)


def plot_conic(A, B, C, D, E, F):
    """
    Plots a conic section given the parameters A, B, C, D, E, F of the general quadratic equation:
    Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
    """
    conic_to_matrix(A, B, C, D, E, F)

    # Generate x and y values
    x = np.linspace(-10, 10, 400)
    y = np.linspace(-10, 10, 400)
    x, y = np.meshgrid(x, y)
    
    # Compute the conic equation
    Z = A*x**2 + B*x*y + C*y**2 + D*x + E*y + F
    
    # Plot
    plt.figure(figsize=(6, 6))
    plt.contour(x, y, Z, levels=[0], colors='r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    plt.axis('equal')
    plt.show()

# Example 1: Circle with radius 3
plot_conic(A=1, B=0, C=1, D=0, E=0, F=-9)

# Example 2: Parabola
plot_conic(A=0, B=0, C=1, D=-4, E=0, F=0)

# Example 3: Hyperbola
plot_conic(A=1, B=0, C=-1, D=0, E=0, F=-9)
