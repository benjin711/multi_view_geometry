

from typing import Tuple
from matplotlib import patches, pyplot as plt

import numpy as np

from enum import Enum

class ConicCategory(Enum):
    """
    An enumeration of the categories of conic sections.
    """
    ELLIPSE = "Ellipse"
    CIRCLE = "Circle"
    PARABOLA = "Parabola"
    HYPERBOLA = "Hyperbola"
    INTERSECTING_LINES = "Intersecting Lines"
    LINE = "Line"
    POINT = "Point"
    PLANE = "Plane"

    def __str__(self) -> str:
        return self.value

def categorize_conic(A: float, B: float, C: float, D: float, E: float, F: float) -> str:
    """
    Return the category of the conic section given the general Cartesian form of the equation:
    Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
    """
    mat = np.array([
        [A, B / 2, D / 2],
        [B / 2, C, E / 2],
        [D / 2, E / 2, F]
    ])
    det = np.linalg.det(mat)

    if det == 0:
        rank = np.linalg.matrix_rank(mat)
        if rank == 0:
            return ConicCategory.PLANE
        elif rank == 1:
            return ConicCategory.POINT
        elif rank == 2:
            sub_det = np.linalg.det(mat[:2, :2])
            if sub_det != 0:
                return ConicCategory.INTERSECTING_LINES
            else:
                return ConicCategory.LINE
    else:
        discriminant = B**2 - 4 * A * C
        if discriminant < 0:
            return ConicCategory.ELLIPSE
        elif discriminant == 0:
            return ConicCategory.PARABOLA
        else:
            return ConicCategory.HYPERBOLA


class Ellipse:

    def __init__(
        self,
        a: float,
        b: float,
        center: Tuple[float, float] = (0, 0),
        theta: float = 0
    ) -> None:
        assert a > 0, "a must be greater than 0"
        assert b > 0, "b must be greater than 0"
        assert a >= b, "a must be greater than or equal to b"
        assert -np.pi <= theta < np.pi, "rotation must be in the range (-pi, pi)"

        self._a = a
        self._b = b
        self._center = center
        self._theta = theta

    @property
    def a(self) -> float:
        """
        Return the length of the major axis.
        """
        return self._a
    
    @property
    def b(self) -> float:
        """
        Return the length of the minor axis.
        """
        return self._b

    @property
    def center(self) -> Tuple[float, float]:
        """
        Return the center of the ellipse.
        """
        return self._center
    
    @property
    def theta(self) -> float:
        """
        Return the rotation of the ellipse.
        """
        return self._theta

    @property
    def eccentricity(self) -> float:
        """
        Return the eccentricity of the ellipse.
        """
        return np.sqrt(self._a**2 - self._b**2) / self._a

    def to_general_cartesian_form(self) -> Tuple[float, float, float, float, float, float]:
        """
        Return the general Cartesian form of the equation of the ellipse:
        Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
        """
        # A = self._a**(-2) * np.cos(self._theta)**2 + self._b**(-2) * np.sin(self._theta)**2
        # B = 2 * (self._b**(-2) - self._a**(-2)) * np.sin(self._theta) * np.cos(self._theta)
        # C = self._a**(-2) * np.sin(self._theta)**2 + self._b**(-2) * np.cos(self._theta)**2
        # D = -2 * A * self._center[0] - B * self._center[1]
        # E = -2 * C * self._center[1] - B * self._center[0]
        # F = A * self._center[0]**2 + B * self._center[0] * self._center[1] + C * self._center[1]**2 - 1

        A = self._a**2*np.sin(self._theta)**2 + self._b**2*np.cos(self._theta)**2
        B = 2*(self._b**2 - self._a**2)*np.sin(self._theta)*np.cos(self._theta)
        C = self._a**2*np.cos(self._theta)**2 + self._b**2*np.sin(self._theta)**2
        D = -2*A*self._center[0] - B*self._center[1]
        E = -B*self._center[0] - 2*C*self._center[1]
        F = A*self._center[0]**2 + B*self._center[0]*self._center[1] + C*self._center[1]**2 - self._a**2*self._b**2
        return A, B, C, D, E, F

    
    def to_foci_tuplet(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Return the foci of the ellipse as a tuplet.
        """
        c = np.sqrt(self._a**2 - self._b**2)
        foci = (
            (self._center[0] + c * np.cos(self._theta), self._center[1] + c * np.sin(self._theta)),
            (self._center[0] - c * np.cos(self._theta), self._center[1] - c * np.sin(self._theta))
        )
        return foci
    
    def to_foci_directrix(self) -> Tuple[
            Tuple[Tuple[float, float], Tuple[float, float]], 
            Tuple[Tuple[float, float], Tuple[float, float]]
        ]:
        """
        Return the foci and the directrix of the ellipse as a tuplet. Since an ellipse
        has two foci and two directrices, the return value is a tuplet of two tuplets.
        """
        c = np.sqrt(self._a**2 - self._b**2)
        foci = (
            (self._center[0] + c * np.cos(self._theta), self._center[1] + c * np.sin(self._theta)),
            (self._center[0] - c * np.cos(self._theta), self._center[1] - c * np.sin(self._theta))
        )

        d = self._a / self.eccentricity

        points_on_directrices = (
            (self._center[0] + d * np.cos(self._theta), self._center[1] + d * np.sin(self._theta)),
            (self._center[0] - d * np.cos(self._theta), self._center[1] - d * np.sin(self._theta))
        )

        normal = np.array([np.cos(self._theta), np.sin(self._theta)])

        res = []
        for pod, focus in zip(points_on_directrices, foci):
            L = np.dot(np.array(pod), normal)
            if L < 0:
                normal = -normal
                L = -L

            theta = np.arctan2(normal[0], normal[1])
            res.append((focus, (L, theta)))

        return tuple(res)


    @classmethod
    def from_general_cartesian_form(
        cls,
        A: float,
        B: float,
        C: float,
        D: float,
        E: float,
        F: float
    ) -> "Ellipse":
        """
        Create an ellipse from the general Cartesian form of the equation of
        an ellipse:
        Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
        """
        if not Ellipse.is_ellipse(A, B, C, D, E, F):
            raise ValueError("The given coefficients do not represent an ellipse")
        
        h = (2*C*D - B*E) / (B**2 - 4*A*C)
        k = (2*A*E - B*D) / (B**2 - 4*A*C)
        theta = 0.5 * np.arctan2(-B, C - A)
        term = 2*(A*E**2 + C*D**2 - B*D*E + (B**2 - 4*A*C)*F)
        a = -np.sqrt(term * ((A + C) + np.sqrt((A - C)**2 + B**2))) / (B**2 - 4*A*C)
        b = -np.sqrt(term * ((A + C) - np.sqrt((A - C)**2 + B**2))) / (B**2 - 4*A*C)

        return cls(a, b, (h, k), theta)
        

    
    @classmethod
    def from_foci_tuplet(
        cls,
        foci: Tuple[Tuple[float, float], Tuple[float, float]],
        major_axis_length: float
    ) -> "Ellipse":
        """
        Create an ellipse from a foci tuplet and the length of the major axis.
        """
        raise NotImplementedError
    
    @classmethod
    def from_foci_directrix(
        cls,
        foci: Tuple[float, float],
        directrix: Tuple[float, float],
        eccentricity: float
    ) -> "Ellipse":
        """
        Create an ellipse from an eccentricity, a foci and a directrix.
        """
        raise NotImplementedError
    
    @staticmethod
    def is_ellipse(
        A: float,
        B: float,
        C: float,
        D: float,
        E: float,
        F: float
    ) -> bool:
        """
        Return True if the general Cartesian form of the equation of an ellipse
        is valid, otherwise return False.
        """
        matrix = np.array([
            [A, B / 2, D / 2],
            [B / 2, C, E / 2],
            [D / 2, E / 2, F]
        ])
        if np.linalg.det(matrix) == 0:
            return False

        if B**2 - 4 * A * C < 0:
            return True
        
        return False
    
    def visualize(self) -> None:
        """
        Visualize the ellipse.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        ellipse = patches.Ellipse((self._center[0], self._center[1]), 2*self._a, 2*self._b, angle=self._theta, edgecolor='red', facecolor='none')
        ax.add_patch(ellipse)
        ax.set_xlim(self._center[0] - self._a - 1, self._center[0] + self._a + 1)
        ax.set_ylim(self._center[1] - self._b - 1, self._center[1] + self._b + 1)
        plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
        plt.show()
