import cv2
import numpy as np
from smolagents import tool


@tool
def get_contour_area(contour: list[list[int | float]]) -> dict:
    """Calculate the area of a contour using cv2.contourArea.

    The tool returns:
        A dict with the key 'contour_area' mapping to the area of the contour.

    Args:
        contour: A list of points representing the contour.
    """
    contour_array = np.asarray(contour, dtype=np.float32)
    return {"contour_area": cv2.contourArea(contour_array)}


@tool
def get_contour_perimeter(contour: list[list[int | float]]) -> dict:
    """Calculate the perimeter of a contour using cv2.arcLength.

    The tool returns:
        A dict with the key 'contour_perimeter' mapping to the perimeter of the contour.

    Args:
        contour: A list of points representing the contour.
    """
    contour_array = np.asarray(contour, dtype=np.float32)
    return {"contour_perimeter": cv2.arcLength(contour_array, closed=True)}


@tool
def get_contour_convex_hull(contour: list[list[int | float]]) -> dict:
    """Calculate the convex hull of a contour using cv2.convexHull.

    The tool returns:
        A dict with the key 'contour_convex_hull' mapping to the convex hull of the contour.

    Args:
        contour: A list of points representing the contour.
    """
    contour_array = np.asarray(contour, dtype=np.float32)
    return {"contour_convex_hull": cv2.convexHull(contour_array)}
