"""
Math utilities for MMU Painter - NumPy accelerated when available
"""

import math

# Check for NumPy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ============================================================================
# Vector Operations
# ============================================================================

if HAS_NUMPY:
    def normalize(v):
        """Normalize a vector"""
        if isinstance(v, (list, tuple)):
            v = np.array(v, dtype=np.float64)
        n = np.linalg.norm(v)
        if n > 1e-10:
            return tuple(v / n)
        return tuple(v)
    
    def cross(a, b):
        """Cross product of two vectors"""
        if isinstance(a, (list, tuple)):
            a = np.array(a, dtype=np.float64)
        if isinstance(b, (list, tuple)):
            b = np.array(b, dtype=np.float64)
        result = np.cross(a, b)
        return tuple(result)
    
    def dot(a, b):
        """Dot product of two vectors"""
        if isinstance(a, (list, tuple)):
            a = np.array(a, dtype=np.float64)
        if isinstance(b, (list, tuple)):
            b = np.array(b, dtype=np.float64)
        return float(np.dot(a, b))
    
    def length(v):
        """Length of a vector"""
        if isinstance(v, (list, tuple)):
            v = np.array(v, dtype=np.float64)
        return float(np.linalg.norm(v))
    
    def vec3(x, y, z):
        """Create a 3D vector"""
        return np.array([x, y, z], dtype=np.float64)
    
    def distance(a, b):
        """Distance between two points"""
        if isinstance(a, (list, tuple)):
            a = np.array(a, dtype=np.float64)
        if isinstance(b, (list, tuple)):
            b = np.array(b, dtype=np.float64)
        return float(np.linalg.norm(a - b))

else:
    def normalize(v):
        """Normalize a vector"""
        n = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
        if n > 1e-10:
            return (v[0]/n, v[1]/n, v[2]/n)
        return v
    
    def cross(a, b):
        """Cross product of two vectors"""
        return (
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]
        )
    
    def dot(a, b):
        """Dot product of two vectors"""
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
    
    def length(v):
        """Length of a vector"""
        return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    
    def vec3(x, y, z):
        """Create a 3D vector"""
        return (x, y, z)
    
    def distance(a, b):
        """Distance between two points"""
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


# ============================================================================
# Barycentric Coordinates
# ============================================================================

def bary_to_world(bary, v0, v1, v2):
    """Convert barycentric coordinates to world coordinates"""
    return (
        bary[0] * v0[0] + bary[1] * v1[0] + bary[2] * v2[0],
        bary[0] * v0[1] + bary[1] * v1[1] + bary[2] * v2[1],
        bary[0] * v0[2] + bary[1] * v1[2] + bary[2] * v2[2]
    )


def midpoint_bary(b1, b2):
    """Midpoint of two barycentric coordinates"""
    return (
        (b1[0] + b2[0]) / 2,
        (b1[1] + b2[1]) / 2,
        (b1[2] + b2[2]) / 2
    )


def subdivide_bary_corners(corners):
    """
    Subdivide a triangle (in barycentric coords) into 4 sub-triangles.
    Returns list of 4 corner triplets: [center, bottom-right, top, bottom-left]
    """
    c0, c1, c2 = corners
    m01 = midpoint_bary(c0, c1)
    m12 = midpoint_bary(c1, c2)
    m02 = midpoint_bary(c0, c2)
    
    return [
        [m01, m12, m02],  # Center
        [m12, c2, m02],   # Bottom-right (corner 2)
        [m01, c1, m12],   # Top (corner 1)
        [c0, m01, m02],   # Bottom-left (corner 0)
    ]


# ============================================================================
# Triangle Utilities
# ============================================================================

def point_in_triangle_2d(p, v0, v1, v2):
    """Check if point p is inside triangle v0,v1,v2 (2D)"""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    d1 = sign(p, v0, v1)
    d2 = sign(p, v1, v2)
    d3 = sign(p, v2, v0)
    
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    
    return not (has_neg and has_pos)


def triangle_area(v0, v1, v2):
    """Calculate area of a 3D triangle"""
    e1 = (v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2])
    e2 = (v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2])
    c = cross(e1, e2)
    return length(c) / 2
