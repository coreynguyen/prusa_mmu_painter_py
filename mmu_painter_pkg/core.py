"""
Core data structures and constants for MMU Painter
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum, auto

# ============================================================================
# Constants
# ============================================================================

DEFAULT_COLORS = [
    (128, 128, 128),  # 0: Gray (default/unpainted)
    (255, 80, 80),    # 1: Red
    (80, 255, 80),    # 2: Green
    (80, 80, 255),    # 3: Blue
    (255, 255, 80),   # 4: Yellow
    (255, 80, 255),   # 5: Magenta
    (80, 255, 255),   # 6: Cyan
    (255, 160, 80),   # 7: Orange
    (160, 80, 255),   # 8: Purple
]

MAX_UNDO = 50
MAX_SUBDIVISION_DEPTH = 10


# ============================================================================
# Enums
# ============================================================================

class PaintTool(Enum):
    PAINT = auto()
    MASK = auto()
    UNMASK = auto()
    ERASE = auto()
    EYEDROPPER = auto()


class TextureMode(Enum):
    NONE = auto()
    COLOR = auto()
    SEGMENTED = auto()


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class SubTriangle:
    """A painted sub-region of a triangle in barycentric coords"""
    bary_corners: List[Tuple[float, float, float]]  # 3 barycentric coords
    extruder_id: int = 0
    depth: int = 0
    masked: bool = False
    
    def get_center_bary(self) -> Tuple[float, float, float]:
        c = self.bary_corners
        return (
            (c[0][0] + c[1][0] + c[2][0]) / 3,
            (c[0][1] + c[1][1] + c[2][1]) / 3,
            (c[0][2] + c[1][2] + c[2][2]) / 3
        )
    
    def copy(self) -> 'SubTriangle':
        return SubTriangle(
            bary_corners=[tuple(c) for c in self.bary_corners],
            extruder_id=self.extruder_id,
            depth=self.depth,
            masked=self.masked
        )


@dataclass
class Triangle:
    """A mesh triangle with paint data"""
    v_idx: Tuple[int, int, int]  # Vertex indices
    uv: Optional[Tuple[Tuple[float, float], ...]] = None  # UV coords per vertex
    normal: Optional[Tuple[float, float, float]] = None
    paint_data: List[SubTriangle] = field(default_factory=list)
    mmu_segmentation: str = ""
    masked: bool = False
    
    def __post_init__(self):
        if not self.paint_data:
            self.paint_data = [SubTriangle(
                bary_corners=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
                extruder_id=0,
                depth=0
            )]
    
    def copy_paint_data(self) -> List[SubTriangle]:
        """Deep copy paint data for undo system"""
        return [sub.copy() for sub in self.paint_data]


@dataclass
class Mesh:
    """3D mesh with painting support"""
    vertices: List[Tuple[float, float, float]] = field(default_factory=list)
    triangles: List[Triangle] = field(default_factory=list)
    texture_path: Optional[str] = None
    
    # Computed bounds
    bounds_min: Tuple[float, float, float] = (0, 0, 0)
    bounds_max: Tuple[float, float, float] = (1, 1, 1)
    center: Tuple[float, float, float] = (0, 0, 0)
    size: float = 1.0
    
    def compute_bounds(self):
        """Compute mesh bounding box"""
        if not self.vertices:
            return
        
        try:
            import numpy as np
            verts = np.array(self.vertices)
            self.bounds_min = tuple(verts.min(axis=0))
            self.bounds_max = tuple(verts.max(axis=0))
        except ImportError:
            xs, ys, zs = zip(*self.vertices)
            self.bounds_min = (min(xs), min(ys), min(zs))
            self.bounds_max = (max(xs), max(ys), max(zs))
        
        self.center = tuple((a + b) / 2 for a, b in zip(self.bounds_min, self.bounds_max))
        self.size = max(b - a for a, b in zip(self.bounds_min, self.bounds_max))
    
    def compute_normals(self):
        """Compute triangle normals"""
        from .math_utils import cross, normalize
        
        for tri in self.triangles:
            v0 = self.vertices[tri.v_idx[0]]
            v1 = self.vertices[tri.v_idx[1]]
            v2 = self.vertices[tri.v_idx[2]]
            
            e1 = (v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2])
            e2 = (v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2])
            n = cross(e1, e2)
            tri.normal = normalize(n)
    
    def get_triangle_size(self, tri_idx: int) -> float:
        """Get approximate size of a triangle"""
        tri = self.triangles[tri_idx]
        v0 = self.vertices[tri.v_idx[0]]
        v1 = self.vertices[tri.v_idx[1]]
        v2 = self.vertices[tri.v_idx[2]]
        
        import math
        e1 = math.sqrt(sum((a-b)**2 for a, b in zip(v0, v1)))
        e2 = math.sqrt(sum((a-b)**2 for a, b in zip(v1, v2)))
        e3 = math.sqrt(sum((a-b)**2 for a, b in zip(v2, v0)))
        
        return (e1 + e2 + e3) / 3
