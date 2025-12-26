"""
Brush and ray casting for MMU Painter - Heavily Optimized for Performance
"""

import math
from typing import List, Tuple, Optional, Dict
from .core import Mesh, Triangle, SubTriangle, PaintTool, MAX_SUBDIVISION_DEPTH

# Check for NumPy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class RayCaster:
    """
    Fast ray-mesh intersection using Möller–Trumbore algorithm.
    """
    
    def cast(self, origin: Tuple[float, float, float],
             direction: Tuple[float, float, float],
             mesh: Mesh) -> Optional[Tuple[int, Tuple[float, float, float], float]]:
        
        if not mesh or not mesh.triangles:
            return None
        
        ox, oy, oz = origin
        dx, dy, dz = direction
        
        # Normalize direction
        d_len = math.sqrt(dx*dx + dy*dy + dz*dz)
        if d_len < 1e-6: return None
        dx, dy, dz = dx/d_len, dy/d_len, dz/d_len
        
        best_hit = None
        best_t = float('inf')
        EPSILON = 1e-8

        for i, tri in enumerate(mesh.triangles):
            v0 = mesh.vertices[tri.v_idx[0]]
            v1 = mesh.vertices[tri.v_idx[1]]
            v2 = mesh.vertices[tri.v_idx[2]]
            
            e1x, e1y, e1z = v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]
            e2x, e2y, e2z = v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]
            
            hx = dy*e2z - dz*e2y
            hy = dz*e2x - dx*e2z
            hz = dx*e2y - dy*e2x
            
            a = e1x*hx + e1y*hy + e1z*hz
            
            if -EPSILON < a < EPSILON:
                continue
                
            f = 1.0 / a
            
            sx = ox - v0[0]
            sy = oy - v0[1]
            sz = oz - v0[2]
            
            u = f * (sx*hx + sy*hy + sz*hz)
            if u < 0.0 or u > 1.0:
                continue
                
            qx = sy*e1z - sz*e1y
            qy = sz*e1x - sx*e1z
            qz = sx*e1y - sy*e1x
            
            v = f * (dx*qx + dy*qy + dz*qz)
            if v < 0.0 or u + v > 1.0:
                continue
                
            t = f * (e2x*qx + e2y*qy + e2z*qz)
            
            if t > EPSILON and t < best_t:
                best_t = t
                w = 1.0 - u - v
                best_hit = (i, (w, u, v), t)
        
        return best_hit


class SphereBrush:
    """
    Optimized Sphere Brush with Iterative Subdivision (no recursion).
    Uses stack-based approach and spatial grid for much better performance.
    """
    
    def __init__(self):
        self.radius: float = 0.5
        self.position: Optional[Tuple[float, float, float]] = None
        self.normal: Optional[Tuple[float, float, float]] = None
        self.hover_tri_idx: int = -1
        
        self.auto_subdivide: bool = True
        self.max_depth: int = 6  # Default depth (6 = good balance of detail/performance)
        self.detail_ratio: float = 0.25  # Larger = faster, less detail
        
        # Spatial grid for faster triangle lookup
        self._grid: Dict[Tuple[int,int,int], List[int]] = {}
        self._grid_cell_size: float = 1.0
        self._grid_mesh = None
    
    def set_position(self, pos, normal, hover_idx: int = -1):
        self.position = pos
        self.normal = normal
        self.hover_tri_idx = hover_idx

    def _build_spatial_grid(self, mesh: Mesh):
        """Build spatial hash grid for fast triangle lookup"""
        if mesh is self._grid_mesh:
            return
        
        self._grid.clear()
        self._grid_mesh = mesh
        
        # Determine cell size based on mesh
        if mesh.size > 0:
            self._grid_cell_size = mesh.size / 10.0
        else:
            self._grid_cell_size = 1.0
        
        cell = self._grid_cell_size
        
        for i, tri in enumerate(mesh.triangles):
            v0 = mesh.vertices[tri.v_idx[0]]
            v1 = mesh.vertices[tri.v_idx[1]]
            v2 = mesh.vertices[tri.v_idx[2]]
            
            # Get bounding box of triangle
            min_x = min(v0[0], v1[0], v2[0])
            max_x = max(v0[0], v1[0], v2[0])
            min_y = min(v0[1], v1[1], v2[1])
            max_y = max(v0[1], v1[1], v2[1])
            min_z = min(v0[2], v1[2], v2[2])
            max_z = max(v0[2], v1[2], v2[2])
            
            # Add to all cells it overlaps
            for gx in range(int(min_x/cell), int(max_x/cell)+1):
                for gy in range(int(min_y/cell), int(max_y/cell)+1):
                    for gz in range(int(min_z/cell), int(max_z/cell)+1):
                        key = (gx, gy, gz)
                        if key not in self._grid:
                            self._grid[key] = []
                        self._grid[key].append(i)

    def find_affected_triangles(self, mesh: Mesh, ignore_mask: bool = False) -> List[int]:
        """Find triangles touching the brush sphere using spatial grid."""
        if self.position is None or mesh is None:
            return []
        
        # Build grid if needed
        self._build_spatial_grid(mesh)
        
        px, py, pz = self.position
        r = self.radius
        r_sq = r * r
        cell = self._grid_cell_size
        
        affected = set()
        
        # Always include hovered triangle
        if 0 <= self.hover_tri_idx < len(mesh.triangles):
            tri = mesh.triangles[self.hover_tri_idx]
            if ignore_mask or not tri.masked:
                affected.add(self.hover_tri_idx)
        
        # Get cells that brush overlaps
        min_gx = int((px - r) / cell)
        max_gx = int((px + r) / cell)
        min_gy = int((py - r) / cell)
        max_gy = int((py + r) / cell)
        min_gz = int((pz - r) / cell)
        max_gz = int((pz + r) / cell)
        
        # Check triangles in those cells
        checked = set()
        for gx in range(min_gx, max_gx + 1):
            for gy in range(min_gy, max_gy + 1):
                for gz in range(min_gz, max_gz + 1):
                    key = (gx, gy, gz)
                    if key not in self._grid:
                        continue
                    
                    for i in self._grid[key]:
                        if i in checked:
                            continue
                        checked.add(i)
                        
                        tri = mesh.triangles[i]
                        if not ignore_mask and tri.masked:
                            continue
                        
                        v0 = mesh.vertices[tri.v_idx[0]]
                        v1 = mesh.vertices[tri.v_idx[1]]
                        v2 = mesh.vertices[tri.v_idx[2]]
                        
                        # Quick vertex check
                        d0 = (v0[0]-px)**2 + (v0[1]-py)**2 + (v0[2]-pz)**2
                        d1 = (v1[0]-px)**2 + (v1[1]-py)**2 + (v1[2]-pz)**2
                        d2 = (v2[0]-px)**2 + (v2[1]-py)**2 + (v2[2]-pz)**2
                        
                        if d0 <= r_sq or d1 <= r_sq or d2 <= r_sq:
                            affected.add(i)
                            continue
                        
                        # Centroid check
                        cx = (v0[0] + v1[0] + v2[0]) / 3
                        cy = (v0[1] + v1[1] + v2[1]) / 3
                        cz = (v0[2] + v1[2] + v2[2]) / 3
                        
                        if (cx-px)**2 + (cy-py)**2 + (cz-pz)**2 <= r_sq:
                            affected.add(i)
                            continue
                        
                        # CRITICAL: Check if brush center is near/on triangle surface
                        # This catches small brush on large triangle
                        dist_sq = self._point_to_triangle_dist_sq(
                            (px, py, pz), v0, v1, v2
                        )
                        if dist_sq <= r_sq:
                            affected.add(i)
        
        return list(affected)

    def find_closest_triangle(self, mesh: Mesh, pos: Tuple[float, float, float]) -> int:
        """Find the triangle closest to the given position (for interpolated strokes)"""
        if mesh is None or not mesh.triangles:
            return -1
        
        self._build_spatial_grid(mesh)
        
        px, py, pz = pos
        cell = self._grid_cell_size
        
        # Check nearby cells
        gx, gy, gz = int(px/cell), int(py/cell), int(pz/cell)
        
        best_idx = -1
        best_dist = float('inf')
        
        # Search in expanding radius
        for radius in range(3):
            for dx in range(-radius, radius+1):
                for dy in range(-radius, radius+1):
                    for dz in range(-radius, radius+1):
                        key = (gx+dx, gy+dy, gz+dz)
                        if key not in self._grid:
                            continue
                        
                        for i in self._grid[key]:
                            tri = mesh.triangles[i]
                            v0 = mesh.vertices[tri.v_idx[0]]
                            v1 = mesh.vertices[tri.v_idx[1]]
                            v2 = mesh.vertices[tri.v_idx[2]]
                            
                            dist_sq = self._point_to_triangle_dist_sq(pos, v0, v1, v2)
                            if dist_sq < best_dist:
                                best_dist = dist_sq
                                best_idx = i
            
            # If we found something close enough, stop searching
            if best_dist < (self.radius * 2) ** 2:
                break
        
        return best_idx

    def _point_to_triangle_dist_sq(self, p, v0, v1, v2) -> float:
        """Compute squared distance from point to triangle surface."""
        # Edge vectors
        e0x, e0y, e0z = v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]
        e1x, e1y, e1z = v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]
        
        # Vector from v0 to point
        v0px, v0py, v0pz = p[0]-v0[0], p[1]-v0[1], p[2]-v0[2]
        
        # Dot products for barycentric
        d00 = e0x*e0x + e0y*e0y + e0z*e0z
        d01 = e0x*e1x + e0y*e1y + e0z*e1z
        d11 = e1x*e1x + e1y*e1y + e1z*e1z
        d20 = v0px*e0x + v0py*e0y + v0pz*e0z
        d21 = v0px*e1x + v0py*e1y + v0pz*e1z
        
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-10:
            return float('inf')
        
        inv_denom = 1.0 / denom
        v = (d11 * d20 - d01 * d21) * inv_denom
        w = (d00 * d21 - d01 * d20) * inv_denom
        u = 1.0 - v - w
        
        # Clamp to triangle and find closest point
        if u >= 0 and v >= 0 and w >= 0:
            # Inside triangle - project to plane
            cx = v0[0] + e0x*v + e1x*w
            cy = v0[1] + e0y*v + e1y*w
            cz = v0[2] + e0z*v + e1z*w
        else:
            # Outside - find closest point on edges
            # Edge v0-v1
            t01 = max(0, min(1, d20/d00 if d00 > 1e-10 else 0))
            c01 = (v0[0]+e0x*t01, v0[1]+e0y*t01, v0[2]+e0z*t01)
            d01_sq = (p[0]-c01[0])**2 + (p[1]-c01[1])**2 + (p[2]-c01[2])**2
            
            # Edge v0-v2
            t02 = max(0, min(1, d21/d11 if d11 > 1e-10 else 0))
            c02 = (v0[0]+e1x*t02, v0[1]+e1y*t02, v0[2]+e1z*t02)
            d02_sq = (p[0]-c02[0])**2 + (p[1]-c02[1])**2 + (p[2]-c02[2])**2
            
            # Edge v1-v2
            e2x, e2y, e2z = v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2]
            v1px, v1py, v1pz = p[0]-v1[0], p[1]-v1[1], p[2]-v1[2]
            d22 = e2x*e2x + e2y*e2y + e2z*e2z
            d2p = v1px*e2x + v1py*e2y + v1pz*e2z
            t12 = max(0, min(1, d2p/d22 if d22 > 1e-10 else 0))
            c12 = (v1[0]+e2x*t12, v1[1]+e2y*t12, v1[2]+e2z*t12)
            d12_sq = (p[0]-c12[0])**2 + (p[1]-c12[1])**2 + (p[2]-c12[2])**2
            
            # Find minimum
            if d01_sq <= d02_sq and d01_sq <= d12_sq:
                return d01_sq
            elif d02_sq <= d12_sq:
                return d02_sq
            else:
                return d12_sq
        
        dx, dy, dz = p[0]-cx, p[1]-cy, p[2]-cz
        return dx*dx + dy*dy + dz*dz

    def paint_triangle(self, mesh: Mesh, tri_idx: int, color_id: int):
        """Paint triangle - respects masks"""
        tri = mesh.triangles[tri_idx]
        if tri.masked:
            return
        self._apply_action_fast(mesh, tri_idx, PaintTool.PAINT, color_id)

    def erase_triangle(self, mesh: Mesh, tri_idx: int):
        """Erase triangle - respects masks"""
        tri = mesh.triangles[tri_idx]
        if tri.masked:
            return
        self._apply_action_fast(mesh, tri_idx, PaintTool.ERASE, 0)

    def mask_triangle(self, mesh: Mesh, tri_idx: int):
        """Mask with subdivision"""
        self._apply_action_fast(mesh, tri_idx, PaintTool.MASK, 1)

    def unmask_triangle(self, mesh: Mesh, tri_idx: int):
        """Unmask with subdivision"""
        self._apply_action_fast(mesh, tri_idx, PaintTool.UNMASK, 0)

    def _apply_action_fast(self, mesh: Mesh, tri_idx: int, tool: PaintTool, val: int):
        """Iterative (non-recursive) subdivision for better performance"""
        if not self.position:
            return
        
        tri = mesh.triangles[tri_idx]
        v0 = mesh.vertices[tri.v_idx[0]]
        v1 = mesh.vertices[tri.v_idx[1]]
        v2 = mesh.vertices[tri.v_idx[2]]
        
        # Pre-compute
        px, py, pz = self.position
        r_sq = self.radius ** 2
        target_size_sq = (self.radius * self.detail_ratio) ** 2
        max_depth = self.max_depth
        
        # Convert vertices to tuples for faster access
        v0 = (v0[0], v0[1], v0[2])
        v1 = (v1[0], v1[1], v1[2])
        v2 = (v2[0], v2[1], v2[2])
        
        result_list = []
        
        # Stack-based iteration: (bary_corners, extruder_id, depth, masked)
        stack = []
        for sub in tri.paint_data:
            stack.append((sub.bary_corners, sub.extruder_id, sub.depth, sub.masked))
        
        while stack:
            bary, ext_id, depth, masked = stack.pop()
            
            # Compute world positions inline
            b0, b1, b2 = bary[0], bary[1], bary[2]
            
            p0x = b0[0]*v0[0] + b0[1]*v1[0] + b0[2]*v2[0]
            p0y = b0[0]*v0[1] + b0[1]*v1[1] + b0[2]*v2[1]
            p0z = b0[0]*v0[2] + b0[1]*v1[2] + b0[2]*v2[2]
            
            p1x = b1[0]*v0[0] + b1[1]*v1[0] + b1[2]*v2[0]
            p1y = b1[0]*v0[1] + b1[1]*v1[1] + b1[2]*v2[1]
            p1z = b1[0]*v0[2] + b1[1]*v1[2] + b1[2]*v2[2]
            
            p2x = b2[0]*v0[0] + b2[1]*v1[0] + b2[2]*v2[0]
            p2y = b2[0]*v0[1] + b2[1]*v1[1] + b2[2]*v2[1]
            p2z = b2[0]*v0[2] + b2[1]*v1[2] + b2[2]*v2[2]
            
            # Distance checks
            d0 = (p0x-px)**2 + (p0y-py)**2 + (p0z-pz)**2
            d1 = (p1x-px)**2 + (p1y-py)**2 + (p1z-pz)**2
            d2 = (p2x-px)**2 + (p2y-py)**2 + (p2z-pz)**2
            
            in0 = d0 <= r_sq
            in1 = d1 <= r_sq
            in2 = d2 <= r_sq
            
            # Centroid
            cx = (p0x + p1x + p2x) / 3
            cy = (p0y + p1y + p2y) / 3
            cz = (p0z + p1z + p2z) / 3
            dc = (cx-px)**2 + (cy-py)**2 + (cz-pz)**2
            
            # Check if brush center is inside this sub-triangle
            # (critical for small brush on large triangle)
            brush_inside = self._point_in_triangle_fast(
                px, py, pz, 
                p0x, p0y, p0z, 
                p1x, p1y, p1z, 
                p2x, p2y, p2z
            )
            
            # CASE A: Fully inside brush
            if in0 and in1 and in2:
                new_ext, new_mask = self._apply_tool(tool, val, ext_id, masked)
                result_list.append(SubTriangle(list(bary), new_ext, depth, new_mask))
                continue
            
            # CASE B: Fully outside - all corners out AND brush not inside tri AND centroid out
            if not in0 and not in1 and not in2 and not brush_inside and dc > r_sq:
                result_list.append(SubTriangle(list(bary), ext_id, depth, masked))
                continue
            
            # Size check
            e0_sq = (p1x-p0x)**2 + (p1y-p0y)**2 + (p1z-p0z)**2
            e1_sq = (p2x-p1x)**2 + (p2y-p1y)**2 + (p2z-p1z)**2
            e2_sq = (p0x-p2x)**2 + (p0y-p2y)**2 + (p0z-p2z)**2
            size_sq = max(e0_sq, e1_sq, e2_sq)
            
            # At max depth or small enough
            if size_sq <= target_size_sq or depth >= max_depth:
                # Paint if centroid in brush OR brush inside sub-tri
                if dc <= r_sq or brush_inside:
                    new_ext, new_mask = self._apply_tool(tool, val, ext_id, masked)
                    result_list.append(SubTriangle(list(bary), new_ext, depth, new_mask))
                else:
                    result_list.append(SubTriangle(list(bary), ext_id, depth, masked))
                continue
            
            # Subdivide - push 4 children to stack
            if self.auto_subdivide:
                c0, c1, c2 = b0, b1, b2
                m01 = ((c0[0]+c1[0])*0.5, (c0[1]+c1[1])*0.5, (c0[2]+c1[2])*0.5)
                m12 = ((c1[0]+c2[0])*0.5, (c1[1]+c2[1])*0.5, (c1[2]+c2[2])*0.5)
                m02 = ((c0[0]+c2[0])*0.5, (c0[1]+c2[1])*0.5, (c0[2]+c2[2])*0.5)
                
                next_depth = depth + 1
                stack.append(([m01, m12, m02], ext_id, next_depth, masked))
                stack.append(([c0, m01, m02], ext_id, next_depth, masked))
                stack.append(([m01, c1, m12], ext_id, next_depth, masked))
                stack.append(([m02, m12, c2], ext_id, next_depth, masked))
            else:
                if dc <= r_sq:
                    new_ext, new_mask = self._apply_tool(tool, val, ext_id, masked)
                    result_list.append(SubTriangle(list(bary), new_ext, depth, new_mask))
                else:
                    result_list.append(SubTriangle(list(bary), ext_id, depth, masked))
        
        tri.paint_data = result_list

    def _apply_tool(self, tool: PaintTool, val: int, ext_id: int, masked: bool) -> Tuple[int, bool]:
        """Apply tool and return new (extruder_id, masked)"""
        if tool == PaintTool.PAINT:
            if not masked:
                return (val, masked)
        elif tool == PaintTool.ERASE:
            if not masked:
                return (0, masked)
        elif tool == PaintTool.MASK:
            return (ext_id, True)
        elif tool == PaintTool.UNMASK:
            return (ext_id, False)
        return (ext_id, masked)

    def _point_in_triangle_fast(self, px, py, pz, 
                                 t0x, t0y, t0z, 
                                 t1x, t1y, t1z, 
                                 t2x, t2y, t2z) -> bool:
        """
        Fast check if point (px,py,pz) is inside triangle using barycentric coords.
        Inline computation for performance.
        """
        # Vectors from t0
        v0x, v0y, v0z = t2x-t0x, t2y-t0y, t2z-t0z
        v1x, v1y, v1z = t1x-t0x, t1y-t0y, t1z-t0z
        v2x, v2y, v2z = px-t0x, py-t0y, pz-t0z
        
        # Dot products
        dot00 = v0x*v0x + v0y*v0y + v0z*v0z
        dot01 = v0x*v1x + v0y*v1y + v0z*v1z
        dot02 = v0x*v2x + v0y*v2y + v0z*v2z
        dot11 = v1x*v1x + v1y*v1y + v1z*v1z
        dot12 = v1x*v2x + v1y*v2y + v1z*v2z
        
        # Barycentric coords
        denom = dot00 * dot11 - dot01 * dot01
        if abs(denom) < 1e-10:
            return False
            
        inv_denom = 1.0 / denom
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        
        # Check if point is in triangle (with small margin)
        margin = 0.02
        return (u >= -margin) and (v >= -margin) and (u + v <= 1 + margin)


class PaintOperation:
    """
    Encapsulates a paint operation for undo/redo.
    """
    def __init__(self, mesh: Mesh, affected_indices: List[int]):
        self.triangle_states: Dict[int, List[SubTriangle]] = {}
        self.triangle_masks: Dict[int, bool] = {}
        for idx in affected_indices:
            if idx < len(mesh.triangles):
                tri = mesh.triangles[idx]
                self.triangle_states[idx] = tri.copy_paint_data()
                self.triangle_masks[idx] = tri.masked
    
    def restore(self, mesh: Mesh):
        for idx, paint_data in self.triangle_states.items():
            if idx < len(mesh.triangles):
                mesh.triangles[idx].paint_data = [sub.copy() for sub in paint_data]
                if idx in self.triangle_masks:
                    mesh.triangles[idx].masked = self.triangle_masks[idx]