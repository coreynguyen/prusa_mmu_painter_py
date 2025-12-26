"""
Fast File I/O for MMU Painter - NumPy accelerated with background loading
"""

import os
import re
import zipfile
import threading
from typing import Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor

# NumPy for fast array operations
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from .core import Mesh, Triangle, SubTriangle
from .codec import MMUCodec


class LoadProgress:
    """Progress tracker for loading operations"""
    def __init__(self, callback: Optional[Callable[[str, float], None]] = None):
        self.callback = callback
        self.cancelled = False
    
    def update(self, stage: str, progress: float):
        if self.callback and not self.cancelled:
            self.callback(stage, progress)
    
    def cancel(self):
        self.cancelled = True


def load_model_async(filepath: str, 
                     on_complete: Callable[[Mesh], None],
                     on_progress: Optional[Callable[[str, float], None]] = None,
                     on_error: Optional[Callable[[Exception], None]] = None) -> LoadProgress:
    """
    Load model in background thread.
    
    Args:
        filepath: Path to model file
        on_complete: Called with mesh when loading completes (on main thread)
        on_progress: Called with (stage_name, 0.0-1.0) during loading
        on_error: Called if loading fails
    
    Returns:
        LoadProgress object that can be used to cancel loading
    """
    progress = LoadProgress(on_progress)
    
    def load_thread():
        try:
            mesh = load_model_fast(filepath, progress)
            if not progress.cancelled:
                on_complete(mesh)
        except Exception as e:
            if on_error and not progress.cancelled:
                on_error(e)
    
    thread = threading.Thread(target=load_thread, daemon=True)
    thread.start()
    
    return progress


def load_model_fast(filepath: str, progress: Optional[LoadProgress] = None) -> Mesh:
    """Load model file with optimizations"""
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.obj':
        return load_obj_fast(filepath, progress)
    elif ext == '.3mf':
        return load_3mf_fast(filepath, progress)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def load_obj_fast(filepath: str, progress: Optional[LoadProgress] = None) -> Mesh:
    """
    Fast OBJ loader using NumPy for large files.
    ~5-10x faster than pure Python for large models.
    """
    if not HAS_NUMPY:
        # Fallback to standard loader
        from .io import load_obj
        return load_obj(filepath)
    
    if progress:
        progress.update("Reading file", 0.0)
    
    # Read entire file at once (faster than line-by-line for large files)
    with open(filepath, 'r') as f:
        content = f.read()
    
    if progress and progress.cancelled:
        return None
    
    lines = content.split('\n')
    total_lines = len(lines)
    
    if progress:
        progress.update("Parsing vertices", 0.1)
    
    # Pre-count elements for array allocation
    v_count = sum(1 for line in lines if line.startswith('v '))
    vt_count = sum(1 for line in lines if line.startswith('vt '))
    f_count = sum(1 for line in lines if line.startswith('f '))
    
    # Pre-allocate numpy arrays
    vertices = np.zeros((v_count, 3), dtype=np.float32)
    uvs = np.zeros((vt_count, 2), dtype=np.float32) if vt_count > 0 else None
    
    # Parse in single pass
    v_idx = 0
    vt_idx = 0
    faces = []
    face_uvs = []
    texture_path = None
    
    for i, line in enumerate(lines):
        if progress and i % 50000 == 0:
            progress.update("Parsing", 0.1 + 0.4 * (i / total_lines))
            if progress.cancelled:
                return None
        
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        if line.startswith('v '):
            parts = line.split()
            if len(parts) >= 4:
                vertices[v_idx] = [float(parts[1]), float(parts[2]), float(parts[3])]
                v_idx += 1
        
        elif line.startswith('vt ') and uvs is not None:
            parts = line.split()
            if len(parts) >= 3:
                uvs[vt_idx] = [float(parts[1]), float(parts[2])]
                vt_idx += 1
        
        elif line.startswith('f '):
            parts = line.split()[1:]
            indices = []
            uv_indices = []
            
            for p in parts:
                idx = p.split('/')
                indices.append(int(idx[0]) - 1)
                if len(idx) > 1 and idx[1]:
                    uv_indices.append(int(idx[1]) - 1)
            
            # Triangulate
            for j in range(1, len(indices) - 1):
                faces.append((indices[0], indices[j], indices[j+1]))
                if uv_indices and len(uv_indices) > j + 1:
                    face_uvs.append((uv_indices[0], uv_indices[j], uv_indices[j+1]))
                else:
                    face_uvs.append(None)
        
        elif line.startswith('mtllib '):
            mtl_path = os.path.join(os.path.dirname(filepath), line.split()[1])
            if os.path.exists(mtl_path):
                texture_path = _find_texture_in_mtl(mtl_path)
    
    if progress:
        progress.update("Building mesh", 0.6)
    
    # Trim arrays to actual size
    vertices = vertices[:v_idx]
    if uvs is not None:
        uvs = uvs[:vt_idx]
    
    # Detect and fix Z-up orientation
    if len(vertices) > 0:
        y_range = vertices[:, 1].max() - vertices[:, 1].min()
        z_range = vertices[:, 2].max() - vertices[:, 2].min()
        z_min = vertices[:, 2].min()
        
        if z_range > y_range * 1.5 and z_min >= -0.1:
            # Z-up to Y-up conversion
            new_vertices = np.zeros_like(vertices)
            new_vertices[:, 0] = vertices[:, 0]
            new_vertices[:, 1] = vertices[:, 2]
            new_vertices[:, 2] = -vertices[:, 1]
            vertices = new_vertices
    
    if progress:
        progress.update("Creating triangles", 0.7)
    
    # Build mesh
    mesh = Mesh()
    mesh.vertices = [tuple(v) for v in vertices]
    mesh.texture_path = texture_path
    
    # Create triangles in batches for better performance
    total_faces = len(faces)
    for i, (face, uv_idx) in enumerate(zip(faces, face_uvs)):
        if progress and i % 20000 == 0:
            progress.update("Creating triangles", 0.7 + 0.2 * (i / total_faces))
            if progress.cancelled:
                return None
        
        tri = Triangle(v_idx=face)
        
        if uv_idx is not None and uvs is not None:
            try:
                tri.uv = (
                    tuple(uvs[uv_idx[0]]),
                    tuple(uvs[uv_idx[1]]),
                    tuple(uvs[uv_idx[2]])
                )
            except IndexError:
                pass
        
        mesh.triangles.append(tri)
    
    if progress:
        progress.update("Computing bounds", 0.9)
    
    mesh.compute_bounds()
    
    if progress:
        progress.update("Computing normals", 0.95)
    
    mesh.compute_normals()
    
    if progress:
        progress.update("Complete", 1.0)
    
    return mesh


def load_3mf_fast(filepath: str, progress: Optional[LoadProgress] = None) -> Mesh:
    """Fast 3MF loader"""
    if progress:
        progress.update("Reading archive", 0.0)
    
    mesh = Mesh()
    
    with zipfile.ZipFile(filepath, 'r') as zf:
        model_file = None
        for name in zf.namelist():
            if '3dmodel.model' in name.lower() or name.endswith('.model'):
                model_file = name
                break
        
        if not model_file:
            raise ValueError("No model file found in 3MF archive")
        
        if progress:
            progress.update("Parsing model", 0.2)
        
        content = zf.read(model_file).decode('utf-8')
        
        # Extract all vertices at once using regex
        vertices = []
        for m in re.finditer(r'<vertex\s+x="([^"]+)"\s+y="([^"]+)"\s+z="([^"]+)"', content):
            x, y, z = float(m.group(1)), float(m.group(2)), float(m.group(3))
            vertices.append((x, z, -y))  # Z-up to Y-up
        
        mesh.vertices = vertices
        
        if progress:
            progress.update("Parsing triangles", 0.5)
            if progress.cancelled:
                return None
        
        # Extract triangles
        tri_pattern = r'<triangle\s+v1="(\d+)"\s+v2="(\d+)"\s+v3="(\d+)"([^/]*)/?\s*>'
        matches = list(re.finditer(tri_pattern, content))
        total = len(matches)
        
        for i, m in enumerate(matches):
            if progress and i % 10000 == 0:
                progress.update("Parsing triangles", 0.5 + 0.3 * (i / total))
                if progress.cancelled:
                    return None
            
            i0, i1, i2 = int(m.group(1)), int(m.group(2)), int(m.group(3))
            attrs = m.group(4)
            
            tri = Triangle(v_idx=(i0, i1, i2))
            
            mmu_match = re.search(
                r'(?:slic3rpe:mmu_segmentation|paint_color)="([^"]*)"',
                attrs
            )
            if mmu_match and mmu_match.group(1):
                tri.mmu_segmentation = mmu_match.group(1)
                tri.paint_data = MMUCodec.decode(tri.mmu_segmentation)
            
            mesh.triangles.append(tri)
    
    if progress:
        progress.update("Computing bounds", 0.9)
    
    mesh.compute_bounds()
    mesh.compute_normals()
    
    if progress:
        progress.update("Complete", 1.0)
    
    return mesh


def _find_texture_in_mtl(mtl_path: str) -> Optional[str]:
    """Find texture path from MTL file"""
    base_dir = os.path.dirname(mtl_path)
    
    try:
        with open(mtl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('map_Kd') or line.startswith('map_Ka'):
                    tex_name = line.split(None, 1)[1]
                    tex_path = os.path.join(base_dir, tex_name)
                    if os.path.exists(tex_path):
                        return tex_path
    except Exception:
        pass
    
    return None


# ============================================================================
# OPTIONAL: Numba JIT acceleration (10-50x faster for critical paths)
# ============================================================================

try:
    from numba import jit, prange
    HAS_NUMBA = True
    
    @jit(nopython=True, parallel=True, cache=True)
    def compute_normals_fast(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """Compute face normals using Numba (parallel)"""
        n = len(faces)
        normals = np.zeros((n, 3), dtype=np.float32)
        
        for i in prange(n):
            i0, i1, i2 = faces[i]
            v0 = vertices[i0]
            v1 = vertices[i1]
            v2 = vertices[i2]
            
            e1 = v1 - v0
            e2 = v2 - v0
            
            # Cross product
            nx = e1[1]*e2[2] - e1[2]*e2[1]
            ny = e1[2]*e2[0] - e1[0]*e2[2]
            nz = e1[0]*e2[1] - e1[1]*e2[0]
            
            # Normalize
            length = np.sqrt(nx*nx + ny*ny + nz*nz)
            if length > 1e-6:
                normals[i, 0] = nx / length
                normals[i, 1] = ny / length
                normals[i, 2] = nz / length
            else:
                normals[i, 1] = 1.0
        
        return normals

except ImportError:
    HAS_NUMBA = False
