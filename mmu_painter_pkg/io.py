"""
File I/O for MMU Painter - Load/save OBJ and 3MF files
"""

import os
import re
import zipfile
from typing import Optional

from .core import Mesh, Triangle, SubTriangle
from .codec import MMUCodec


def load_model(filepath: str) -> Mesh:
    """Load model file (auto-detect format by extension)"""
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.obj':
        return load_obj(filepath)
    elif ext == '.3mf':
        return load_3mf(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def load_obj(filepath: str) -> Mesh:
    """
    Load OBJ file with automatic Z-up to Y-up conversion.
    Also loads UVs and attempts to find texture from MTL file.
    """
    mesh = Mesh()
    uvs = []
    normals = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if not parts:
                continue
            
            cmd = parts[0]
            
            if cmd == 'v' and len(parts) >= 4:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                mesh.vertices.append((x, y, z))
                
            elif cmd == 'vt' and len(parts) >= 3:
                uvs.append((float(parts[1]), float(parts[2])))
                
            elif cmd == 'vn' and len(parts) >= 4:
                normals.append((float(parts[1]), float(parts[2]), float(parts[3])))
                
            elif cmd == 'f':
                indices = []
                uv_indices = []
                normal_indices = []
                
                for p in parts[1:]:
                    idx = p.split('/')
                    indices.append(int(idx[0]) - 1)
                    if len(idx) > 1 and idx[1]:
                        uv_indices.append(int(idx[1]) - 1)
                    if len(idx) > 2 and idx[2]:
                        normal_indices.append(int(idx[2]) - 1)
                
                # Triangulate faces with more than 3 vertices
                for i in range(1, len(indices) - 1):
                    tri = Triangle(v_idx=(indices[0], indices[i], indices[i+1]))
                    
                    # Assign UVs if available
                    if uv_indices and len(uv_indices) > i + 1:
                        try:
                            tri.uv = (
                                uvs[uv_indices[0]],
                                uvs[uv_indices[i]],
                                uvs[uv_indices[i+1]]
                            )
                        except IndexError:
                            pass
                    
                    mesh.triangles.append(tri)
                    
            elif cmd == 'mtllib':
                # Look for texture in MTL file
                mtl_path = os.path.join(os.path.dirname(filepath), parts[1])
                if os.path.exists(mtl_path):
                    mesh.texture_path = _find_texture_in_mtl(mtl_path)
    
    # Auto-detect and fix Z-up orientation
    mesh = _fix_orientation(mesh)
    
    mesh.compute_bounds()
    mesh.compute_normals()
    
    return mesh


def _find_texture_in_mtl(mtl_path: str) -> Optional[str]:
    """Find texture path from MTL file"""
    base_dir = os.path.dirname(mtl_path)
    
    try:
        with open(mtl_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Look for diffuse or ambient texture maps
                if line.startswith('map_Kd') or line.startswith('map_Ka'):
                    tex_name = line.split(None, 1)[1]
                    tex_path = os.path.join(base_dir, tex_name)
                    if os.path.exists(tex_path):
                        return tex_path
    except Exception:
        pass
    
    return None


def _fix_orientation(mesh: Mesh) -> Mesh:
    """
    Detect if mesh is Z-up and convert to Y-up if needed.
    
    Heuristic: If Z range >> Y range and min(Z) is near 0, likely Z-up.
    """
    if not mesh.vertices:
        return mesh
    
    ys = [v[1] for v in mesh.vertices]
    zs = [v[2] for v in mesh.vertices]
    
    y_range = max(ys) - min(ys)
    z_range = max(zs) - min(zs)
    
    # If Z range is much larger and model sits on Z=0 plane, convert
    if z_range > y_range * 1.5 and min(zs) >= -0.1:
        # Convert Z-up to Y-up: swap Y and Z, negate new Z
        mesh.vertices = [(v[0], v[2], -v[1]) for v in mesh.vertices]
    
    return mesh


def load_3mf(filepath: str) -> Mesh:
    """
    Load 3MF file with MMU segmentation data.
    3MF files are Z-up by convention, so we convert to Y-up.
    """
    mesh = Mesh()
    
    with zipfile.ZipFile(filepath, 'r') as zf:
        # Find the model file
        model_file = None
        for name in zf.namelist():
            if '3dmodel.model' in name.lower() or name.endswith('.model'):
                model_file = name
                break
        
        if not model_file:
            raise ValueError("No model file found in 3MF archive")
        
        content = zf.read(model_file).decode('utf-8')
        
        # Parse vertices (convert Z-up to Y-up)
        for m in re.finditer(r'<vertex\s+x="([^"]+)"\s+y="([^"]+)"\s+z="([^"]+)"', content):
            x, y, z = float(m.group(1)), float(m.group(2)), float(m.group(3))
            # Z-up to Y-up: (x, y, z) -> (x, z, -y)
            mesh.vertices.append((x, z, -y))
        
        # Parse triangles with MMU segmentation
        tri_pattern = r'<triangle\s+v1="(\d+)"\s+v2="(\d+)"\s+v3="(\d+)"([^/]*)/?\s*>'
        for m in re.finditer(tri_pattern, content):
            i0, i1, i2 = int(m.group(1)), int(m.group(2)), int(m.group(3))
            attrs = m.group(4)
            
            tri = Triangle(v_idx=(i0, i1, i2))
            
            # Look for MMU segmentation attribute (PrusaSlicer or OrcaSlicer)
            mmu_match = re.search(
                r'(?:slic3rpe:mmu_segmentation|paint_color)="([^"]*)"',
                attrs
            )
            if mmu_match and mmu_match.group(1):
                tri.mmu_segmentation = mmu_match.group(1)
                tri.paint_data = MMUCodec.decode(tri.mmu_segmentation)
            
            mesh.triangles.append(tri)
    
    mesh.compute_bounds()
    mesh.compute_normals()
    
    return mesh


def save_3mf(filepath: str, mesh: Mesh) -> bool:
    """
    Save mesh to 3MF with MMU segmentation.
    
    Note: This is a simplified exporter. A full implementation would
    preserve all original 3MF metadata and structure.
    """
    # TODO: Implement full 3MF export
    # This requires:
    # 1. Creating proper 3MF XML structure
    # 2. Converting Y-up back to Z-up
    # 3. Encoding paint data to MMU segmentation strings
    # 4. Packaging into ZIP archive
    
    raise NotImplementedError("3MF export not yet implemented")


def save_obj(filepath: str, mesh: Mesh) -> bool:
    """Save mesh to OBJ format (without paint data)"""
    try:
        with open(filepath, 'w') as f:
            f.write("# MMU Painter Export\n")
            f.write(f"# Vertices: {len(mesh.vertices)}\n")
            f.write(f"# Triangles: {len(mesh.triangles)}\n\n")
            
            # Write vertices
            for v in mesh.vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            
            f.write("\n")
            
            # Write faces (1-indexed)
            for tri in mesh.triangles:
                i0, i1, i2 = tri.v_idx
                f.write(f"f {i0+1} {i1+1} {i2+1}\n")
        
        return True
    except Exception as e:
        print(f"Error saving OBJ: {e}")
        return False
