"""
OpenGL 3D Viewport for MMU Painter - VBO Accelerated & Robust Raycasting
"""

import math
import ctypes
from typing import Optional, Tuple, List
from collections import deque

# Check for NumPy (Required for VBOs)
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Error: NumPy is required for VBO acceleration. Please install it.")

from .core import Mesh, PaintTool, DEFAULT_COLORS, MAX_UNDO
from .math_utils import cross, normalize, bary_to_world
from .brush import SphereBrush, RayCaster, PaintOperation

# Qt imports
try:
    from PySide6.QtWidgets import QWidget
    from PySide6.QtCore import Qt, Signal
    from PySide6.QtGui import QMouseEvent, QWheelEvent
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
except ImportError:
    from PyQt5.QtWidgets import QWidget
    from PyQt5.QtCore import Qt, pyqtSignal as Signal
    from PyQt5.QtGui import QMouseEvent, QWheelEvent
    from PyQt5.QtOpenGL import QGLWidget as QOpenGLWidget

# OpenGL
from OpenGL import GL
from OpenGL import GLU


class RenderCache:
    """
    Handles flattening the mesh hierarchy into GPU buffers (VBOs).
    """
    def __init__(self):
        self.vbo_id = None
        self.vertex_count = 0
        self.is_dirty = True
        
        # We pack data as: [x, y, z, nx, ny, nz, r, g, b, u, v]
        # Stride = 11 floats * 4 bytes = 44 bytes
        self.STRIDE = 11 * 4 

    def update(self, mesh: Mesh, palette: List[Tuple[int, int, int]]):
        """Bake the complex mesh/sub-triangle structure into a flat float array"""
        if not HAS_NUMPY or not mesh:
            return

        data_list = []
        
        # Pre-calculate palette colors as 0.0-1.0 floats
        float_palette = [
            (c[0]/255.0, c[1]/255.0, c[2]/255.0) for c in palette
        ]

        for tri in mesh.triangles:
            v0 = mesh.vertices[tri.v_idx[0]]
            v1 = mesh.vertices[tri.v_idx[1]]
            v2 = mesh.vertices[tri.v_idx[2]]
            
            # Normal calculation
            if tri.normal is not None:
                nx, ny, nz = tri.normal
            else:
                e1 = (v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2])
                e2 = (v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2])
                cx, cy, cz = e1[1]*e2[2] - e1[2]*e2[1], e1[2]*e2[0] - e1[0]*e2[2], e1[0]*e2[1] - e1[1]*e2[0]
                l = math.sqrt(cx*cx + cy*cy + cz*cz)
                if l > 0:
                    nx, ny, nz = cx/l, cy/l, cz/l
                else:
                    nx, ny, nz = 0, 1, 0

            # UVs - pass through as-is (texture is flipped on upload)
            if tri.uv:
                t_uv = tri.uv
            else:
                t_uv = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))

            # Iterate sub-triangles (Leaf nodes)
            for sub in tri.paint_data:
                # Color
                color_idx = sub.extruder_id % len(float_palette)
                r, g, b = float_palette[color_idx]
                
                # Dim if masked - add slight red tint for visibility
                if sub.masked or tri.masked:
                    r, g, b = r*0.4 + 0.15, g*0.3, b*0.3

                # Process 3 vertices of the sub-triangle
                for i in range(3):
                    bary = sub.bary_corners[i]
                    
                    # Position (Interpolate World)
                    px = bary[0]*v0[0] + bary[1]*v1[0] + bary[2]*v2[0]
                    py = bary[0]*v0[1] + bary[1]*v1[1] + bary[2]*v2[1]
                    pz = bary[0]*v0[2] + bary[1]*v1[2] + bary[2]*v2[2]
                    
                    # UV (Interpolate Texture Coords - already flipped)
                    u = bary[0]*t_uv[0][0] + bary[1]*t_uv[1][0] + bary[2]*t_uv[2][0]
                    v = bary[0]*t_uv[0][1] + bary[1]*t_uv[1][1] + bary[2]*t_uv[2][1]
                    
                    # Add to list: Vert(3) + Norm(3) + Color(3) + UV(2)
                    data_list.extend([px, py, pz, nx, ny, nz, r, g, b, u, v])

        # Convert to Numpy Array (float32)
        vertex_data = np.array(data_list, dtype=np.float32)
        self.vertex_count = len(data_list) // 11

        # Upload to GPU
        if self.vbo_id is None:
            self.vbo_id = GL.glGenBuffers(1)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo_id)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        
        self.is_dirty = False


class Viewport3D(QOpenGLWidget):
    paint_performed = Signal()
    color_picked = Signal(int)
    status_update = Signal(str)
    brush_size_sync = Signal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.mesh = None
        self.brush = SphereBrush()
        self.ray_caster = RayCaster()
        self.render_cache = RenderCache()
        
        self.undo_stack = deque(maxlen=MAX_UNDO)
        self.redo_stack = deque(maxlen=MAX_UNDO)
        
        # Camera - 3ds Max style
        self.rot_x = 30.0   # Pitch (degrees)
        self.rot_y = 45.0   # Yaw (degrees)
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.view_distance = 5.0
        self.cam_target = (0.0, 0.0, 0.0)
        
        # Tools
        self.current_tool = PaintTool.PAINT
        self.current_color = 1
        self.is_painting = False
        
        # Cursor
        self.cursor_world_pos = None
        self.cursor_normal = None
        self.hover_tri_idx = -1
        
        # Display
        self.show_wireframe = False  # Default OFF
        self.show_ground = True
        self.show_texture = False
        self.show_brush = True
        self.texture_id = None
        self.texture_manager = None
        self.palette = list(DEFAULT_COLORS)
        
        # Mouse state
        self.last_mouse_pos = None
        self.last_paint_pos = None  # For stroke interpolation
        self.middle_mouse_down = False
        self._stroke_affected = None  # Track affected triangles for undo
        self._stroke_initial_state = None
        
        # CRITICAL: Enable Mouse Tracking and Focus
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    # =========================================================================
    # Public API
    # =========================================================================
    
    def set_mesh(self, mesh: Mesh):
        self.mesh = mesh
        if mesh:
            self.view_distance = mesh.size * 2.5
            self.zoom = 1.0
            self.pan_x = 0.0
            self.pan_y = 0.0
            self.cam_target = mesh.center
            self.render_cache.is_dirty = True
        self.update()

    def set_palette(self, palette):
        self.palette = palette
        self.render_cache.is_dirty = True
        self.update()

    def set_texture_manager(self, tm):
        self.texture_manager = tm
        self.update_texture()

    def update_texture(self):
        self.makeCurrent()
        self._upload_texture()
        self.doneCurrent()
        self.update()
        
    def frame_mesh(self):
        if self.mesh:
            self.rot_x = 30.0
            self.rot_y = 45.0
            self.zoom = 1.0
            self.pan_x = 0.0
            self.pan_y = 0.0
            self.view_distance = self.mesh.size * 2.5
            self.cam_target = self.mesh.center
        self.update()

    def undo(self):
        if not self.undo_stack or not self.mesh: 
            return
        affected = list(self.undo_stack[-1].triangle_states.keys())
        self.redo_stack.append(PaintOperation(self.mesh, affected))
        self.undo_stack.pop().restore(self.mesh)
        self.render_cache.is_dirty = True
        self.update()

    def redo(self):
        if not self.redo_stack or not self.mesh: 
            return
        affected = list(self.redo_stack[-1].triangle_states.keys())
        self.undo_stack.append(PaintOperation(self.mesh, affected))
        self.redo_stack.pop().restore(self.mesh)
        self.render_cache.is_dirty = True
        self.update()

    def clear_all_masks(self):
        """Clear all masks from the mesh"""
        if not self.mesh:
            return
        for tri in self.mesh.triangles:
            tri.masked = False
            for sub in tri.paint_data:
                sub.masked = False
        self.render_cache.is_dirty = True
        self.update()
        self.status_update.emit("All masks cleared")

    def force_refresh(self):
        """Force a full refresh of the viewport"""
        self.render_cache.is_dirty = True
        self.update()

    # =========================================================================
    # OpenGL Core
    # =========================================================================
    
    def initializeGL(self):
        GL.glClearColor(0.18, 0.18, 0.22, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_LIGHT0)
        GL.glEnable(GL.GL_COLOR_MATERIAL)
        GL.glColorMaterial(GL.GL_FRONT_AND_BACK, GL.GL_AMBIENT_AND_DIFFUSE)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, [1, 2, 1, 0])
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT, [0.3, 0.3, 0.3, 1])
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, [0.7, 0.7, 0.7, 1])
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
        GL.glPolygonOffset(1, 1)

    def resizeGL(self, w, h):
        # Account for High DPI display in projection
        ratio = self.devicePixelRatio()
        GL.glViewport(0, 0, int(w * ratio), int(h * ratio))
        
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GLU.gluPerspective(45, w/h if h else 1, 0.01, 1000)
        GL.glMatrixMode(GL.GL_MODELVIEW)

    def _upload_texture(self):
        """Upload texture to GPU with proper V-flip for OpenGL"""
        if not self.texture_manager: 
            return
        img_data = self.texture_manager.get_image_as_bytes('working')
        if not img_data: 
            return
        data, w, h = img_data
        
        # Flip texture vertically for OpenGL (OpenGL has bottom-left origin)
        if HAS_NUMPY:
            arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
            arr = np.flipud(arr).copy()
            data = arr.tobytes()
        
        if self.texture_id is None: 
            self.texture_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, w, h, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, data)

    def paintGL(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glLoadIdentity()
        
        # Camera transform (3ds Max style):
        # 1. Move back by view distance
        # 2. Apply pan offset  
        # 3. Rotate around X (pitch)
        # 4. Rotate around Y (yaw)
        # 5. Translate to look at target
        
        dist = self.view_distance / self.zoom
        GL.glTranslatef(0, 0, -dist)
        GL.glTranslatef(self.pan_x, self.pan_y, 0)
        GL.glRotatef(self.rot_x, 1, 0, 0)
        GL.glRotatef(self.rot_y, 0, 1, 0)
        GL.glTranslatef(-self.cam_target[0], -self.cam_target[1], -self.cam_target[2])
        
        if self.show_ground: 
            self._draw_ground()
        if self.mesh: 
            self._draw_mesh_fast()
        if self.show_brush and self.cursor_world_pos is not None: 
            self._draw_brush()

    def _draw_mesh_fast(self):
        if self.render_cache.is_dirty: 
            self.render_cache.update(self.mesh, self.palette)
        
        GL.glEnable(GL.GL_LIGHTING)
        has_texture = self.show_texture and self.texture_id is not None
        
        if has_texture:
            GL.glEnable(GL.GL_TEXTURE_2D)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
            GL.glColor3f(1, 1, 1)
        else:
            GL.glDisable(GL.GL_TEXTURE_2D)
            
        if self.render_cache.vbo_id:
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.render_cache.vbo_id)
            GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
            GL.glEnableClientState(GL.GL_NORMAL_ARRAY)
            
            GL.glVertexPointer(3, GL.GL_FLOAT, self.render_cache.STRIDE, ctypes.c_void_p(0))
            GL.glNormalPointer(GL.GL_FLOAT, self.render_cache.STRIDE, ctypes.c_void_p(12))
            
            if has_texture:
                GL.glEnableClientState(GL.GL_TEXTURE_COORD_ARRAY)
                GL.glDisableClientState(GL.GL_COLOR_ARRAY)
                GL.glTexCoordPointer(2, GL.GL_FLOAT, self.render_cache.STRIDE, ctypes.c_void_p(36))
            else:
                GL.glEnableClientState(GL.GL_COLOR_ARRAY)
                GL.glDisableClientState(GL.GL_TEXTURE_COORD_ARRAY)
                GL.glColorPointer(3, GL.GL_FLOAT, self.render_cache.STRIDE, ctypes.c_void_p(24))
            
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.render_cache.vertex_count)
            
            GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
            GL.glDisableClientState(GL.GL_NORMAL_ARRAY)
            GL.glDisableClientState(GL.GL_COLOR_ARRAY)
            GL.glDisableClientState(GL.GL_TEXTURE_COORD_ARRAY)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

        # Wireframe overlay
        if self.show_wireframe and self.render_cache.vbo_id:
            GL.glDisable(GL.GL_LIGHTING)
            GL.glDisable(GL.GL_TEXTURE_2D)
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
            GL.glColor4f(0, 0, 0, 0.25)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.render_cache.vbo_id)
            GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
            GL.glVertexPointer(3, GL.GL_FLOAT, self.render_cache.STRIDE, ctypes.c_void_p(0))
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.render_cache.vertex_count)
            GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

    def _draw_brush(self):
        """Draw brush as transparent dark grey sphere"""
        if self.cursor_world_pos is None: 
            return
        
        GL.glDisable(GL.GL_LIGHTING)
        GL.glDisable(GL.GL_TEXTURE_2D)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        
        pos = self.cursor_world_pos
        r = self.brush.radius
        
        # Choose color based on tool
        if self.current_tool == PaintTool.ERASE:
            GL.glColor4f(0.8, 0.2, 0.2, 0.25)  # Red tint for erase
        elif self.current_tool == PaintTool.MASK:
            GL.glColor4f(0.2, 0.2, 0.8, 0.25)  # Blue tint for mask
        elif self.current_tool == PaintTool.UNMASK:
            GL.glColor4f(0.2, 0.8, 0.2, 0.25)  # Green tint for unmask
        else:
            # Paint tool - dark grey with slight color tint
            c = self.palette[self.current_color % len(self.palette)]
            GL.glColor4f(c[0]/255 * 0.3 + 0.2, c[1]/255 * 0.3 + 0.2, c[2]/255 * 0.3 + 0.2, 0.35)
        
        # Draw filled sphere
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(GL.GL_FALSE)  # Don't write to depth buffer
        
        slices = 16
        stacks = 12
        
        for i in range(stacks):
            lat0 = math.pi * (-0.5 + float(i) / stacks)
            z0 = math.sin(lat0)
            zr0 = math.cos(lat0)
            
            lat1 = math.pi * (-0.5 + float(i + 1) / stacks)
            z1 = math.sin(lat1)
            zr1 = math.cos(lat1)
            
            GL.glBegin(GL.GL_QUAD_STRIP)
            for j in range(slices + 1):
                lng = 2 * math.pi * float(j) / slices
                x = math.cos(lng)
                y = math.sin(lng)
                
                GL.glVertex3f(pos[0] + r * x * zr0, pos[1] + r * z0, pos[2] + r * y * zr0)
                GL.glVertex3f(pos[0] + r * x * zr1, pos[1] + r * z1, pos[2] + r * y * zr1)
            GL.glEnd()
        
        GL.glDepthMask(GL.GL_TRUE)

    def _draw_ground(self):
        GL.glDisable(GL.GL_LIGHTING)
        GL.glDisable(GL.GL_TEXTURE_2D)
        size = self.view_distance * 1.5
        GL.glBegin(GL.GL_LINES)
        GL.glColor4f(0.35, 0.35, 0.35, 0.5)
        for i in range(-10, 11):
            s = i * size / 10
            GL.glVertex3f(s, 0, -size)
            GL.glVertex3f(s, 0, size)
            GL.glVertex3f(-size, 0, s)
            GL.glVertex3f(size, 0, s)
        GL.glEnd()
        
        # Axis indicators
        GL.glLineWidth(2)
        GL.glBegin(GL.GL_LINES)
        # X - Red
        GL.glColor3f(0.8, 0.2, 0.2)
        GL.glVertex3f(0, 0.01, 0)
        GL.glVertex3f(size * 0.2, 0.01, 0)
        # Z - Blue  
        GL.glColor3f(0.2, 0.2, 0.8)
        GL.glVertex3f(0, 0.01, 0)
        GL.glVertex3f(0, 0.01, size * 0.2)
        GL.glEnd()
        GL.glLineWidth(1)
        
        GL.glEnable(GL.GL_LIGHTING)

    # =========================================================================
    # Raycasting (FIXED FOR HIGH DPI)
    # =========================================================================
    
    def _screen_to_ray(self, x: int, y: int):
        """Convert screen coordinates to world-space ray using OpenGL matrices"""
        try:
            self.makeCurrent()
            
            # 1. Get Viewport (returns Physical pixels: x, y, w, h)
            viewport = GL.glGetIntegerv(GL.GL_VIEWPORT)
            modelview = GL.glGetDoublev(GL.GL_MODELVIEW_MATRIX)
            projection = GL.glGetDoublev(GL.GL_PROJECTION_MATRIX)
            
            if viewport is None or modelview is None or projection is None:
                return self._screen_to_ray_fallback(x, y)
            
            # 2. Get DPI Ratio
            ratio = self.devicePixelRatio()
            
            # 3. Convert Mouse (Logical) to OpenGL (Physical) coordinates
            # Mouse Y is top-down, OpenGL Y is bottom-up
            phys_x = x * ratio
            phys_y = (viewport[3]) - (y * ratio) # Use physical height
            
            near_point = GLU.gluUnProject(phys_x, phys_y, 0.0, modelview, projection, viewport)
            far_point = GLU.gluUnProject(phys_x, phys_y, 1.0, modelview, projection, viewport)
            
            direction = (
                far_point[0] - near_point[0],
                far_point[1] - near_point[1],
                far_point[2] - near_point[2]
            )
            
            return near_point, direction
            
        except Exception as e:
            print(f"Raycast Error: {e}")
            return self._screen_to_ray_fallback(x, y)
    
    def _screen_to_ray_fallback(self, x: int, y: int):
        """Fallback raycasting when GL matrices unavailable"""
        w = self.width()
        h = self.height()
        
        ndc_x = (2.0 * x) / w - 1.0
        ndc_y = 1.0 - (2.0 * y) / h
        
        aspect = w / h if h else 1.0
        fov_rad = math.radians(45.0)
        tan_half_fov = math.tan(fov_rad / 2.0)
        
        ray_x = ndc_x * aspect * tan_half_fov
        ray_y = ndc_y * tan_half_fov
        ray_z = -1.0
        
        rx = math.radians(-self.rot_x)
        ry = math.radians(-self.rot_y)
        
        x1 = ray_x
        y1 = ray_y * math.cos(rx) - ray_z * math.sin(rx)
        z1 = ray_y * math.sin(rx) + ray_z * math.cos(rx)
        
        x2 = x1 * math.cos(ry) + z1 * math.sin(ry)
        y2 = y1
        z2 = -x1 * math.sin(ry) + z1 * math.cos(ry)
        
        direction = (x2, y2, z2)
        
        dist = self.view_distance / self.zoom
        
        cx, cy, cz = 0, 0, dist
        cy1 = cy * math.cos(-rx) - cz * math.sin(-rx)
        cz1 = cy * math.sin(-rx) + cz * math.cos(-rx)
        cx2 = cx * math.cos(-ry) + cz1 * math.sin(-ry)
        cy2 = cy1
        cz2 = -cx * math.sin(-ry) + cz1 * math.cos(-ry)
        
        origin = (
            cx2 + self.cam_target[0] + self.pan_x, 
            cy2 + self.cam_target[1] + self.pan_y, 
            cz2 + self.cam_target[2]
        )
        return origin, direction

    def _update_cursor(self, x, y):
        if not self.mesh: 
            self.cursor_world_pos = None
            return
        
        origin, direction = self._screen_to_ray(x, y)
        hit = self.ray_caster.cast(origin, direction, self.mesh)
        
        if hit:
            tri_idx, bary, dist = hit
            tri = self.mesh.triangles[tri_idx]
            v = [self.mesh.vertices[i] for i in tri.v_idx]
            self.cursor_world_pos = bary_to_world(bary, v[0], v[1], v[2])
            self.cursor_normal = tri.normal
            self.hover_tri_idx = tri_idx
            self.brush.set_position(self.cursor_world_pos, self.cursor_normal, tri_idx)
        else:
            self.cursor_world_pos = None
            self.hover_tri_idx = -1

    def _do_paint(self):
        """Perform paint operation - creates undo state for stroke start"""
        if not self.mesh or self.cursor_world_pos is None: 
            return
        
        # For mask/unmask tools, we need to find triangles regardless of mask state
        ignore_mask = self.current_tool in (PaintTool.MASK, PaintTool.UNMASK)
        affected = self.brush.find_affected_triangles(self.mesh, ignore_mask=ignore_mask)
        if not affected: 
            return
        
        # Create undo state at stroke start
        # For continuous strokes, we accumulate affected triangles
        if not hasattr(self, '_stroke_affected') or self._stroke_affected is None:
            self._stroke_affected = set()
            # Save initial state of all triangles we might affect
            self._stroke_initial_state = {}
            self._stroke_initial_masks = {}
        
        # Track newly affected triangles
        for idx in affected:
            if idx not in self._stroke_affected:
                self._stroke_affected.add(idx)
                # Save initial state including mask
                self._stroke_initial_state[idx] = self.mesh.triangles[idx].copy_paint_data()
                self._stroke_initial_masks[idx] = self.mesh.triangles[idx].masked
        
        for idx in affected:
            if self.current_tool == PaintTool.PAINT:
                self.brush.paint_triangle(self.mesh, idx, self.current_color)
            elif self.current_tool == PaintTool.ERASE:
                self.brush.erase_triangle(self.mesh, idx)
            elif self.current_tool == PaintTool.MASK:
                self.brush.mask_triangle(self.mesh, idx)
            elif self.current_tool == PaintTool.UNMASK:
                self.brush.unmask_triangle(self.mesh, idx)
            elif self.current_tool == PaintTool.EYEDROPPER:
                if self.mesh.triangles[idx].paint_data:
                    self.current_color = self.mesh.triangles[idx].paint_data[0].extruder_id
                    self.color_picked.emit(self.current_color)
                break

        self.render_cache.is_dirty = True
        self.paint_performed.emit()
    
    def _finalize_stroke(self):
        """Called when stroke ends - saves undo state"""
        if hasattr(self, '_stroke_initial_state') and self._stroke_initial_state:
            # Create undo operation from initial states
            op = PaintOperation.__new__(PaintOperation)
            op.triangle_states = {idx: data for idx, data in self._stroke_initial_state.items()}
            op.triangle_masks = getattr(self, '_stroke_initial_masks', {})
            self.undo_stack.append(op)
            self.redo_stack.clear()
        
        self._stroke_affected = None
        self._stroke_initial_state = None
        self._stroke_initial_masks = None

    # =========================================================================
    # Mouse Events - 3ds Max Style Navigation
    # =========================================================================
    
    def mousePressEvent(self, e):
        # FIX: Ensure focus is always set on click, enabling key events
        self.setFocus()
        
        
        pos = e.position() if hasattr(e, 'position') else e.pos()
        x, y = int(pos.x()), int(pos.y())
        self.last_mouse_pos = (x, y)
        self.last_paint_pos = None  # Reset paint interpolation
        
        if e.button() == Qt.LeftButton:
            self.is_painting = True
            self._update_cursor(x, y)
            self._do_paint()
            # Store position for interpolation
            if self.cursor_world_pos:
                self.last_paint_pos = self.cursor_world_pos
        elif e.button() == Qt.MiddleButton:
            self.middle_mouse_down = True
        
        self.update()

    def mouseMoveEvent(self, e):
        pos = e.position() if hasattr(e, 'position') else e.pos()
        x, y = int(pos.x()), int(pos.y())
        
        # Update cursor for painting
        self._update_cursor(x, y)
        
        if self.last_mouse_pos:
            dx = x - self.last_mouse_pos[0]
            dy = y - self.last_mouse_pos[1]
            
            if e.buttons() & Qt.LeftButton:
                if self.is_painting and self.cursor_world_pos is not None:
                    # Interpolate paint strokes for smooth lines
                    self._do_paint_interpolated()
                    
            elif e.buttons() & Qt.MiddleButton:
                alt_down = bool(e.modifiers() & Qt.AltModifier)
                
                if alt_down:
                    # Alt+MMB = Orbit
                    self.rot_y += dx * 0.4
                    self.rot_x += dy * 0.4
                else:
                    # MMB = Pan
                    pan_scale = self.view_distance / self.zoom * 0.002
                    self.pan_x += dx * pan_scale
                    self.pan_y -= dy * pan_scale
                    
            elif e.buttons() & Qt.RightButton:
                # RMB = Orbit
                self.rot_y += dx * 0.4
                self.rot_x += dy * 0.4
        
        self.last_mouse_pos = (x, y)
        self.update()

    def mouseReleaseEvent(self, e):
        
        if e.button() == Qt.LeftButton:
            self.is_painting = False
            self.last_paint_pos = None
            self._finalize_stroke()  # Save undo state for completed stroke
        elif e.button() == Qt.MiddleButton:
            self.middle_mouse_down = False
        self.last_mouse_pos = None
    
    def _do_paint_interpolated(self):
        """Paint with interpolation between last and current position for smooth strokes"""
        if not self.mesh or self.cursor_world_pos is None:
            return
        
        current_pos = self.cursor_world_pos
        
        # If we have a previous position, interpolate
        if self.last_paint_pos is not None:
            # Calculate distance
            dx = current_pos[0] - self.last_paint_pos[0]
            dy = current_pos[1] - self.last_paint_pos[1]
            dz = current_pos[2] - self.last_paint_pos[2]
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            # Step size - smaller = smoother strokes
            step_size = self.brush.radius * 0.15
            
            if dist > step_size:
                # Interpolate positions
                steps = max(1, int(dist / step_size))
                for i in range(1, steps + 1):
                    t = i / (steps + 1)
                    interp_pos = (
                        self.last_paint_pos[0] + dx * t,
                        self.last_paint_pos[1] + dy * t,
                        self.last_paint_pos[2] + dz * t
                    )
                    # Temporarily set brush position and paint
                    self.brush.set_position(interp_pos, self.cursor_normal, self.hover_tri_idx)
                    self._do_paint_at_brush()
        
        # Paint at current position
        self.brush.set_position(current_pos, self.cursor_normal, self.hover_tri_idx)
        self._do_paint_at_brush()
        
        # Update last paint position
        self.last_paint_pos = current_pos

    def _do_paint_at_brush(self):
        """Paint at current brush position with proper stroke tracking"""
        if not self.mesh or self.brush.position is None:
            return
        
        # For mask/unmask tools, we need to find triangles regardless of mask state
        ignore_mask = self.current_tool in (PaintTool.MASK, PaintTool.UNMASK)
        affected = self.brush.find_affected_triangles(self.mesh, ignore_mask=ignore_mask)
        if not affected:
            return
        
        # Track for undo (same as _do_paint)
        if not hasattr(self, '_stroke_affected') or self._stroke_affected is None:
            self._stroke_affected = set()
            self._stroke_initial_state = {}
            self._stroke_initial_masks = {}
        
        # Track newly affected triangles
        for idx in affected:
            if idx not in self._stroke_affected:
                self._stroke_affected.add(idx)
                self._stroke_initial_state[idx] = self.mesh.triangles[idx].copy_paint_data()
                self._stroke_initial_masks[idx] = self.mesh.triangles[idx].masked
        
        for idx in affected:
            if self.current_tool == PaintTool.PAINT:
                self.brush.paint_triangle(self.mesh, idx, self.current_color)
            elif self.current_tool == PaintTool.ERASE:
                self.brush.erase_triangle(self.mesh, idx)
            elif self.current_tool == PaintTool.MASK:
                self.brush.mask_triangle(self.mesh, idx)
            elif self.current_tool == PaintTool.UNMASK:
                self.brush.unmask_triangle(self.mesh, idx)
        
        self.render_cache.is_dirty = True
    
    def wheelEvent(self, e):
        # FIX: Check both X and Y axes. 
        # Holding Alt often shifts scroll to the X axis (Horizontal) in Qt.
        angle_delta = e.angleDelta()
        delta_y = angle_delta.y()
        delta_x = angle_delta.x()
        
        # Use Y if available, otherwise fallback to X
        delta = delta_y if delta_y != 0 else delta_x
        
        # Debug Print
        # print(f"DEBUG: Wheel dy={delta_y} dx={delta_x} effective={delta} mods={e.modifiers()}")
        
        # Skip only if BOTH are 0
        if delta == 0:
            return
        
        # Alt+Scroll = Brush resize
        if e.modifiers() & Qt.AltModifier:
            # Use continuous scaling based on delta magnitude
            # PATCH: Increased divisor to 1200.0 for finer, high-fidelity control
            scale_amount = 1.0 + (delta / 1200.0) 
            scale_amount = max(0.5, min(2.0, scale_amount))  # Clamp factor
            
            new_radius = self.brush.radius * scale_amount
            self.brush.radius = max(0.01, min(100.0, new_radius))
            
            # PATCH: Broadcast the change to the UI
            self.brush_size_sync.emit(self.brush.radius)
            self.status_update.emit(f"Brush: {self.brush.radius:.3f}")
            
            # Accept the event so it doesn't bubble up
            e.accept()
        else:
            # Normal scroll = Zoom
            factor = 1.1 if delta > 0 else 0.9
            self.zoom = max(0.1, min(50.0, self.zoom * factor))
            e.accept()
        
        self.update()

    # =========================================================================
    # Keyboard Events - Focus is critical!
    # =========================================================================
    
    def keyPressEvent(self, e):
        # Debug print for keys
        
        key = e.key()
        handled = True
        
        # Tool hotkeys
        if key == Qt.Key_P:
            self.current_tool = PaintTool.PAINT
            self.status_update.emit("Tool: Paint")
        elif key == Qt.Key_E:
            self.current_tool = PaintTool.ERASE
            self.status_update.emit("Tool: Erase")
        elif key == Qt.Key_M:
            self.current_tool = PaintTool.MASK
            self.status_update.emit("Tool: Mask")
        elif key == Qt.Key_U:
            self.current_tool = PaintTool.UNMASK
            self.status_update.emit("Tool: Unmask")
        elif key == Qt.Key_I:
            self.current_tool = PaintTool.EYEDROPPER
            self.status_update.emit("Tool: Eyedropper")
            
        # Brush size
        elif key == Qt.Key_BracketLeft:
            self.brush.radius = max(0.01, self.brush.radius * 0.8)
            self.status_update.emit(f"Brush: {self.brush.radius:.2f}")
        elif key == Qt.Key_BracketRight:
            self.brush.radius = min(100.0, self.brush.radius * 1.25)
            self.status_update.emit(f"Brush: {self.brush.radius:.2f}")
            
        # Display toggles
        elif key == Qt.Key_W:
            self.show_wireframe = not self.show_wireframe
            self.status_update.emit(f"Wireframe: {'On' if self.show_wireframe else 'Off'}")
        elif key == Qt.Key_G:
            self.show_ground = not self.show_ground
            self.status_update.emit(f"Ground: {'On' if self.show_ground else 'Off'}")
        elif key == Qt.Key_T:
            self.show_texture = not self.show_texture
            self.status_update.emit(f"Texture: {'On' if self.show_texture else 'Off'}")
        elif key == Qt.Key_F:
            self.frame_mesh()
            self.status_update.emit("View reset")
        elif key == Qt.Key_C:
            self.clear_all_masks()
            
        # Color selection (1-9)
        elif Qt.Key_1 <= key <= Qt.Key_9:
            self.current_color = key - Qt.Key_0
            self.status_update.emit(f"Color: {self.current_color}")
            self.color_picked.emit(self.current_color)
        else:
            handled = False
        
        self.update()
        
        if not handled:
            super().keyPressEvent(e)