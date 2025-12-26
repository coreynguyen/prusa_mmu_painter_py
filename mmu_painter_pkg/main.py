"""
MMU Painter - Main Application Entry Point
"""

import sys
import os

try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QSplitter, QFileDialog, QMessageBox, QStatusBar, QProgressDialog
    )
    from PySide6.QtCore import Qt, QTimer, Signal, QObject
    from PySide6.QtGui import QColor, QKeySequence, QShortcut, QIcon
except ImportError:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QSplitter, QFileDialog, QMessageBox, QStatusBar, QShortcut, QProgressDialog
    )
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal as Signal, QObject
    from PyQt5.QtGui import QColor, QKeySequence, QIcon

import threading

from .core import PaintTool, DEFAULT_COLORS
from .io import load_model
from .viewport import Viewport3D
from .ui import TexturePanel, ToolPanel, DisplayPanel

# Try to import fast loader
try:
    from .io_fast import load_model_async, load_model_fast, LoadProgress
    HAS_FAST_LOADER = True
except ImportError:
    HAS_FAST_LOADER = False


class LoaderSignals(QObject):
    """Thread-safe signals for async loading"""
    progress = Signal(str, float)  # stage, 0.0-1.0
    complete = Signal(object)      # mesh
    error = Signal(str)            # error message


def get_icon_path():
    """Find the icon file in common locations"""
    # Check various locations
    locations = [
        os.path.join(os.path.dirname(__file__), 'icon.png'),
        os.path.join(os.path.dirname(__file__), 'icon.ico'),
        os.path.join(os.path.dirname(__file__), '..', 'icon.png'),
        os.path.join(os.path.dirname(__file__), '..', 'icon.ico'),
        os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'icon.png'),
        os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'icon.ico'),
    ]
    for path in locations:
        if os.path.exists(path):
            return path
    return None


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("MMU Painter v4.2")
        self.setMinimumSize(1400, 900)
        
        # Set application icon
        icon_path = get_icon_path()
        if icon_path:
            self.setWindowIcon(QIcon(icon_path))
        
        self.mesh = None
        
        self._setup_ui()
        self._connect_signals()
        self._setup_shortcuts()
    
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Tools
        left_widget = QWidget()
        left_widget.setMaximumWidth(260)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # File buttons at top
        from PySide6.QtWidgets import QGroupBox, QPushButton
        file_group = QGroupBox("File")
        file_layout = QVBoxLayout(file_group)
        
        open_btn = QPushButton("Open Model (Ctrl+O)")
        open_btn.clicked.connect(self._open_model)
        file_layout.addWidget(open_btn)
        
        save_btn = QPushButton("Save 3MF (Ctrl+S)")
        save_btn.clicked.connect(self._save_model)
        file_layout.addWidget(save_btn)
        
        left_layout.addWidget(file_group)
        
        # Tool panel
        self.tool_panel = ToolPanel()
        left_layout.addWidget(self.tool_panel)
        
        # Display panel
        self.display_panel = DisplayPanel()
        left_layout.addWidget(self.display_panel)
        
        splitter.addWidget(left_widget)
        
        # Center - 3D Viewport
        self.viewport = Viewport3D()
        splitter.addWidget(self.viewport)
        
        # Right panel - Texture
        self.texture_panel = TexturePanel()
        self.texture_panel.setMaximumWidth(280)
        splitter.addWidget(self.texture_panel)
        
        splitter.setSizes([240, 880, 280])
        
        # Status bar
        self.statusBar().showMessage("Ready - Open a model (Ctrl+O)")
    
    def _connect_signals(self):
        # Tool panel -> Viewport
        self.tool_panel.tool_changed.connect(self._on_tool_changed)
        self.tool_panel.color_changed.connect(self._on_color_changed)
        self.tool_panel.brush_size_changed.connect(self._on_brush_size_changed)
        self.tool_panel.subdivision_changed.connect(self._on_subdivision_changed)
        self.tool_panel.auto_subdivide_changed.connect(self._on_auto_subdiv_changed)
        self.tool_panel.clear_masks_requested.connect(self.viewport.clear_all_masks)
        
        # Display panel -> Viewport
        self.display_panel.wireframe_toggled.connect(self._on_wireframe_toggled)
        self.display_panel.ground_toggled.connect(self._on_ground_toggled)
        self.display_panel.texture_toggled.connect(self._on_texture_toggled)
        self.display_panel.frame_requested.connect(self.viewport.frame_mesh)
        
        # Viewport -> UI
        self.viewport.color_picked.connect(self._on_color_picked)
        self.viewport.status_update.connect(self._show_status)
        
        # Texture panel -> Viewport
        self.texture_panel.palette_updated.connect(self._on_palette_updated)
        self.texture_panel.project_requested.connect(self._on_project_texture)
        
        # Connect texture loading events to GPU upload
        self.texture_panel.texture_loaded.connect(self.viewport.update_texture)
        self.texture_panel.texture_updated.connect(self.viewport.update_texture)
        
        # Link texture manager
        self.viewport.set_texture_manager(self.texture_panel.get_texture_manager())
    
    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+O"), self, self._open_model)
        QShortcut(QKeySequence("Ctrl+T"), self, lambda: self.texture_panel.load_texture())
        QShortcut(QKeySequence("Ctrl+S"), self, self._save_model)
        QShortcut(QKeySequence("Ctrl+Z"), self, self.viewport.undo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, self.viewport.redo)
        QShortcut(QKeySequence("Ctrl+Y"), self, self.viewport.redo)  # Alt redo
        QShortcut(QKeySequence("Ctrl+Q"), self, self.texture_panel.do_quantize)
        QShortcut(QKeySequence("Ctrl+P"), self, self._on_project_texture)
    
    def _show_status(self, msg: str):
        self.statusBar().showMessage(msg, 3000)
    
    def _open_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Model", "",
            "3D Models (*.obj *.3mf);;OBJ Files (*.obj);;3MF Files (*.3mf)"
        )
        
        if not path:
            return
        
        # Check file size for async loading decision
        file_size = os.path.getsize(path)
        use_async = file_size > 5 * 1024 * 1024  # 5MB threshold
        
        if use_async and HAS_FAST_LOADER:
            self._load_model_async(path)
        else:
            self._load_model_sync(path)
    
    def _load_model_sync(self, path: str):
        """Synchronous loading for small files"""
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.mesh = load_model(path)
            self._finish_loading(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load:\n{e}")
        finally:
            QApplication.restoreOverrideCursor()
    
    def _load_model_async(self, path: str):
        """Asynchronous loading with progress dialog for large files"""
        # Create progress dialog
        self._progress_dialog = QProgressDialog(
            "Loading model...", "Cancel", 0, 100, self
        )
        self._progress_dialog.setWindowTitle("Loading")
        self._progress_dialog.setWindowModality(Qt.WindowModal)
        self._progress_dialog.setMinimumDuration(0)
        self._progress_dialog.setValue(0)
        self._progress_dialog.show()
        
        # Create signal bridge for thread-safe updates
        self._loader_signals = LoaderSignals()
        self._loader_signals.progress.connect(self._on_load_progress)
        self._loader_signals.complete.connect(lambda m: self._on_load_complete(m, path))
        self._loader_signals.error.connect(self._on_load_error)
        
        # Store path for completion handler
        self._loading_path = path
        
        # Track cancellation
        self._load_cancelled = False
        self._progress_dialog.canceled.connect(self._on_load_cancelled)
        
        # Start background loading
        def load_thread():
            try:
                progress = LoadProgress(self._thread_progress_callback)
                mesh = load_model_fast(path, progress)
                if not self._load_cancelled and mesh:
                    self._loader_signals.complete.emit(mesh)
            except Exception as e:
                if not self._load_cancelled:
                    self._loader_signals.error.emit(str(e))
        
        self._load_thread = threading.Thread(target=load_thread, daemon=True)
        self._load_thread.start()
    
    def _thread_progress_callback(self, stage: str, progress: float):
        """Called from loading thread - emits signal to main thread"""
        if not self._load_cancelled:
            self._loader_signals.progress.emit(stage, progress)
    
    def _on_load_progress(self, stage: str, progress: float):
        """Handle progress update on main thread"""
        if hasattr(self, '_progress_dialog') and self._progress_dialog:
            self._progress_dialog.setLabelText(f"{stage}...")
            self._progress_dialog.setValue(int(progress * 100))
    
    def _on_load_complete(self, mesh, path: str):
        """Handle load completion on main thread"""
        # Do NOT close dialog yet. The VBO generation in _finish_loading
        # is synchronous and heavy. We keep the dialog open to show activity.
        if hasattr(self, '_progress_dialog') and self._progress_dialog:
            self._progress_dialog.setLabelText("Initializing 3D View (this may take a moment)...")
            self._progress_dialog.setValue(99)
            # Force UI update so text changes before the heavy freeze
            QApplication.processEvents()
        
        self.mesh = mesh
        self._finish_loading(path)
        
        # Now close it
        if hasattr(self, '_progress_dialog') and self._progress_dialog:
            self._progress_dialog.close()
            self._progress_dialog = None
    
    def _on_load_error(self, error_msg: str):
        """Handle load error on main thread"""
        if hasattr(self, '_progress_dialog') and self._progress_dialog:
            self._progress_dialog.close()
            self._progress_dialog = None
        
        QMessageBox.critical(self, "Error", f"Failed to load:\n{error_msg}")
    
    def _on_load_cancelled(self):
        """Handle user cancellation"""
        self._load_cancelled = True
        self._show_status("Loading cancelled")
    
    def _finish_loading(self, path: str):
        """Common loading completion logic"""
        self.viewport.set_mesh(self.mesh)
        
        # CRITICAL FIX: Force VBO generation now while we still have 
        # the progress dialog (or WaitCursor) active.
        # Otherwise, the UI will freeze immediately after the dialog closes.
        if self.mesh:
            self.viewport.render_cache.update(self.mesh, self.viewport.palette)
        
        # Extract UVs for texture panel overlay
        uvs = []
        if self.mesh:
            for tri in self.mesh.triangles:
                if tri.uv:
                    uvs.append(tri.uv)
        self.texture_panel.set_uv_data(uvs)
        
        # Load texture if found in MTL
        if self.mesh and self.mesh.texture_path:
            self.texture_panel.load_texture(self.mesh.texture_path)
        
        tri_count = len(self.mesh.triangles) if self.mesh else 0
        vert_count = len(self.mesh.vertices) if self.mesh else 0
        self._show_status(f"Loaded: {os.path.basename(path)} ({vert_count:,} verts, {tri_count:,} tris)")
    
    def _save_model(self):
        if not self.mesh:
            QMessageBox.warning(self, "Warning", "No model to save")
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Save 3MF", "", "3MF Files (*.3mf)"
        )
        
        if path:
            try:
                self._show_status("Save logic requires io.py implementation")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save:\n{e}")
    
    def _on_tool_changed(self, tool_idx: int):
        tools = [PaintTool.PAINT, PaintTool.ERASE, PaintTool.MASK, 
                 PaintTool.UNMASK, PaintTool.EYEDROPPER]
        if tool_idx < len(tools):
            self.viewport.current_tool = tools[tool_idx]
            self._show_status(f"Tool: {tools[tool_idx].name}")
    
    def _on_color_changed(self, color_idx: int):
        self.viewport.current_color = color_idx
        self._show_status(f"Color: {color_idx}")
    
    def _on_color_picked(self, color_idx: int):
        self.tool_panel.set_color(color_idx)
        self._show_status(f"Picked color: {color_idx}")
    
    def _on_brush_size_changed(self, size: float):
        self.viewport.brush.radius = size
    
    def _on_subdivision_changed(self, depth: int):
        self.viewport.brush.max_depth = depth
    
    def _on_auto_subdiv_changed(self, enabled: bool):
        self.viewport.brush.auto_subdivide = enabled
    
    def _on_wireframe_toggled(self, checked: bool):
        self.viewport.show_wireframe = checked
        self.viewport.update()
    
    def _on_ground_toggled(self, checked: bool):
        self.viewport.show_ground = checked
        self.viewport.update()

    def _on_texture_toggled(self, checked: bool):
        self.viewport.show_texture = checked
        self.viewport.update()
    
    def _on_palette_updated(self, palette: list):
        self.tool_panel.set_palette(palette)
        self.viewport.set_palette(palette)
    
    def _on_project_texture(self):
        """Project texture colors onto mesh"""
        if not self.mesh:
            QMessageBox.warning(self, "Warning", "No mesh loaded")
            return
        
        tex_mgr = self.texture_panel.get_texture_manager()
        if tex_mgr.labels is None:
            QMessageBox.warning(self, "Warning", "Quantize texture first")
            return
        
        # NEW: Progress dialog for projection
        progress = QProgressDialog("Projecting texture colors...", "Cancel", 0, 0, self)
        progress.setWindowTitle("Please Wait")
        progress.setWindowModality(Qt.WindowModal)
        progress.setCancelButton(None) # Disable cancel for sync operation
        progress.show()
        QApplication.processEvents() # Force it to appear
        
        # Get Max Depth from Brush Settings
        max_depth = self.viewport.brush.max_depth
        
        success = tex_mgr.project_texture_to_mesh(self.mesh, max_depth)
        
        progress.close()
        
        if success:
            # Force full viewport refresh
            self.viewport.force_refresh()
            self._show_status(f"Texture projected (Depth {max_depth})")
        else:
            QMessageBox.warning(self, "Warning", "Projection failed - check UVs")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set application icon (for taskbar)
    icon_path = get_icon_path()
    if icon_path:
        app.setWindowIcon(QIcon(icon_path))
    
    # On Windows, need to set AppUserModelID for proper taskbar icon
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('mmu_painter.app.1.0')
    except:
        pass  # Not Windows or failed
    
    # Dark theme
    palette = app.palette()
    palette.setColor(palette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(palette.ColorRole.WindowText, QColor(255, 255, 255))
    palette.setColor(palette.ColorRole.Base, QColor(35, 35, 35))
    palette.setColor(palette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(palette.ColorRole.ToolTipBase, QColor(25, 25, 25))
    palette.setColor(palette.ColorRole.ToolTipText, QColor(255, 255, 255))
    palette.setColor(palette.ColorRole.Text, QColor(255, 255, 255))
    palette.setColor(palette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(palette.ColorRole.ButtonText, QColor(255, 255, 255))
    palette.setColor(palette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(palette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(palette.ColorRole.HighlightedText, QColor(35, 35, 35))
    app.setPalette(palette)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()