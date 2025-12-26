"""
UI Panels for MMU Painter - Texture panel, tool panels, etc.
"""

from typing import List, Tuple, Optional

try:
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QSpinBox, QCheckBox, QComboBox, QGroupBox, QFileDialog,
        QSlider, QGridLayout
    )
    # FIXED: QPointF is in QtCore
    from PySide6.QtCore import Qt, Signal, QPointF
    # FIXED: Removed QPointF from QtGui
    from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
except ImportError:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QSpinBox, QCheckBox, QComboBox, QGroupBox, QFileDialog,
        QSlider, QGridLayout
    )
    # FIXED: QPointF is in QtCore
    from PyQt5.QtCore import Qt, pyqtSignal as Signal, QPointF
    # FIXED: Removed QPointF from QtGui
    from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor

from .texture import TextureManager
from .core import DEFAULT_COLORS


class TexturePanel(QWidget):
    """
    Panel for texture viewing, manipulation, and quantization.
    
    Hotkeys (when viewport focused):
        Ctrl+T: Load texture
        Ctrl+Q: Quantize
        Ctrl+P: Project to mesh
    
    Signals:
        palette_updated: Emitted when palette changes (list of RGB tuples)
        texture_loaded: Emitted when texture is loaded
        project_requested: Emitted when user clicks Project to Mesh
        texture_updated: Emitted when texture flips (requires re-upload to GPU)
    """
    
    palette_updated = Signal(list)
    texture_loaded = Signal()
    project_requested = Signal()
    texture_updated = Signal()
    quantize_requested = Signal()  # For hotkey
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.texture_mgr = TextureManager()
        self.show_uv_overlay = False
        self.uv_triangles: List[Tuple] = []
        
        self._setup_ui()
        self.setFocusPolicy(Qt.StrongFocus)  # Enable keyboard input
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Texture display area
        self.texture_label = QLabel("No texture loaded")
        self.texture_label.setAlignment(Qt.AlignCenter)
        self.texture_label.setMinimumHeight(180)
        self.texture_label.setStyleSheet(
            "background: #2a2a2a; border: 1px solid #444; border-radius: 3px;"
        )
        layout.addWidget(self.texture_label)
        
        # Load button
        load_btn = QPushButton("Load Texture (Ctrl+T)")
        load_btn.clicked.connect(self._load_texture_dialog)
        layout.addWidget(load_btn)
        
        # Options group
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout(options_group)
        
        flip_row = QHBoxLayout()
        
        self.flip_h_check = QCheckBox("Flip H")
        self.flip_h_check.toggled.connect(self._on_flip_h)
        flip_row.addWidget(self.flip_h_check)
        
        self.flip_v_check = QCheckBox("Flip V")
        self.flip_v_check.toggled.connect(self._on_flip_v)
        flip_row.addWidget(self.flip_v_check)
        
        self.uv_check = QCheckBox("UV Overlay")
        self.uv_check.toggled.connect(self._on_uv_toggle)
        flip_row.addWidget(self.uv_check)
        
        options_layout.addLayout(flip_row)
        layout.addWidget(options_group)
        
        # Quantization group
        quant_group = QGroupBox("Quantization")
        quant_layout = QVBoxLayout(quant_group)
        
        # Method selection
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Uniform", "K-Means", "Median Cut", "Octree"])
        self.method_combo.setCurrentIndex(0)  # Default: Uniform
        self.method_combo.setToolTip(
            "Uniform: Fastest, even color division\n"
            "K-Means: Best quality, slower\n"
            "Median Cut: Good for distinct regions\n"
            "Octree: Fast, preserves colors"
        )
        method_row.addWidget(self.method_combo)
        quant_layout.addLayout(method_row)
        
        # Colors count
        colors_row = QHBoxLayout()
        colors_row.addWidget(QLabel("Colors:"))
        
        self.num_colors_spin = QSpinBox()
        self.num_colors_spin.setRange(2, 32)
        self.num_colors_spin.setValue(5)
        colors_row.addWidget(self.num_colors_spin)
        
        self.quantize_btn = QPushButton("Quantize (Ctrl+Q)")
        self.quantize_btn.clicked.connect(self._do_quantize)
        colors_row.addWidget(self.quantize_btn)
        
        quant_layout.addLayout(colors_row)
        
        # Advanced options (collapsible)
        self.advanced_check = QCheckBox("Advanced Options")
        self.advanced_check.toggled.connect(self._toggle_advanced)
        quant_layout.addWidget(self.advanced_check)
        
        self.advanced_widget = QWidget()
        adv_layout = QVBoxLayout(self.advanced_widget)
        adv_layout.setContentsMargins(10, 0, 0, 0)
        
        # Pre-blur
        blur_row = QHBoxLayout()
        blur_row.addWidget(QLabel("Pre-Blur:"))
        self.blur_spin = QSpinBox()
        self.blur_spin.setRange(0, 10)
        self.blur_spin.setValue(0)
        self.blur_spin.setToolTip("Blur before quantizing (reduces noise)")
        blur_row.addWidget(self.blur_spin)
        adv_layout.addLayout(blur_row)
        
        # Saturation
        sat_row = QHBoxLayout()
        sat_row.addWidget(QLabel("Saturation:"))
        self.sat_slider = QSlider(Qt.Horizontal)
        self.sat_slider.setRange(50, 200)
        self.sat_slider.setValue(100)
        self.sat_slider.setToolTip("Color saturation boost (100 = normal)")
        sat_row.addWidget(self.sat_slider)
        self.sat_label = QLabel("1.0")
        self.sat_label.setMinimumWidth(30)
        sat_row.addWidget(self.sat_label)
        self.sat_slider.valueChanged.connect(lambda v: self.sat_label.setText(f"{v/100:.1f}"))
        adv_layout.addLayout(sat_row)
        
        # Dithering
        self.dither_check = QCheckBox("Dithering")
        self.dither_check.setToolTip("Floyd-Steinberg dithering (slower, smoother gradients)")
        adv_layout.addWidget(self.dither_check)
        
        # Min region size
        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("Min Pixels:"))
        self.thresh_spin = QSpinBox()
        self.thresh_spin.setRange(0, 1000)
        self.thresh_spin.setValue(0)
        self.thresh_spin.setToolTip("Ignore colors with fewer than N pixels")
        thresh_row.addWidget(self.thresh_spin)
        adv_layout.addLayout(thresh_row)
        
        self.advanced_widget.setVisible(False)
        quant_layout.addWidget(self.advanced_widget)
        
        # View mode - checkbox instead of dropdown
        view_row = QHBoxLayout()
        self.segmented_check = QCheckBox("Show Segmented (S)")
        self.segmented_check.setToolTip("Toggle between original and quantized view")
        self.segmented_check.toggled.connect(self._on_view_toggle)
        view_row.addWidget(self.segmented_check)
        view_row.addStretch()
        quant_layout.addLayout(view_row)
        
        layout.addWidget(quant_group)
        
        # Palette display
        self.palette_widget = QWidget()
        self.palette_layout = QHBoxLayout(self.palette_widget)
        self.palette_layout.setContentsMargins(0, 0, 0, 0)
        self.palette_layout.setSpacing(2)
        layout.addWidget(self.palette_widget)
        
        # Project button
        self.project_btn = QPushButton("Project to Mesh (Ctrl+P)")
        self.project_btn.setToolTip("Apply texture colors to mesh UVs")
        self.project_btn.clicked.connect(self.project_requested.emit)
        self.project_btn.setEnabled(False)
        layout.addWidget(self.project_btn)
        
        layout.addStretch()
    
    def _toggle_advanced(self, checked):
        self.advanced_widget.setVisible(checked)
    
    def keyPressEvent(self, e):
        """Handle keyboard shortcuts for texture panel"""
        key = e.key()
        mods = e.modifiers()
        
        if mods & Qt.ControlModifier:
            if key == Qt.Key_T:
                self._load_texture_dialog()
                return
            elif key == Qt.Key_Q:
                self._do_quantize()
                return
            elif key == Qt.Key_P:
                if self.project_btn.isEnabled():
                    self.project_requested.emit()
                return
        elif key == Qt.Key_S:
            self.segmented_check.setChecked(not self.segmented_check.isChecked())
            return
        
        super().keyPressEvent(e)
    
    def load_texture(self, filepath: str = None) -> bool:
        """Load texture from file path or show dialog"""
        if filepath is None:
            filepath, _ = QFileDialog.getOpenFileName(
                self, "Open Texture", "",
                "Images (*.png *.jpg *.jpeg *.bmp *.tga *.tif);;All Files (*)"
            )
        
        if filepath and self.texture_mgr.load(filepath):
            self._update_display()
            self.texture_loaded.emit()
            self.project_btn.setEnabled(True)
            return True
        return False
    
    def set_uv_data(self, uv_triangles: List[Tuple]):
        """Set UV triangle data for overlay"""
        self.uv_triangles = uv_triangles
        if self.show_uv_overlay:
            self._update_display()
    
    def get_texture_manager(self) -> TextureManager:
        """Get the texture manager"""
        return self.texture_mgr
    
    def do_quantize(self):
        """Public method to trigger quantization (for hotkey)"""
        self._do_quantize()
    
    def _load_texture_dialog(self):
        self.load_texture()
    
    def _update_display(self):
        """Update texture display"""
        if self.segmented_check.isChecked() and self.texture_mgr.segmented_image:
            img_data = self.texture_mgr.get_image_as_bytes('segmented')
        else:
            img_data = self.texture_mgr.get_image_as_bytes('working')
        
        if img_data is None:
            self.texture_label.setText("No texture loaded")
            return
        
        data, width, height = img_data
        
        qimg = QImage(data, width, height, width * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        # UV overlay
        if self.show_uv_overlay and self.uv_triangles:
            painter = QPainter(pixmap)
            painter.setPen(QPen(QColor(255, 255, 0, 180), 1))
            
            w, h = pixmap.width(), pixmap.height()
            for uv_tri in self.uv_triangles:
                if len(uv_tri) >= 3:
                    points = []
                    for u, v in uv_tri:
                        x = u * (w - 1)
                        y = (1.0 - v) * (h - 1)
                        points.append(QPointF(x, y))
                    
                    if len(points) >= 3:
                        painter.drawLine(points[0], points[1])
                        painter.drawLine(points[1], points[2])
                        painter.drawLine(points[2], points[0])
            
            painter.end()
        
        label_size = self.texture_label.size()
        scaled = pixmap.scaled(
            label_size.width() - 10,
            label_size.height() - 10,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.texture_label.setPixmap(scaled)
    
    def _on_flip_h(self, checked):
        self.texture_mgr.set_flip_h(checked)
        self._update_display()
        self.texture_updated.emit()
    
    def _on_flip_v(self, checked):
        self.texture_mgr.set_flip_v(checked)
        self._update_display()
        self.texture_updated.emit()
    
    def _on_uv_toggle(self, checked):
        self.show_uv_overlay = checked
        self._update_display()
    
    def _on_view_toggle(self, checked):
        """Toggle between original and segmented view"""
        self._update_display()
    
    def _do_quantize(self):
        if self.texture_mgr.working_image is None:
            return
            
        # Get method - reordered: Uniform first
        methods = ['uniform', 'kmeans', 'median_cut', 'octree']
        method = methods[self.method_combo.currentIndex()]
        
        # Apply advanced options
        self.texture_mgr.blur_radius = self.blur_spin.value()
        self.texture_mgr.saturation_boost = self.sat_slider.value() / 100.0
        self.texture_mgr.dither = self.dither_check.isChecked()
        self.texture_mgr.ignore_threshold = self.thresh_spin.value()
        
        # Quantize
        num_colors = self.num_colors_spin.value()
        palette = self.texture_mgr.quantize(num_colors, method)
        
        self._update_palette_display()
        self.segmented_check.setChecked(True)  # Auto-show segmented
        self._update_display()
        self.palette_updated.emit(palette)
    
    def _update_palette_display(self):
        while self.palette_layout.count():
            item = self.palette_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        for i, color in enumerate(self.texture_mgr.palette):
            swatch = QLabel(str(i))
            swatch.setAlignment(Qt.AlignCenter)
            swatch.setFixedSize(24, 24)
            brightness = sum(color) / 3
            text_color = 'black' if brightness > 128 else 'white'
            swatch.setStyleSheet(
                f"background: rgb({color[0]},{color[1]},{color[2]}); "
                f"color: {text_color}; font-weight: bold; border-radius: 3px;"
            )
            self.palette_layout.addWidget(swatch)
        
        self.palette_layout.addStretch()
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()


class ToolPanel(QWidget):
    """Panel for paint tools and brush settings"""
    
    tool_changed = Signal(int)
    color_changed = Signal(int)
    brush_size_changed = Signal(float)
    subdivision_changed = Signal(int)
    auto_subdivide_changed = Signal(bool)
    clear_masks_requested = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_palette = list(DEFAULT_COLORS)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Tools
        tools_group = QGroupBox("Tools")
        tools_layout = QGridLayout(tools_group)
        
        self.tool_buttons = []
        tools = [
            ("Paint (P)", 0, 0),
            ("Erase (E)", 0, 1),
            ("Mask (M)", 1, 0),
            ("Unmask (U)", 1, 1),
            ("Pick (I)", 2, 0),
        ]
        
        for i, (label, row, col) in enumerate(tools):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.clicked.connect(lambda c, idx=i: self._on_tool_clicked(idx))
            tools_layout.addWidget(btn, row, col)
            self.tool_buttons.append(btn)
        
        # Clear Masks button (not checkable)
        clear_btn = QPushButton("Clear Masks (C)")
        clear_btn.setToolTip("Clear all masks from mesh")
        clear_btn.clicked.connect(self.clear_masks_requested.emit)
        tools_layout.addWidget(clear_btn, 2, 1)
        
        self.tool_buttons[0].setChecked(True)
        layout.addWidget(tools_group)
        
        # Brush
        brush_group = QGroupBox("Brush")
        brush_layout = QVBoxLayout(brush_group)
        
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Size [/]:"))
        
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(1, 200)
        self.size_slider.setValue(20)
        self.size_slider.valueChanged.connect(self._on_size_changed)
        size_row.addWidget(self.size_slider)
        
        self.size_label = QLabel("0.10")
        self.size_label.setMinimumWidth(40)
        size_row.addWidget(self.size_label)
        
        brush_layout.addLayout(size_row)
        
        self.auto_subdiv_check = QCheckBox("Auto Subdivide")
        self.auto_subdiv_check.setChecked(True)
        self.auto_subdiv_check.toggled.connect(self.auto_subdivide_changed.emit)
        brush_layout.addWidget(self.auto_subdiv_check)
        
        depth_row = QHBoxLayout()
        depth_row.addWidget(QLabel("Max Depth:"))
        
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(1, 10)
        self.depth_spin.setValue(6)
        self.depth_spin.valueChanged.connect(self.subdivision_changed.emit)
        depth_row.addWidget(self.depth_spin)
        
        brush_layout.addLayout(depth_row)
        layout.addWidget(brush_group)
        
        # Colors
        colors_group = QGroupBox("Colors (1-9)")
        colors_layout = QGridLayout(colors_group)
        
        self.color_buttons = []
        for i, color in enumerate(DEFAULT_COLORS):
            btn = QPushButton(str(i))
            btn.setCheckable(True)
            btn.setFixedSize(32, 32)
            self._style_color_button(btn, color, i == 1)
            btn.clicked.connect(lambda c, idx=i: self._on_color_clicked(idx))
            colors_layout.addWidget(btn, i // 3, i % 3)
            self.color_buttons.append(btn)
        
        self.color_buttons[1].setChecked(True)
        layout.addWidget(colors_group)
        
        layout.addStretch()
    
    def _style_color_button(self, btn, color, selected=False):
        brightness = sum(color) / 3
        text_color = 'black' if brightness > 128 else 'white'
        border = '3px solid white' if selected else '2px solid #555'
        btn.setStyleSheet(
            f"background: rgb({color[0]},{color[1]},{color[2]}); "
            f"color: {text_color}; font-weight: bold; border: {border};"
        )
    
    def set_palette(self, palette: List[Tuple[int, int, int]]):
        """Update color buttons with new palette"""
        self.current_palette = palette
        for i, btn in enumerate(self.color_buttons):
            if i < len(palette):
                selected = btn.isChecked()
                self._style_color_button(btn, palette[i], selected)
    
    def set_color(self, color_idx: int):
        """Set current color selection"""
        for i, btn in enumerate(self.color_buttons):
            btn.setChecked(i == color_idx)
            if i < len(self.current_palette):
                self._style_color_button(btn, self.current_palette[i], i == color_idx)
    
    def set_brush_size(self, size: float):
        self.size_slider.blockSignals(True)
        self.size_slider.setValue(int(size * 200))
        self.size_slider.blockSignals(False)
        self.size_label.setText(f"{size:.2f}")
    
    def _on_tool_clicked(self, idx):
        for i, btn in enumerate(self.tool_buttons):
            btn.setChecked(i == idx)
        self.tool_changed.emit(idx)
    
    def _on_color_clicked(self, idx):
        for i, btn in enumerate(self.color_buttons):
            btn.setChecked(i == idx)
            if i < len(self.current_palette):
                self._style_color_button(btn, self.current_palette[i], i == idx)
        self.color_changed.emit(idx)
    
    def _on_size_changed(self, value):
        size = value / 200.0
        self.size_label.setText(f"{size:.2f}")
        self.brush_size_changed.emit(size)


class DisplayPanel(QWidget):
    """Panel for display options"""
    
    wireframe_toggled = Signal(bool)
    ground_toggled = Signal(bool)
    texture_toggled = Signal(bool)  # New signal
    frame_requested = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        group = QGroupBox("Display")
        group_layout = QVBoxLayout(group)
        
        # New Texture Toggle
        self.texture_check = QCheckBox("Texture (T)")
        self.texture_check.setChecked(False)
        self.texture_check.toggled.connect(self.texture_toggled.emit)
        group_layout.addWidget(self.texture_check)
        
        self.wireframe_check = QCheckBox("Wireframe (W)")
        self.wireframe_check.setChecked(False)  # Default OFF
        self.wireframe_check.toggled.connect(self.wireframe_toggled.emit)
        group_layout.addWidget(self.wireframe_check)
        
        self.ground_check = QCheckBox("Ground (G)")
        self.ground_check.setChecked(True)
        self.ground_check.toggled.connect(self.ground_toggled.emit)
        group_layout.addWidget(self.ground_check)
        
        frame_btn = QPushButton("Frame View (F)")
        frame_btn.clicked.connect(self.frame_requested.emit)
        group_layout.addWidget(frame_btn)
        
        layout.addWidget(group)
        layout.addStretch()