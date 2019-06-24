import numpy as np
import sys, os

from PyQt5.QtCore import pyqtSignal, QSize, Qt, QTimer, QPoint
from PyQt5.QtWidgets import (QAction, QApplication, QGridLayout, QLabel,
                             QMainWindow, QMessageBox, QOpenGLWidget, QScrollArea,
                             QSizePolicy, QSlider, QWidget)
import OpenGL.GL as gl

def print_error(*message):
    print('\033[91m', *message, '\033[0m')

def print_ok(*message):
    print('\033[92m', *message, '\033[0m')

def print_warning(*message):
    print('\033[93m', *message, '\033[0m')

def print_info(*message):
    print('\033[96m', *message, '\033[0m')

class PyglWidget(QOpenGLWidget):
    def __init__(self, shapes, fps=30, parent=None):
        super(PyglWidget, self).__init__(parent)

        # Camera-related.
        self.last_pos = QPoint()
        self.lon_speed = 0.01
        self.lat_speed = 0.01

        # Let's determine the best camera positions.
        max_corner = -np.array([np.inf, np.inf, np.inf])
        min_corner = -max_corner
        for shape in shapes:
            max_corner = np.maximum(max_corner, shape.max_corner)
            min_corner = np.minimum(min_corner, shape.min_corner)

        dist = 1.25 * (np.linalg.norm(max_corner - min_corner) / 2)

        self.camera_lookat = (max_corner + min_corner) / 2
        view_dir = np.array([3.0, 4.0, 5.0])
        view_dir /= np.linalg.norm(view_dir)
        self.camera_position = self.camera_lookat + view_dir * dist
        self.camera_up = np.array([0.0, 0.0, 1.0])

        self.init_height = 800
        self.init_width = 1200
        self.h_ratio, self.w_ratio = 1.0, 1.0
        self.asp_ratio = self.init_width * 1.0 / self.init_height

        self.shapes = shapes
        self.shape_lists = []

        timer = QTimer(self)
        timer.timeout.connect(self.update_frame_idx)
        self.frame_idx = 0
        timer.start(int(1000 / fps))

    def update_frame_idx(self):
        self.frame_idx += 1
        self.update()

    # Don't change the function signature here.
    def minimumSizeHint(self):
        return QSize(min(self.init_height, self.init_width), min(self.init_height, self.init_width))

    # Don't change the function signature here.
    def sizeHint(self):
        return QSize(self.init_width, self.init_height)

    # Don't change the function signature here.
    def resizeGL(self, width, height):
        side = min(width, height)
        if side < 0:
            return
        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        self.h_ratio = height * 1.0 / self.init_height
        self.w_ratio = width * 1.0 / self.init_width
        gl.glOrtho(-1.5 * self.w_ratio * self.asp_ratio, +1.5 * self.w_ratio * self.asp_ratio,
                   -1.5 * self.h_ratio, +1.5 * self.h_ratio, -1e6, 1e6)

    def initializeGL(self):
        gl.glEnable(gl.GL_DEPTH_TEST)

        gl.glEnable(gl.GL_NORMALIZE)
        gl.glClearColor(0x11 / 255, 0x2F / 255, 0x41 / 255, 1.0)

        self.shape_lists = []
        for shape in self.shapes:
            l = gl.glGenLists(1)
            gl.glNewList(l, gl.GL_COMPILE)
            gl.glBegin(gl.GL_LINES)
            for triangle in shape.triangles:
                for i in range(3):
                    v = triangle[i]
                    u = triangle[(i + 1) % 3]
                    gl.glVertex3d(v[0], v[1], v[2])
                    gl.glVertex3d(u[0], u[1], u[2])
            gl.glEnd()
            gl.glEndList()
            self.shape_lists.append(l)

    def paintGL(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(-1.5 * self.w_ratio * self.asp_ratio, +1.5 * self.w_ratio * self.asp_ratio,
                   -1.5 * self.h_ratio, +1.5 * self.h_ratio, -1e6, 1e6)

        gl.glMatrixMode(gl.GL_MODELVIEW)
        # View matrix.
        V = self.generate_view_matrix()

        for shape, l in zip(self.shapes, self.shape_lists):
            material = shape.get_material(self.frame_idx)
            transform = shape.get_transform(self.frame_idx)
            gl.glColor(material[0], material[1], material[2])
            gl.glLoadMatrixd((V @ transform).T)
            gl.glCallList(l)

    # Don't change the function signature here.
    def mousePressEvent(self, event):
        self.last_pos = event.pos()

    # Don't change the function signature here.
    def mouseMoveEvent(self, event):
        # x: left to right.
        # y: top to bottom.
        dx = event.x() - self.last_pos.x()
        dy = event.y() - self.last_pos.y()

        if event.buttons() & Qt.LeftButton:
            # Update position, lookat, and up.
            point_to = self.camera_position - self.camera_lookat
            len = np.linalg.norm(point_to)
            lon = np.cross(self.camera_up, point_to)
            lon = lon / np.linalg.norm(lon)
            lat = np.cross(point_to, lon)
            lat = lat / np.linalg.norm(lat)

            lon_delta = -dx * self.lon_speed
            point_to = np.cos(lon_delta) * point_to + np.sin(lon_delta) * lon * len

            lat_delta = dy * self.lat_speed
            cos_lat = np.cos(lat_delta)
            sin_lat = np.sin(lat_delta)
            new_point_to = cos_lat * point_to + lat * sin_lat * len

            self.camera_position = self.camera_lookat + new_point_to
            self.camera_up = cos_lat * lat - sin_lat * point_to / len

            self.update()

        self.last_pos = event.pos()

    def generate_view_matrix(self):
        V = np.eye(4)
        view_z = self.camera_position - self.camera_lookat
        view_z /= np.linalg.norm(view_z)
        view_x = np.cross(self.camera_up, view_z)
        view_x /= np.linalg.norm(view_x)
        view_y = np.cross(view_z, view_x)
        view_y /= np.linalg.norm(view_y)
        Rt = np.array([view_x, view_y, view_z])
        V[:3,:3] = Rt
        V[:3,-1] = -Rt.dot(self.camera_position)
        return V

class PyglQtMainWindow(QMainWindow):
    def __init__(self, name, parent):
        super(PyglQtMainWindow, self).__init__()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        self.gl_widget = PyglWidget(shapes=parent.shapes, fps=parent.fps)

        self.gl_widget_area = QScrollArea()
        self.gl_widget_area.setWidget(self.gl_widget)
        self.gl_widget_area.setWidgetResizable(True)
        self.gl_widget_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.gl_widget_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.gl_widget_area.setSizePolicy(QSizePolicy.Ignored,
                QSizePolicy.Ignored)
        self.gl_widget_area.setMinimumSize(50, 50)

        central_layout = QGridLayout()
        central_layout.addWidget(self.gl_widget_area, 0, 0)
        central_widget.setLayout(central_layout)

        self.setWindowTitle('Pygl window: ' + name)
        self.name = name
        self.resize(1200, 800)

        self.parent = parent
        if parent.record:
            timer = QTimer(self)
            timer.timeout.connect(self.take_snapshot)
            self.frame_idx = 0
            timer.start(int(1000 / parent.fps))

    def take_snapshot(self):
        if self.parent.record:
            image = self.gl_widget.grabFramebuffer()
            os.makedirs(self.name, exist_ok=True)
            image.save(os.path.join(self.name, str(self.frame_idx) + '.png'))
            self.frame_idx += 1

class PyglShape(object):
    def __init__(self, vertices, faces, transform, material):
        assert vertices.shape[1] == 3
        assert faces.shape[1] == 3

        self.vertices = vertices
        self.faces = faces
        triangles = []
        for f in faces:
            triangles.append(vertices[f])
        self.triangles = np.array(triangles)

        self.default_transform = np.array(transform, dtype=np.float64)
        self.default_material = np.array(material, dtype=np.float64)
        self.transform_in_frames = {}
        self.material_in_frames = {}

        # Compute the bounding box for the default settings.
        transformed_vertices = np.hstack([self.vertices, np.ones((self.vertices.shape[0], 1))]) @ self.default_transform.T
        transformed_vertices[:, :3] /= transformed_vertices[:, -1][:, None]
        self.max_corner = np.max(transformed_vertices[:, :3], axis=0)
        self.min_corner = np.min(transformed_vertices[:, :3], axis=0)

    def set_transform(self, transform, frame_idx):
        transform = np.array(transform, dtype=np.float64)
        self.transform_in_frames[frame_idx] = transform

    def set_trs_transform(self, translation=np.zeros(3), rotation=np.eye(3), scaling=np.ones(3), frame_idx=0):
        transform = np.eye(4)
        translation = np.array(translation, dtype=np.float64)
        rotation = np.array(rotation, dtype=np.float64)
        scaling = np.array(scaling, dtype=np.float64)
        if scaling.size == 1:
            scaling = np.array([scaling[0], scaling[0], scaling[0]])
        transform[:3, :3] = rotation @ np.diag(scaling)
        transform[:3, -1] = translation
        self.set_transform(transform, frame_idx)

    def set_material(self, material, frame_idx):
        material = np.array(material, dtype=np.float64)
        self.material_in_frames[frame_idx] = material

    def get_transform(self, frame_idx):
        if frame_idx not in self.transform_in_frames:
            return self.default_transform
        return self.transform_in_frames[frame_idx]

    def get_material(self, frame_idx):
        if frame_idx not in self.material_in_frames:
            return self.default_material
        return self.material_in_frames[frame_idx]

class PyglWindow:
    def __init__(self, name='default window', fps=30, record=False):
        self.name = name
        self.shapes = []
        self.fps = fps
        self.record = record

    def render_shape(self, vertices, faces,
                     rotation=np.eye(3),
                     translation=np.array([0.0, 0.0, 0.0]),
                     scaling=np.array([1.0, 1.0, 1.0]),
                     material=np.array([0.3, 0.7, 0.1]),
                     commit=True):
        translation = np.array(translation, dtype=np.float64)
        scaling = np.array(scaling, dtype=np.float64)
        material = np.array(material, dtype=np.float64)
        transform = np.eye(4)
        transform[:3, :3] = rotation @ np.diag(scaling)
        transform[:3, -1] = translation
        shape = PyglShape(vertices, faces, transform, material)

        if commit:
            self.shapes.append(shape)
            return None
        else:
            return shape

    def render_arrow(self, tip=np.array([1.0, 0.0, 0.0]), tail=np.array([0.0, 0.0, 0.0]),
                     radius=0.025, tip_length=0.3, tip_radius=0.045,
                     rotation=np.eye(3),
                     translation=np.array([0.0, 0.0, 0.0]),
                     scaling=np.array([1.0, 1.0, 1.0]),
                     material=np.array([0.8, 0.1, 0.2]),
                     commit=True):
        tip = np.array(tip, dtype=np.float64)
        tail = np.array(tail, dtype=np.float64)
        dir = tip - tail
        height = np.linalg.norm(dir)
        dir /= height
        # Draw the cylinder.
        cylinder_height = height - tip_length
        cylinder = self.render_cylinder(center=tail + dir * cylinder_height / 2.0, dir=dir, radius=radius,
                                        height=cylinder_height, rotation=rotation, translation=translation,
                                        scaling=scaling, material=material, commit=False)
        # Draw the cone.
        cone = self.render_cone(center=tail + dir * cylinder_height, dir=dir, radius=tip_radius, height=tip_length,
                                rotation=rotation, translation=translation, scaling=scaling, material=material,
                                commit=False)
        # Assemble the arrow.
        v_cylinder, f_cylinder = cylinder.vertices, cylinder.faces
        v_cone, f_cone = cone.vertices, cone.faces
        v_arrow = np.vstack([v_cylinder, v_cone])
        f_arrow = np.vstack([f_cylinder, f_cone + v_cylinder.shape[0]])
        arrow = self.render_shape(vertices=v_arrow, faces=f_arrow, rotation=rotation, translation=translation,
                                  scaling=scaling, material=material, commit=commit)
        return arrow

    def render_sphere(self, center=np.array([0.0, 0.0, 0.0]), radius=1.0,
                      rotation=np.eye(3),
                      translation=np.array([0.0, 0.0, 0.0]),
                      scaling=np.array([1.0, 1.0, 1.0]),
                      material=np.array([0.2, 0.7, 0.3]),
                      commit=True):
        center = np.array(center, dtype=np.float64)
        # theta \in [0, 2pi], phi \in [-pi / 2, pi / 2].
        theta_num = 16
        phi_num = 8
        dtheta = 2.0 * np.pi / theta_num
        dphi = np.pi / phi_num
        vertices = np.zeros((theta_num * (phi_num - 1) + 2, 3))
        faces = np.zeros((theta_num * 2 + (phi_num - 2) * theta_num * 2, 3), dtype=np.int)
        vertices[0] = [0.0, 0.0, 1.0]
        for i in range(theta_num):
            vertices[1 + i] = [
                np.cos(np.pi / 2.0 - dphi) * np.cos(i * dtheta),
                np.cos(np.pi / 2.0 - dphi) * np.sin(i * dtheta),
                np.sin(np.pi / 2.0 - dphi)
            ]
            faces[i] = [0, i + 1, i + 2]
        faces[theta_num - 1, 2] = 1

        for i in range(phi_num - 2):
            phi = np.pi / 2.0 - dphi * (i + 2)
            for j in range(theta_num):
                vertices[1 + theta_num * (i + 1) + j] = [
                    np.cos(phi) * np.cos(j * dtheta),
                    np.cos(phi) * np.sin(j * dtheta),
                    np.sin(phi)
                ]
                faces[theta_num * (2 * i + 1) + 2 * j] = [
                    1 + theta_num * (i + 1) + j,
                    1 + theta_num * (i + 1) + j + 1,
                    1 + theta_num * i + j
                ]
                faces[theta_num * (2 * i + 1) + 2 * j + 1] = [
                    1 + theta_num * (i + 1) + j + 1,
                    1 + theta_num * i + j + 1,
                    1 + theta_num * i + j
                ]
            faces[theta_num * (2 * i + 3) - 2, 1] = 1 + theta_num * (i + 1)
            faces[theta_num * (2 * i + 3) - 1, 0] = 1 + theta_num * (i + 1)
            faces[theta_num * (2 * i + 3) - 1, 1] = 1 + theta_num * i
        vertices[1 + theta_num * (phi_num - 1)] = [0.0, 0.0, -1.0]

        for i in range(theta_num):
            faces[theta_num * (2 * phi_num - 3) + i] = [
                1 + theta_num * (phi_num - 2) + i + 1,
                1 + theta_num * (phi_num - 2) + i,
                1 + theta_num * (phi_num - 1)
            ]
        faces[2 * theta_num * (phi_num - 1) - 1, 0] = 1 + theta_num * (phi_num - 2)

        vertices = vertices * radius + center
        sphere = self.render_shape(vertices=vertices, faces=faces, rotation=rotation, translation=translation,
                                   scaling=scaling, material=material, commit=commit)
        return sphere

    def render_cube(self, center=np.array([0.0, 0.0, 0.0]), size=np.array([1.0, 1.0, 1.0]),
                    rotation=np.eye(3),
                    translation=np.array([0.0, 0.0, 0.0]),
                    scaling=np.array([1.0, 1.0, 1.0]),
                    material=np.array([0.2, 0.7, 0.3]),
                    commit=True):
        vertices = np.array([
            [-0.5, -0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, 0.5, 0.5],
            [0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, -0.5],
            [0.5, 0.5, 0.5]
        ]) * size + center
        faces = np.array([
            [0, 2, 6],
            [0, 6, 4],
            [1, 5, 7],
            [1, 7, 3],
            [7, 5, 4],
            [7, 4, 6],
            [3, 7, 6],
            [3, 6, 2],
            [1, 3, 2],
            [1, 2, 0],
            [1, 0, 4],
            [1, 4, 5]
        ])
        cube = self.render_shape(vertices=vertices, faces=faces, rotation=rotation, translation=translation,
                                 scaling=scaling, material=material, commit=commit)
        return cube

    def render_cylinder(self, center=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 1.0, 0.0]),
                        radius=1.0, height=1.0,
                        rotation=np.eye(3),
                        translation=np.array([0.0, 0.0, 0.0]),
                        scaling=np.array([1.0, 1.0, 1.0]),
                        material=np.array([0.2, 0.7, 0.3]),
                        commit=True):
        center = np.array(center, dtype=np.float64)
        dir = np.array(dir, dtype=np.float64)

        # Create a local coordinate.
        xyz = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
        x = xyz[np.argmax([np.linalg.norm(np.cross(n, dir)) for n in xyz])]
        x = np.cross(x, dir)
        x /= np.linalg.norm(x)
        y = np.cross(dir, x)

        n = 6
        d = np.pi * 2.0 / n
        vertices = [center + dir * height / 2.0, center - dir * height / 2.0]
        faces = []
        for i in range(n):
            v = (np.cos(d * i) * x + np.sin(d * i) * y) * radius
            vertices.append(vertices[0] + v)
            vertices.append(vertices[1] + v)
            i00 = 2 + i * 2 + 1
            i01 = 2 + (i * 2 + 3) % (2 * n)
            i10 = 2 + i * 2
            i11 = 2 + (i * 2 + 2) % (2 * n)
            faces.append([i00, i01, i10])
            faces.append([i10, i01, i11])
            faces.append([i10, i11, 0])
            faces.append([i01, i00, 1])
        vertices = np.array(vertices, dtype=np.float64)
        faces = np.array(faces)
        cylinder = self.render_shape(vertices=vertices, faces=faces, rotation=rotation, translation=translation,
                                     scaling=scaling, material=material, commit=commit)
        return cylinder

    def render_cone(self, center=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 1.0, 0.0]),
                    radius=1.0, height=1.0,
                    rotation=np.eye(3),
                    translation=np.array([0.0, 0.0, 0.0]),
                    scaling=np.array([1.0, 1.0, 1.0]),
                    material=np.array([0.2, 0.3, 0.8]),
                    commit=True):
        center = np.array(center, dtype=np.float64)
        dir = np.array(dir, dtype=np.float64)

        # Create a local coordinate.
        xyz = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
        x = xyz[np.argmax([np.linalg.norm(np.cross(n, dir)) for n in xyz])]
        x = np.cross(x, dir)
        x /= np.linalg.norm(x)
        y = np.cross(dir, x)

        n = 6
        d = np.pi * 2.0 / n
        vertices = [center, center + height * dir]
        faces = []
        for i in range(n):
            p = center + radius * (np.cos(d * i) * x + np.sin(d * i) * y)
            vertices.append(p)
            j = (i + 1) % n
            faces.append([0, j + 2, i + 2])
            faces.append([i + 2, j + 2, 1])
        vertices = np.array(vertices, dtype=np.float64)
        faces = np.array(faces)
        cone = self.render_shape(vertices=vertices, faces=faces, rotation=rotation, translation=translation,
                                 scaling=scaling, material=material, commit=commit)
        return cone

    def show(self, uncommitted_shapes=[]):
        self.shapes += uncommitted_shapes
        app = QApplication([])
        main_window = PyglQtMainWindow(self.name, parent=self)
        main_window.show()
        sys.exit(app.exec_())

if __name__ == '__main__':
    pass