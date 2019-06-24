from pygl_window import *

# A simple example demonstrating the APIs.
window = PyglWindow()

window.render_arrow(tip=[1, 0, 0], tail=[0, 0, 0], material=[1, 0, 0])
window.render_arrow(tip=[0, 1, 0], tail=[0, 0, 0], material=[0, 1, 0])
window.render_arrow(tip=[0, 0, 1], tail=[0, 0, 0], material=[0, 0, 1])

sphere_center = np.array([0.5, 0.5, 0.5])
material = np.array([0.3, 0.6, 0.8])
sphere = window.render_sphere(center=sphere_center, radius=0.25, material=material, commit=False)

max_frame = 1024
for f in range(max_frame):
    # Let the sphere move.
    sphere_translate = np.array([0.0, np.sin(f / max_frame * np.pi * 2.0), 0.0])
    sphere.set_trs_transform(translation=sphere_translate, frame_idx=f)
    sphere.set_material(material=material + np.sin(f / max_frame * np.pi * 2.0) * 0.25, frame_idx=f)

window.show([sphere])
