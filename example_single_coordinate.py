from pygl_window import *

# A simple example demonstrating the APIs.
window = PyglWindow()

# Use the follow command to create objects that you would like to render throughout the whole process.
window.render_arrow(tip=[1, 0, 0], tail=[0, 0, 0], material=[1, 0, 0])
window.render_arrow(tip=[0, 1, 0], tail=[0, 0, 0], material=[0, 1, 0])
window.render_arrow(tip=[0, 0, 1], tail=[0, 0, 0], material=[0, 0, 1])

window.show()
