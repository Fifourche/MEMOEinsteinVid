import numpy as np
from manim import *

class WireCube(VGroup):
    def __init__(self, side=1, center=[0.,0.,0.], color=BLUE):
        super().__init__()
        
        # edges of a unit cube with corner at (0,0,0)
        edges = np.array([[(0,0,0),(1,0,0)],
            [(0,0,0),(0,1,0)],
            [(0,0,0),(0,0,1)],
            [(1,0,0),(1,1,0)],
            [(1,0,0),(1,0,1)],
            [(0,1,0),(1,1,0)],
            [(0,1,0),(0,1,1)],
            [(1,1,0),(1,1,1)],
            [(1,1,1),(1,0,1)],
            [(1,1,1),(0,1,1)],
            [(1,0,1),(0,0,1)],
            [(0,1,1),(0,0,1)]], dtype="f")
                
        # define the cube as lines
        for edge in edges:
            edge -= center + np.array([0.5,0.5,0.5])
            edge *= side
            self.add(Line(*edge, color=color))

class Particle(Sphere):
    def __init__(self, radius=0.1, color="#4DFF3D", resolution=6, **kwargs):
        super().__init__(checkerboard_colors=[color,color], stroke_color=color, radius=radius, resolution=resolution, **kwargs)

class Particles(VGroup):
    def __init__(self, N=10, r_init=np.zeros((10,3)), **kwargs):
        super().__init__()

        for i in range(N):
            self.add(Particle(**kwargs).move_to(r_init[i,:]))
