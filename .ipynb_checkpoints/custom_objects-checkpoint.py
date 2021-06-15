import numpy as np
from manim import *
import math

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


class Gas(Particles):            
    def __init__(self,
              N = 15,
              radius = 0.1,
              x_lims=np.array([-0.5, 0.5]), 
              y_lims=np.array([-0.5, 0.5]),
              z_lims=np.array([-0.5, 0.5]),
              r_init=None, 
              v_init=None,
              v_std=4./1.,
              run_time=2,
              sim_time=4,
              *args,
              **kwargs):

        # for the box
        self.x_lims = x_lims
        self.y_lims = y_lims
        self.z_lims = z_lims

        # for the gas
        self.N = N
        self.radius = radius
        self.rng = None
        self.v_std = v_std
        self.r = r_init if r_init else self._random_init_r()
        self.v = v_init if v_init else self._random_init_v()
        self.v_std = np.std(self.v)

        # for animation
        self.run_time = run_time
        self.sim_time = sim_time

        self.update_pos_func = self.default_update_pos_func
        
        super().__init__(N=N, r_init=self.r, radius=radius)
        
        self.add_updater(self.update_particles)   
        
    def _random_init_r(self):
        # Initialize r
        if not self.rng:
            self.rng = np.random.default_rng()
        
        lows = [self.x_lims[0], self.y_lims[0], self.z_lims[0]]
        highs = [self.x_lims[1], self.y_lims[1], self.z_lims[1]]
        r_init = self.rng.uniform(low=lows, high=highs, size=(self.N,3))

        return r_init
    
    def _random_init_v(self):
        # Initialize v
        if not self.rng:
            self.rng = np.random.default_rng()
        v_init = self.rng.normal(loc=0, scale=self.v_std, size=(self.N,3))
        
        return v_init
        
        
    def update_pos(self, dt, *args, **kwargs):
        self.r, self.v = self.update_pos_func(self, dt)

    @staticmethod
    def default_update_pos_func(obj, dt):
        
        r = obj.r
        v = obj.v
        
        # update the positions
        r += dt * v

        # check for collisions with the walls
        lows = np.array([obj.x_lims[0], obj.y_lims[0], obj.z_lims[0]]).T
        highs = np.array([obj.x_lims[1], obj.y_lims[1], obj.z_lims[1]]).T
        
        walls = (r - lows < 0) | (r - highs > 0)

        while walls.any():
            v[walls] *= -1
            r[walls] -= obj.radius * np.sign(r[walls])
            walls = (r - lows < 0) | (r - highs > 0)
        
        return r, v

    @staticmethod
    def update_particles(particles, dt):
        particles.update_pos(dt)
        for i, particle in enumerate(particles):
            particle.move_to(particles.r[i,:])
            
    def construct(self):
        self.wait(self.sim_time)
        
        

class PropaSine:
    def __init__(self, k=[2*math.pi,0,0], polar=[0,1,0], omega=2*math.pi, t_range=[0,1,0.1], r_init = [0,0,0], phi=0, *args, **kwargs):
        # for now, t_max is useless...
        self.k = k
        self.polar = polar
        self.omega = omega
        self.t_range = t_range
        self.r_init = r_init
        self.phi = phi
        
        self.t = t_range[1]
        
        self.curve = ParametricFunction(self.sine_vector(self.t, self.k, self.polar, self.omega, self.r_init, self.phi), t_range=self.t_range, *args, **kwargs)
        
        def updater_test(m, dt):
            m.become(ParametricFunction(self.sine_vector(self.t, self.k, self.polar, self.omega, self.r_init, self.phi), t_range=self.t_range, *args, **kwargs))
            self.t += dt
            self.t_range[1] += dt
        
        self.curve.add_updater(updater_test)

    def sine_vector(self, t, k=[2*math.pi,0,0], polar=[0,1,0], omega=2*math.pi, r_init = [0,0,0], phi=0):
        k = np.array(k)
        polar = polar/np.linalg.norm(polar)
        r_init = np.array(r_init)

        def sine_wave(u):
            s = u*k/np.linalg.norm(k)
            s += math.sin(np.linalg.norm(k)*u - omega*t + phi)*polar + r_init
            return s
        return sine_wave
    
class PropaEM(VGroup):
    def __init__(self, k=[2*math.pi,0,0], polar=[0,1,0], omega=2*math.pi, t_range=[0,1,0.1], r_init = [0,0,0], phi=0, color_E=BLUE, color_B=YELLOW, *args, **kwargs):
        super().__init__()
        E = PropaSine(k, polar, omega, t_range, r_init, phi, color=color_E, *args, **kwargs)
        B = PropaSine(k, np.cross(k, polar), omega, t_range, r_init, phi + math.pi/2, color=color_B, *args, **kwargs)
        self.add(E.curve, B.curve) 