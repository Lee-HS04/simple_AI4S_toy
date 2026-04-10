# this code simulates the descent of a projectile launched from various angles under the effect of variable air resistance and fixed gravity. 
# The surface area of the projectile is variable
# the introduction of air resistance makes the simulation more realistic
import numpy
import pandas

def experiment(mass, area, velocity_u, angle_launched):
    #defining our constants
    g = -9.81 # gravity
    rho = 1 # air density
    Cd = 0.01 # drag coefficient (this is how much an object resists air)
    time_step = 0.001 # time step = 1ms
    
    #variables to be tracked in experiment
    time  = 0
    x = 0 # current x-coord
    y = 0 # current y-coord
    velo_x = velocity_u*numpy.cos(angle_launched) # v_x = v_u * cos(theta)
    velo_y = velocity_u*numpy.sin(angle_launched) # v_y = v_u * sin(theta)
    
    data = []
    
    while y > 0: #y>0 means object still falling
        
        velocity = numpy.sqrt(velo_x**2 + velo_y**2) 
        drag = 0.5 * rho * velocity**2 * Cd * area
        
        #Accelerations
        a_x = -(drag * (velo_x/velocity)) / mass
        a_y = g - (drag * (velo_y/velocity) / mass)
        
        #Update velocities 
        velo_x += a_x*time_step
        velo_y += a_y*time_step
        
        #Update positions
        x += velo_x*time_step
        y += velo_y*time_step
        
        # Keep track of how much time has elapsed
        time += time_step
        
        data.append((time, x, y, velo_x, velo_y))

    return pandas.DataFrame(data, columns = ['time', 'x', 'y', 'velo_x', 'velo_y'])
        
    