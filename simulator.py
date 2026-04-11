# this code simulates the descent of a projectile launched from various angles under the effect of variable air resistance and fixed gravity. 
# The surface area of the projectile is variable
# the introduction of air resistance makes the simulation more realistic
import numpy
import pandas
import matplotlib.pyplot as plt

def experiment(mass, area, velocity_u, angle_launched):
    #defining our constants
    g = -9.81 # gravity
    rho = 1 # air density
    Cd = 0.5 # drag coefficient (this is how much an object resists air)
    time_step = 0.001 # time step = 1ms
    
    #variables to be tracked in experiment
    time  = 0
    x = 0 # current x-coord
    y = 0 # current y-coord
    velo_x = velocity_u*numpy.cos(numpy.radians(angle_launched)) # v_x = v_u * cos(theta)
    velo_y = velocity_u*numpy.sin(numpy.radians(angle_launched)) # v_y = v_u * sin(theta)
    
    data = []
    
    while y >= 0: #y>0 means object still falling
        
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
        
        # TODO: Add noise to the data to increase realism. Do make sure the noise added is not too big though (somewhere around 1mm will do)
        y_noisy = y + numpy.random.normal(0,0.001)
        x_noisy = x + numpy.random.normal(0,0.001)
        
        data.append((time, x_noisy, y_noisy, velo_x, velo_y, mass, area))

    return pandas.DataFrame(data, columns = ['time', 'x', 'y', 'velo_x', 'velo_y', 'mass', 'area'])
        
    
def plot_graph(data):
    x_velocities = []
    y_velocities = []
    duration = []
    for sample in data:
        x_velocities.append(sample[3])
        y_velocities.append(sample[4])
        duration.append(sample[0])
    
    fig, graph = plt.subplots(nrows = 1, ncols = 2)
    graph[0].plot(duration, x_velocities, marker='o', linestyle='-', label='x velocity')
    graph[0].set_title('x velocity against time')
    graph[0].set_xlabel('time(s)')
    graph[0].set_ylabel('x velocity (m/s)')
    graph[1].plot(duration, y_velocities, marker='o', linestyle='-', label='y velocity')
    graph[1].set_title('y velocity against time')
    graph[1].set_xlabel('time(s)')
    graph[1].set_ylabel('y velocity (m/s)')
    plt.tight_layout()
    plt.show()
    
# Use these to test and see if the simulator and graph plotter are working 
#sim1 = experiment(10, 2, 15, 30)
#plot_graph(sim1.values)
    