import pandas as pd
import matplotlib.pyplot as plt
from simulator import plot_graph, experiment

def data_cleaner(data):
    data_local = data.copy() # preserve original data
    
    # remove any data set with y<0
    data_local = data_local[data_local['y']>=0]
    
    # remove the 2nd data set in a pair of data sets that gives a velocity surpassing a certain threshold
    # calculate y-velocity between 2 data sets
    data_local['velocity'] = data_local['y'].diff()/data_local['time'].diff() #diff() calculates the difference between the values of the current and previous sets' specified fields
    # remove the data set if its 'velovity' is greater than x
    x = 330 # speed of sound
    data_local = data_local[data_local['velocity'].abs()<x].copy() #we add .abs() to account for both directions
    # LEARNER's note: what we just did is called vectorized filtering. We use the sequence of code above to remove absurd velocities instead of for loops as vectorized filtering is much faster.
    # LEARNER's note: we use .copy() as it creates a new separate data table consisting of only the filtered data. This allows pandas to delete the messy, larger table, thus saving memory and lookup time later on
    # LEARNER's note: we ald calculated the y and x velo in simulator, so why do it agn? We do this bcs those are the ground truths, whereas we want our AI to be able to discover these truths on its own instead of just "copying" our answers
    
    #calculate acceleration using velocity calculated 
    data_local['acceleration'] = data_local['velocity'].diff()/data_local['time'].diff()
    
    data_local = data_local.dropna() # dropna() tells pandas to remove any sets that have NaN values. e.g. in our vectorized filtering, a new entry will be generated when calculating the diff for the first set as there is no previous set
    # This dummy set used to calculate the .diff() of first set will have NaN value, so we should remove it.
    
    # Smooth the data (without smoothing the results will be too random as differentiation increases the effect of noise)
    
    
    #plot graph to see results
    plt.plot(data_local['time'], data_local['acceleration'], marker = 'o', linestyle = '-')
    plt.title('acceleration velocity against time')
    plt.xlabel('time(s)')
    plt.ylabel('acceleration (m/s^2)')
    plt.show()
    
    return data_local

data_cleaner(experiment(10,2,15,30))