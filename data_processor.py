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
    
    # Smooth the data (without smoothing the results will be too random as differentiation increases the effect of noise). We do this using a rolling average
    data_local['smoothed_acceleration'] = data_local['acceleration'].rolling(window=10).mean()
    # LEARNER's note: rolling average is a method where we take the average of a certain value in a window of sets. It 'rolls' because we slide the window over the data sets 1 by 1. e.g. our window here is 10, so the first window will be 9 dummies + 1st set, 2nd window will be 8 dummies + 1st set + 2nd set and so on.
    # Syntax breakdown: pandas does the rolling window for us using the .rolling() method. window is the amount of sets used to calculate the value (in this case the mean).
    # e.g. when we set window=10, this means that the 10th set in the window will have its specified value (smoothed_acceleration in this case) set to the average of the acceleration of the 10 sets in the window
    
    data_local = data_local.dropna() # dropna() tells pandas to remove any sets that have NaN values. 
    #e.g. in our vectorized filtering, a new entry will be generated when calculating the diff for the first set as there is no previous set, in rolling window window-1 dummies will be created.
    # All dummy sets have NaN value, so we should remove them.
    
    #plot graph to see results. Here we plot the original acceleration and smoothed acceleration side by side to see the difference
    fig, graph = plt.subplots(nrows = 1, ncols = 2)
    graph[0].plot(data_local['time'], data_local['acceleration'], marker = 'o', linestyle = '-')
    graph[0].set_title('accleration against time')
    graph[0].set_xlabel('time(s)')
    graph[0].set_ylabel('a (m/s^2)')
    graph[1].plot(data_local['time'], data_local['smoothed_acceleration'], marker = 'o', linestyle = '-')
    graph[1].set_title('smoothed acceleration against time')
    graph[1].set_xlabel('time(s)')
    graph[1].set_ylabel('a (m/s^2)')
    plt.tight_layout()
    plt.show()
    
    return data_local

data_cleaner(experiment(10,2,15,30))