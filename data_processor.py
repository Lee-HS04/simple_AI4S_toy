import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from simulator import experiment

def data_cleaner(data):
    data_local = data.copy() # preserve original data
    
    # remove any data set with y<0
    data_local = data_local[data_local['y']>=0].copy()
    
    window_length = 151
    
    # We have realised that smoothing only the acceleration at the end using savgol is not enough to create a sufficiently clean data set for PySr to analyse. Therefore, we shall smooth at every step using savgol.
    #STEP 1 OF SMOOTHING: SMOOTH Y
    data_local['y_smoothed'] = savgol_filter(data_local['y'].fillna(0), window_length, polyorder = 2)
    
    # remove the 2nd data set in a pair of data sets that gives a velocity surpassing a certain threshold
    # calculate y-velocity between 2 data sets
    data_local['velocity'] = data_local['y_smoothed'].diff()/data_local['time'].diff() #diff() calculates the difference between the values of the current and previous sets' specified fields
    # remove the data set if its 'velovity' is greater than x
    x = 330 # speed of sound
    data_local = data_local[data_local['velocity'].abs()<x].copy() #we add .abs() to account for both directions
    # LEARNER's note: what we just did is called vectorized filtering. We use the sequence of code above to remove absurd velocities instead of for loops as vectorized filtering is much faster.
    # LEARNER's note: we use .copy() as it creates a new separate data table consisting of only the filtered data. This allows pandas to delete the messy, larger table, thus saving memory and lookup time later on
    # LEARNER's note: we ald calculated the y and x velo in simulator, so why do it agn? We do this bcs those are the ground truths, whereas we want our AI to be able to discover these truths on its own instead of just "copying" our answers
    
    #STEP 2 OF SMOOTHING: SMOOTH VELOCITY
    data_local['smoothed_velocity'] = savgol_filter(data_local['velocity'].fillna(0), window_length, polyorder =2)
    
    #calculate acceleration using velocity calculated 
    data_local['acceleration'] = data_local['smoothed_velocity'].diff()/data_local['time'].diff()
    
    
    # STEP 3 OF SMOOTHING: SMOOTH ACCELERATION
    # Smooth the data (without smoothing the results will be too random as differentiation increases the effect of noise). 
    # METHOD 1: rolling average
    #data_local['smoothed_acceleration'] = data_local['acceleration'].rolling(window=1000).mean()
    # LEARNER's note: rolling average is a method where we take the average of a certain value in a window of sets. It 'rolls' because we slide the window over the data sets 1 by 1. e.g. our window here is 10, so the first window will be 9 dummies + 1st set, 2nd window will be 8 dummies + 1st set + 2nd set and so on.
    # Syntax breakdown: pandas does the rolling window for us using the .rolling() method. window is the amount of sets used to calculate the value (in this case the mean).
    # e.g. when we set window=10, this means that the 10th set in the window will have its specified value (smoothed_acceleration in this case) set to the average of the acceleration of the 10 sets in the window
    # LEARNER's note: try out different window sizes. Larger window sizes introduce better smoothing but larger lags (a right shift of the smoothed curve compared to the actual curve). Smaller window sizes smooth less but have smaller lag as well
    # LEARNER's note: we have realised that the rolling average is insufficient for our purpose. A smaller window does not smooth remotely enough to make the curve make sense. Using a huge window introduces lag and data losses too large to be acceptable
    
    #METHOD 2: Savtizky-Golay filter
    data_local['smoothed_acceleration'] = savgol_filter(data_local['acceleration'].fillna(0), window_length, polyorder = 2)
    # LEARNER's note: Savitzky-Golay (savgol) filter works by drawing a polynomial curve that best fits the defined window of points. It does chooses this curve of best fit using the least squares method. It then takes the value of the curve at the central point of the window and returns that value. The window shifts one by one like in rolling average
    # LEAST SQUARES: in the least squares method, we look at the difference between the actual data value and the value it would have on the curve at the same x coordinate. This is the error. To ensure that +ve and -ve errors dont cancel out, we square them. 
    # Additionally, squaring heavily punishes larger errors (error>1) while reducing the effect of smaller errors. The savgol filter then sums these errors and uses calculus to find the coefficients for the polynomial that make its curve have as small of a total error as possible
    # LEARNER's note: How do we choose the order of the polynomial curve (polyorder)? We choose the order based on the order in the formula we expect to see. e.g. in this case, we expect y = ut + 0.5at^2. This is 2nd order.
    # BUT what if we are trying to discover a new formula and hence do not know what the order should be?
        # Mathmatically: Any smooth continuous function can be expanded into the Taylor Series. The Taylor series is basically just a polynomial of n-degrees. If we look at a small enough window of time, the polynomial would still be low order. Hence, usually start with low polyorder. Do note that in this case the window of time is our window*time_step used in simulator
        # Occam's Razor: Occam's Razor states that among competing hypotheses, the one with the fewest assumptions is usually the correct one as "Entities should not be multiplied beyond necessity". In this case, among competing formulas, the one with the lowest degree is usualyl most likely to be true.
        # Cross-Validation: This is a technique used to find the best polyorder. We split the data into training and validation sets (like in Reinforcement Learning). Try multiple filters on the training set and see which one gives us a formula that can most accurately predict the validation sets.
        # Spectral Analysis (Fourier Transform): What if the underlying law happens to involve high frequency signals? This causes jitters in our graphs as well. 
            # In this case, we can use Fourier Transform to decompose the curve into the sine waves it is composed of (every curve no matter the shape is just a superposition of sine waves). Each sine wave represemts a frequency. If we see that the high frequency waves are all small in amplitude, the jittery graph is probably just a result of noise.
            # However, if the there are high frequency waves with consistently large amplitudes, this means there indeed are high frequency signals, and that our jittery graph is caused not just by noise. In this case, we must use a smaller window_length and higher polyorder
            # smaller window length: avoids bunching too many data points tgt and averaging, causing oversmoothing (NOTE: not too small as a window length that is too small doesnt smooth enough)
            # higher polyorder: higher frequency data results in curves that change direction quickly. A higher polyorder allows best fit curve to capture these changes instead of just cutting them off due to not being able to follow.
    
    
    data_local = data_local.dropna() # dropna() tells pandas to remove any sets that have NaN values. 
    #e.g. in our vectorized filtering, a new entry will be generated when calculating the diff for the first set as there is no previous set, in rolling window window-1 dummies will be created.
    # All dummy sets have NaN value (be it in the y field, the velocity field or the acceleration field), and dropna() removes all such fields.
    
    # CROP THE FIRST AND LAST window_length data points. 
    # This is because savgol looks at the data points window_length/2 before and after the current data point. When we are at the 1st data point, the savgol filter would only be looking at the window_lenght/2 points after the current data point as there are no data points before it. This causes imbalance. Vice versa for last data point
    data_local = data_local.iloc[window_length: -window_length].copy()
    
    #plot graph to see results. Here we plot the original acceleration and smoothed acceleration side by side to see the difference
    fig, graph = plt.subplots(nrows = 1, ncols = 2)
    graph[0].plot(data_local['time'], data_local['acceleration'], marker = 'o', linestyle = '-')
    graph[0].set_title('acceleration against time')
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