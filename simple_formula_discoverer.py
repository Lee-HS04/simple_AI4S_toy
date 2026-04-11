from pysr import PySRRegressor
from simulator import experiment
from data_processor import data_cleaner

data = data_cleaner(experiment(10, 2, 200, 45)) # Parameters from left to right: mass, area, inital velocity, launch angle
# About the parameters: if we are using a large initial velocity, we need a very small time step to accurately capture the changes in acceleration as the initial acceleration will be huge due to the drag force. 


# Define features (X) and target (y)
# We include 'v_smooth' so pysr can discover the drag formula
X = data[['time', 'mass', 'area', 'smoothed_velocity']] 
y = data['smoothed_acceleration']

model = PySRRegressor(
    niterations=100,           
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["square", "inv"], 
    model_selection="best",     # Balance simplicity and accuracy. This prevents the pysr from coming up with super long formulas
    maxsize=20, 
)
# LEARNER's note: PySr is an evolution based machine. It sees all formulas as trees with operators as roots and values as leaves. PySr creates many random formulas and tests them against the data. Those with huge errors are eliminated. The remaining ones are randomly preserved, mixed, or mutated. PySr obeys Occam's Razor (see data_processor.py)
                # preserved: The surviving formula continues to the next round of testing as it is
                # mixed: Surviving formulas could their equations changed by including other formulas' parts or removing certain parts before moving on to next round of testing
                # mutated: Suriviving formulas' operators are randomly changed to other operators. e.g. a + is changed to a *
                
print("AI is now analyzing the data to discover physical laws...")
model.fit(X, y)

print("\n--- DISCOVERY RESULTS ---")
print(model.equations_)