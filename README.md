# simple_AI4S_toy

### Requirements
<ol>
<li>Download all the libraries listed in requirements.txt (Just run pip install -r requirements.txt in the terminal)</li>
<li>If pysr fails to download, try this command: pip install pysr --trusted-host pypi.org --trusted-host files.pythonhosted.org</li>
<li> After installing dependencies, run this command: python -c "import pysr; pysr.install()" to install Julia for pysr </li>
<ol>

### Usage
* Please note that the values given to the simulators are to be in the following units:
  * mass in kgs
  * area in m^2
  * velocities in m/s
  * angles in radians
* Feel free to change the constants in simulator.py
* When running simple_formula_discoverer.py, the program will generate graphs. Make sure to close the graphs to prevent matplotlib from blocking the program from continuing.
* Do note that the window size used in data_processor.py depends on the size of input data, and that the size of this input data depends on how fast the ball falls to the ground (if the ball falls very quickly, the data set will be smaller). The window_size MUST NOT be bigger than the number of data points in the input data