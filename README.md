# Python_Profiling

Combines the outputs of profiling packages: line_profiler and memory_profiler into one plot, with annotations for function locations.

## Files
profiling_examples.ipynb -- demonstrates functionality by showing the output plots from two Python scripts

modified_Lorenz.py -- Script that integrates a modified (3-"lobes) Lorenz ODE system, & plots the trajectory using Plotly.
profiler_testing.py -- Script that does some arbitrary NumPy operations

profile_processing.py -- Contains the Python class that processes the profiling results

create_profiles.sh -- bash script that runs memory_profiler, line_profiler, and the profile processing all at once
  NOTE: Only works on UNIX
  HOW TO: run with something like ~ pythonFile=profiler_testing bash create_profiles.sh
  
profiles (.txt and .lprof) -- Contains the results of the profiling
