# Set up some variables
mpath="${pythonFile}__memory_profile.txt"
lpath="${pythonFile}__line_profile.txt"

# Do the profiling and save to txt files
python -m memory_profiler "${pythonFile}.py" > $mpath
kernprof -l "${pythonFile}.py"
python -m line_profiler "${pythonFile}.py.lprof" > $lpath

# Run the plotting script
python profile_processing.py $mpath $lpath