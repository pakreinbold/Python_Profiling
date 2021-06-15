# Set up some variables
mpath="./profiles/${pythonFile}__memory_profile.txt"
lpath="./profiles/${pythonFile}__line_profile.txt"

# Do the profiling and save to txt files
python -m memory_profiler "./scripts_to_profile/${pythonFile}.py" > $mpath
kernprof -l "./scripts_to_profile/${pythonFile}.py"
mv "${pythonFile}.py.lprof" "./profiles/${pythonFile}.py.lprof"
python -m line_profiler "./profiles/${pythonFile}.py.lprof" > $lpath

# Run the plotting script
python profile_processing.py $mpath $lpath