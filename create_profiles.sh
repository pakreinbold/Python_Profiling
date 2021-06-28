# Define the python file
$pythonFile = 'profiler_testing'

# Set up some variables
mpath="./profiles/$pythonFile__memory_profile.txt"
lpath="./profiles/$pythonFile__line_profile.txt"

# Do the profiling and save to txt files
python -m memory_profiler "./scripts_to_profile/${pythonFile}.py" > $mpath
kernprof -l "./scripts_to_profile/${pythonFile}.py"
# mv "${pythonFile}.py.lprof" "./profiles/${pythonFile}.py.lprof"
python -m line_profiler "./${pythonFile}.py.lprof" > $lpath
rm "./${pythonFile}.py.lprof"

# Run the plotting script
python profile_processing.py $mpath $lpath