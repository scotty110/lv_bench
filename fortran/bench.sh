#!/bin/bash

# Number of runs
RUNS=2000

# File to store execution times
TIME_FILE="execution_times.txt"

# Remove previous time file if it exists
rm -f $TIME_FILE

echo "Starting benchmark at $(date)"
echo "Running Fortran program $RUNS times..."

# Run the program multiple times and measure execution time
for ((i=1; i<=$RUNS; i++)); do
    # Get timing information using time command
    TIME_OUTPUT=$( { /usr/bin/time -p ./main >/dev/null; } 2>&1 )
    
    # Extract the real time value
    EXECUTION_TIME=$(echo "$TIME_OUTPUT" | grep real | awk '{print $2}')
    
    # Store the time
    echo $EXECUTION_TIME >> $TIME_FILE
done

# Calculate statistics using awk
echo "All $RUNS runs completed at $(date)"

# Calculate min, max, avg from the time file
STATS=$(awk '
BEGIN { min = 999999; max = 0; sum = 0; count = 0; }
{
    sum += $1;
    count++;
    if ($1 < min) min = $1;
    if ($1 > max) max = $1;
}
END {
    printf "Min: %.6f s\nMax: %.6f s\nAvg: %.6f s\nSamples: %d", min, max, sum/count, count;
}' $TIME_FILE)

echo -e "\nExecution Time Statistics:"
echo "$STATS"
echo "Individual run times saved in $TIME_FILE"
rm -f $TIME_FILE
