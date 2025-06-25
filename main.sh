#!/bin/bash

# Configuration
BATCH_SIZE=6
MAX_TICKERS=483
START_DATE="2023-08-14"
END_DATE="2023-08-30"

# Clear the positive return tickers file before starting
> data/positive_return_tickers.txt

# Read tickers from file
TICKERS=($(cat data/positive_return_tickers_v2.txt))

# Calculate number of batches needed
NUM_TICKERS=${#TICKERS[@]}
if [ $NUM_TICKERS -gt $MAX_TICKERS ]; then
    NUM_TICKERS=$MAX_TICKERS
fi

# Process tickers in batches
for ((i=0; i<$NUM_TICKERS; i+=$BATCH_SIZE)); do
    # Get batch of tickers
    BATCH=()
    for ((j=0; j<$BATCH_SIZE && i+j<$NUM_TICKERS; j++)); do
        BATCH+=(${TICKERS[i+j]})
    done
    
    echo "Processing batch $((i/BATCH_SIZE + 1)): ${BATCH[@]}"
    
    # Run Python script with current batch
    python main.py \
        --tickers "${BATCH[@]}" \
        --start-date "$START_DATE" \
        --end-date "$END_DATE" \

    
    # Optional: Add a small delay between batches
    sleep 1
done

echo "Processing complete. Results saved in data/positive_return_tickers.txt"
