#!/bin/bash

# Configuration
BATCH_SIZE=3
MAX_TICKERS=483
START_YEAR=2023
START_MONTH=6
END_MONTH=12

# Clear the positive return tickers file before starting
> data/positive_return_tickers.txt

# Read tickers from file
TICKERS=($(cat data/positive_return_tickers_v2.txt))

# Calculate number of batches needed
NUM_TICKERS=${#TICKERS[@]}
if [ $NUM_TICKERS -gt $MAX_TICKERS ]; then
    NUM_TICKERS=$MAX_TICKERS
fi

for MONTH in $(seq -w $START_MONTH $END_MONTH); do
    # Get first and last day of the month
    START_DATE="${START_YEAR}-${MONTH}-01"
    END_DATE=$(date -d "${START_YEAR}-${MONTH}-01 +1 month -1 day" +"%Y-%m-%d")

    echo "Processing month: $START_DATE to $END_DATE"

    for ((i=0; i<$NUM_TICKERS; i+=$BATCH_SIZE)); do
        BATCH=()
        for ((j=0; j<$BATCH_SIZE && i+j<$NUM_TICKERS; j++)); do
            BATCH+=(${TICKERS[i+j]})
        done

        echo "Processing batch $((i/BATCH_SIZE + 1)): ${BATCH[@]}"

        python main.py \
            --tickers "${BATCH[@]}" \
            --start-date "$START_DATE" \
            --end-date "$END_DATE"

        sleep 1
    done
done

echo "Processing complete. Results saved in data/positive_return_tickers.txt"

python src/results_analyzer.py