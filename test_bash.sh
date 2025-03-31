#!/bin/bash
python ProductMetaSourceGen.py
status=$?
timestamp=$(date +%Y%m%d-%H%M%S)
if [ -d "log" ]; then
    if [ $status -eq 0 ]; then
        mv log log-kettle-$timestamp
    else
        mv log "log-kettle-fail-$timestamp"
    fi
else
    echo "Log folder does not exist. No renaming performed."
fi

python product_meta_src_gen_repeated_sampling_sources.py
status=$?
timestamp=$(date +%Y%m%d-%H%M%S)
if [ -d "log" ]; then
    if [ $status -eq 0 ]; then
        mv log log-kettle-$timestamp-repeated-sampling-sources
    else
        mv log "log-kettle-fail-$timestamp-repeated-sampling-sources"
    fi
else
    echo "Log folder does not exist. No renaming performed."
fi