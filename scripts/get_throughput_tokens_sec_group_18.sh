#!/bin/bash

python ./tools/main.py --trace "./trace_collection/$1" --metric "throughput_tokens_sec"
