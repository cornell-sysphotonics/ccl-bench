#!/bin/bash
salloc --nodes 1 --qos interactive --time 02:00:00 --constraint gpu --gpus 2 --account m4999