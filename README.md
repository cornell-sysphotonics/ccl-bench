# ccl-bench

## Initialization
```
conda create --name ccl-bench python=3.10.12
conda activate ccl-bench
pip install -r requirements.txt
```

## Process FLow
`#TODO: add guidelines for each step`
1. Run workload
2. Profile, generate traces
3. Store traces, attach metadata, upload to repository
4. Define metrics
5. Develop tools
6. Calculate metrics
7. Upload metrics

## Layout
```
├── README.md
├── requirements.txt
├── workload_card_template.yaml # metadata template, should be located in trace_collection/<workload> folder
├── scripts  # scripts to execute tools for different metrics
├── tools   # main.py, and various plug-ins for different metrics
└── trace_collection # place to store temparary traces locally, which are downloaded from Google Drive.
```