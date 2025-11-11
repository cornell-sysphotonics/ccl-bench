# CCL-Bench

![CCL-Bench logo](./assets/logo.png)


We are going to pursue a class-wide benchmarking endeavor! As ML models get larger and larger, various distributed implementations and techniques have been proposed in industry and academia. However, a distributed implementation on hardware A using communication library X may behave drastically differently from the implementation on hardware B with communication library Y. Different implementations may lead to different scaling challenges. To better understand the nuances and to gain better insight on the challenges involved, we are planning to construct a benchmark, evaluating a wide variety of models on various frameworks and hardware types. The end goal is building a benchmark, CCL-Bench, which the community could benefit from.


The search space is huge. It is impossible to explore all the possible implementation-metric combinations by brute-force exploration. However, we think we can make it scalable by defining a **general trace collection and analysis framework**. Group A could collect X traces and measure M types of metrics, and group B could collect Y traces and measure N types of metrics. Then we can apply M tools developed by group A to the Y traces by group B, and apply N tools developed by group B to the X traces by group A. As long as the trace format is universal, we can make it scalable. During this semester, every group should have a central theme due to limited time, e.g. understanding the communication in SGLang inference framework, comparing the performance of different collective communication library.

## Initialization
```
conda create --name ccl-bench python=3.10.12
conda activate ccl-bench
pip install -r requirements.txt
```

## Process FLow
1. Select a workload from `./workloads`
2. Select suitable dataset, batch size, sequence length, etc., and specify the infrastructure (execution environment) listed in `workload_card_template.yaml`.
3. Determine suitbale exeuction plan (parallelization strategy, communication backend selection, etc.) for the workload and framework selected, and specify those choices in `workload_card_template.yaml`.

4. Profile and collect traces by following the guidelines in `trace_gen`
5. Copy the workload card template, fill in the card, store it under `trace_collection/<workload_name>`, and upload the traces to [Google drive](https://drive.google.com/drive/u/0/folders/1shHsa3WvlGh9YRaX7iqYBYTLnwdDfLX6).
6. Select or define metrics (which is stored in `tools` - "Current list of metrics"). This step may happen before step 3.
7. Develop tools, and store it under `tools`
8. Calculate metrics
9. Upload metrics

There are READMEs inside each folder. Make sure you take a look at them.

## Layout
```
├── README.md
├── requirements.txt
├── workload_card_template.yaml # metadata template, should be located in trace_collection/<workload> folder
├── scripts  # scripts to execute tools for different metrics
├── tools   # main.py, and various plug-ins for different metrics
└── trace_collection # place to store temparary traces locally, which are downloaded from Google Drive.
```