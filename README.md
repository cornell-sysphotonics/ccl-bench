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
3. Determine suitable exeuction plan (parallelization strategy, communication backend selection, etc.) for the workload and framework selected, and specify those choices in `workload_card_template.yaml`.
4. Profile and collect traces by following the guidelines in `trace_gen`, and store it under `trace_collection/<workload_name>`
5. Copy the workload card template, fill in the card, store it under `trace_collection/<workload_name>` (you should not upload the trace to github).
6. Determine metrics (some are suggested in `tools/README.md`). You can also do this step before step 3.
7. Develop tools, and store it under `tools`
8. Calculate metrics


There are READMEs inside each folder. Take a look at them.

## Final deliverables
1. Traces. We expect you to upload traces you collected to Canvas (we will explore other methods if the traces are too large).
2. Tools for metrics that you selected, and workload cards. We expect you to issue a pull request to the main branch containing the tools you developed, and workload metadata that is associated with the traces you collected.
3. An Excel sheet with metric results and workload metadata (the result will of course be a single number for each metric-workload pair so that it can be presented on a datasheet). We expect you to upload this to Canvas as well.
4. A project report explaining things you've done and the metrics you've collected, and also results that could not be presented on a datasheet such as graphs. The report should also contain introduction, background, discussion on the tools and metrics, execution details, evaluation on the results, and final conclusion and discussion. We expect you to upload this to Canvas.

Naming of the workload and workload card:
`<model>-<framework>-<platform>-<group_number>`

Naming of the metrics and tools for the metrics:
`<metric_name>-<group_number>`


## Layout
```
├── README.md
├── pyproject.toml                     # Root project config with uv workspace
├── requirements.txt
├── workload_card_template.yaml        # Metadata template
├── perlmutter/                        # NERSC Perlmutter shared scripts
│   ├── common.sh                      # Shared configuration and functions
│   ├── setup_env.sh                   # Environment setup script
│   ├── activate.sh                    # Environment activation
│   └── submit_all.sh                  # Submit all workloads
├── scripts/                           # Scripts to execute metric tools
├── tools/                             # main.py and metric calculation plugins
└── trace_collection/                  # Workload configs, scripts, and metadata
    ├── <model>-<framework+parallelism>-<platform>-<group>/
    │   ├── run.sh                     # Simple wrapper to submit job
    │   ├── run.sbatch                 # SLURM batch script
    │   ├── train_config.toml          # TorchTitan training config
    │   ├── workload_card.yaml         # Workload metadata
    │   ├── pyproject.toml             # uv workspace member
    │   └── README.md                  # Workload documentation
    └── ...
```

## Group 16 Workloads (TorchTitan on Perlmutter)

| Workload | Model | Parallelism |
|----------|-------|-------------|
| `llama3.1-8b-torchtitan-pp-perlmutter-16` | LLaMA-3.1-8B | PP=4 |
| `llama3.1-8b-torchtitan-tp-perlmutter-16` | LLaMA-3.1-8B | TP=4 |
| `deepseek-v2-lite-torchtitan-dp+pp-perlmutter-16` | DeepSeek-V2-Lite | DP=2, PP=2 |
| `deepseek-v2-lite-torchtitan-dp+tp-perlmutter-16` | DeepSeek-V2-Lite | DP=2, TP=2 |
| `qwen3-32b-torchtitan-3d-perlmutter-16` | Qwen3-32B | DP=2, TP=2, PP=2 |
| `qwen3-32b-torchtitan-dp+pp-perlmutter-16` | Qwen3-32B | DP=2, PP=2 |
| `qwen3-32b-torchtitan-dp+tp-perlmutter-16` | Qwen3-32B | DP=2, TP=2 |
