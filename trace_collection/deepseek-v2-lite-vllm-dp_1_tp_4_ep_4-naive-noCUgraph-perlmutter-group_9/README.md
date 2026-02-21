Online serving of DeepSeek-V2-Lite on 4 A100 GPUs with EP=4.
naive kernel is used as AlltoAll backend.


This workload is used for exploring DP, TP effect on MOE communication, and CUDA graph is disabled.

use `python run_script` to send 10 client requests.