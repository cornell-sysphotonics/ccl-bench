Intra: ring 4 300GBps, Inter Switch

DeepSeek-v3 EP=8
=== Bandwidth Sensitivity Summary ===
    BW(GB/s)      Step(ms)     Delta(ms)    Improve(%)       Comm(%)
        1.25        4199.5           0.0         0.00%         60.3%
           5        3307.4         892.1        21.24%         49.6%
        12.5        3138.5        1061.0        25.26%         46.9%
          25        3084.3        1115.2        26.56%         45.9%
          50        3057.5        1142.0        27.19%         45.4%
         100        3044.3        1155.2        27.51%         45.2%
         200        3037.7        1161.8        27.67%         45.1%


EP=16
=== Bandwidth Sensitivity Summary ===
    BW(GB/s)      Step(ms)     Delta(ms)    Improve(%)       Comm(%)
        1.25        8527.1           0.0         0.00%         37.3%
           5        7765.0         762.1         8.94%         31.2%
        12.5        7617.4         909.7        10.67%         29.9%
          25        7568.3         958.8        11.24%         29.4%
          50        7543.9         983.2        11.53%         29.2%
         100        7531.7         995.4        11.67%         29.1%
         200        7525.6        1001.5        11.74%         29.0%

Intra bw: 300
    ccl_bench_sim_bw*_ep32
Intra bw: 450
    ccl_bench_sim_bw*_ep32_450
Intra bw: 900
    ccl_bench_sim_bw*_ep32_900


Intra: ring 32, inter Switch
    ccl_bench_sim_bw*_ep32_900_32

Intra: switch 32, inter switch
    ccl_bench_sim_bw*_ep32_900_32


Sweep scale up domain size
