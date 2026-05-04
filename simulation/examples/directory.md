Intra: ring 4 300GBps, Inter Switch

DeepSeek-v3 EP=8
=== Bandwidth Sensitivity Summary ===
    BW(GB/s)      Step(ms)     Delta(ms)    Improve(%)       Comm(%)
        1.25        3111.9           0.0         0.00%         73.9%
           5        2176.9         935.0        30.05%         62.5%
        12.5        1995.5        1116.4        35.88%         59.1%
          25        1938.0        1173.9        37.72%         57.9%
          50        1910.1        1201.8        38.62%         57.2%
         100        1896.8        1215.1        39.05%         56.9%
         200        1890.2        1221.7        39.26%         56.8%


=== Bandwidth Sensitivity Summary ===
    BW(GB/s)      Step(ms)     Delta(ms)    Improve(%)       Comm(%)
        1.25        7425.9           0.0         0.00%         40.7%
           5        6638.3         787.6        10.61%         33.6%
        12.5        6481.7         944.2        12.71%         32.0%
          25        6429.6         996.3        13.42%         31.5%
          50        6403.6        1022.3        13.77%         31.2%
         100        6390.6        1035.3        13.94%         31.0%
         200        6384.1        1041.8        14.03%         31.0%

intra_bandwidth_GBps	step_ms	comm_fraction_pct
EP=32
112.5	8144.6	34.4
225	7513.6	28.9
450	7199.3	25.8
900	7042.4	24.1
1800	6963.9	23.3
3600	6924.7	22.8


ep 8
112.5	2035.4	59.9
225	1937.6	57.9
450	1889.2	56.8
900	1865.6	56.2
1800	1854.0	55.9
3600	1848.2	55.8



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
