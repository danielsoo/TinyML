# Recall-priority compression (means over fine-tuning draws; thresholds from client validation data)

| model | variant | selection | draws | size_kb | threshold_mean | fn_mean | fn_max | recall_mean | far_mean | far_max | f1_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cic_near_iid | fp32_federated | fixed 0.1 | 1 | 802.44 | 0.1 | 11 | 11 | 99.987 | 22.23 | 22.23 | 64.88 |
| cic_near_iid | fp32_federated | fixed 0.2 | 1 | 802.44 | 0.2 | 22 | 22 | 99.974 | 11.01 | 11.01 | 78.85 |
| cic_near_iid | fp32_federated | fixed 0.3 | 1 | 802.44 | 0.3 | 52 | 52 | 99.939 | 6.84 | 6.84 | 85.7 |
| cic_near_iid | fp32_federated | val recall>=0.999 | 1 | 802.44 | 0.3276 | 103 | 103 | 99.879 | 6.3 | 6.3 | 86.64 |
| cic_near_iid | fp32_federated | val recall>=0.9995 | 1 | 802.44 | 0.3147 | 67 | 67 | 99.921 | 6.58 | 6.58 | 86.16 |
| cic_near_iid | fp32_federated | val recall>=0.9999 | 1 | 802.44 | 0.2692 | 37 | 37 | 99.957 | 7.56 | 7.56 | 84.43 |
| cic_near_iid | kdclip | fixed 0.1 | 3 | 65.43 | 0.1 | 7 | 12 | 99.992 | 34.35 | 39.06 | 54.63 |
| cic_near_iid | kdclip | fixed 0.2 | 3 | 65.43 | 0.2 | 61 | 101 | 99.928 | 9.92 | 10.88 | 80.52 |
| cic_near_iid | kdclip | fixed 0.3 | 3 | 65.43 | 0.3 | 231.3 | 292 | 99.728 | 6.04 | 6.14 | 87.04 |
| cic_near_iid | kdclip | val recall>=0.999 | 3 | 65.43 | 0.2135 | 77 | 133 | 99.91 | 9.54 | 11.62 | 81.23 |
| cic_near_iid | kdclip | val recall>=0.9995 | 3 | 65.43 | 0.1979 | 44 | 55 | 99.948 | 10.43 | 11.62 | 79.8 |
| cic_near_iid | kdclip | val recall>=0.9999 | 3 | 65.43 | 0.1706 | 23.7 | 25 | 99.972 | 14.33 | 15.84 | 74.15 |
| cic_near_iid | kdclip_p30 | fixed 0.1 | 3 | 112.07 | 0.1 | 3.7 | 6 | 99.996 | 34.8 | 35.53 | 54.13 |
| cic_near_iid | kdclip_p30 | fixed 0.2 | 3 | 112.07 | 0.2 | 32.7 | 33 | 99.962 | 8.47 | 8.83 | 82.89 |
| cic_near_iid | kdclip_p30 | fixed 0.3 | 3 | 112.07 | 0.3 | 186 | 231 | 99.782 | 5.15 | 5.25 | 88.74 |
| cic_near_iid | kdclip_p30 | val recall>=0.999 | 3 | 112.07 | 0.2539 | 106 | 138 | 99.876 | 6.27 | 6.76 | 86.7 |
| cic_near_iid | kdclip_p30 | val recall>=0.9995 | 3 | 112.07 | 0.2214 | 53.7 | 73 | 99.937 | 7.63 | 8.37 | 84.33 |
| cic_near_iid | kdclip_p30 | val recall>=0.9999 | 3 | 112.07 | 0.1888 | 23.3 | 27 | 99.973 | 9.65 | 10.02 | 80.97 |
| cic_near_iid | kdclip_p0 | fixed 0.1 | 3 | 206.75 | 0.1 | 3 | 4 | 99.996 | 32.74 | 34.09 | 55.66 |
| cic_near_iid | kdclip_p0 | fixed 0.2 | 3 | 206.75 | 0.2 | 30.3 | 34 | 99.964 | 7.77 | 7.99 | 84.08 |
| cic_near_iid | kdclip_p0 | fixed 0.3 | 3 | 206.75 | 0.3 | 178.3 | 195 | 99.791 | 5.04 | 5.25 | 88.98 |
| cic_near_iid | kdclip_p0 | val recall>=0.999 | 3 | 206.75 | 0.2617 | 85.7 | 94 | 99.899 | 5.84 | 6.09 | 87.5 |
| cic_near_iid | kdclip_p0 | val recall>=0.9995 | 3 | 206.75 | 0.2292 | 50.7 | 52 | 99.941 | 6.71 | 7.03 | 85.93 |
| cic_near_iid | kdclip_p0 | val recall>=0.9999 | 3 | 206.75 | 0.1823 | 22.7 | 28 | 99.973 | 10.39 | 11.86 | 79.89 |
| cic_near_iid | kdclip_T4 | fixed 0.1 | 3 | 65.44 | 0.1 | 1.3 | 2 | 99.998 | 98.86 | 99.63 | 29.35 |
| cic_near_iid | kdclip_T4 | fixed 0.2 | 3 | 65.44 | 0.2 | 23.7 | 38 | 99.972 | 16.67 | 20.45 | 71.27 |
| cic_near_iid | kdclip_T4 | fixed 0.3 | 3 | 65.44 | 0.3 | 221.7 | 268 | 99.74 | 6.36 | 6.61 | 86.47 |
| cic_near_iid | kdclip_T4 | val recall>=0.999 | 3 | 65.44 | 0.2292 | 58.7 | 73 | 99.931 | 11.39 | 14.33 | 78.42 |
| cic_near_iid | kdclip_T4 | val recall>=0.9995 | 3 | 65.44 | 0.2188 | 39.3 | 42 | 99.954 | 13.03 | 15.14 | 76.06 |
| cic_near_iid | kdclip_T4 | val recall>=0.9999 | 3 | 65.44 | 0.1992 | 22.3 | 28 | 99.974 | 16.86 | 17.79 | 70.91 |
| cic_near_iid | kdclip_long | fixed 0.1 | 3 | 65.44 | 0.1 | 10.7 | 15 | 99.987 | 31.64 | 33.53 | 56.52 |
| cic_near_iid | kdclip_long | fixed 0.2 | 3 | 65.44 | 0.2 | 62 | 114 | 99.927 | 8.85 | 9.31 | 82.25 |
| cic_near_iid | kdclip_long | fixed 0.3 | 3 | 65.44 | 0.3 | 220 | 266 | 99.742 | 5.66 | 5.88 | 87.76 |
| cic_near_iid | kdclip_long | val recall>=0.999 | 3 | 65.44 | 0.2201 | 73.7 | 132 | 99.914 | 8.26 | 10.06 | 83.29 |
| cic_near_iid | kdclip_long | val recall>=0.9995 | 3 | 65.44 | 0.2096 | 54.3 | 74 | 99.936 | 8.66 | 10.06 | 82.59 |
| cic_near_iid | kdclip_long | val recall>=0.9999 | 3 | 65.44 | 0.1836 | 31 | 45 | 99.964 | 10.94 | 13.59 | 79.06 |
| cic_near_iid | kdclip_fed | fixed 0.1 | 3 | 65.44 | 0.1 | 7.7 | 13 | 99.991 | 34.06 | 35.43 | 54.68 |
| cic_near_iid | kdclip_fed | fixed 0.2 | 3 | 65.44 | 0.2 | 47 | 57 | 99.945 | 9.07 | 9.56 | 81.89 |
| cic_near_iid | kdclip_fed | fixed 0.3 | 3 | 65.44 | 0.3 | 220 | 233 | 99.742 | 5.87 | 6.22 | 87.38 |
| cic_near_iid | kdclip_fed | val recall>=0.999 | 3 | 65.44 | 0.2188 | 72 | 80 | 99.915 | 8.13 | 8.63 | 83.44 |
| cic_near_iid | kdclip_fed | val recall>=0.9995 | 3 | 65.44 | 0.2044 | 50.7 | 79 | 99.941 | 9.31 | 10.98 | 81.56 |
| cic_near_iid | kdclip_fed | val recall>=0.9999 | 3 | 65.44 | 0.1797 | 27 | 33 | 99.968 | 11.28 | 12.07 | 78.46 |
