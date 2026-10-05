# Recall-priority compression (means over fine-tuning draws; thresholds from client validation data)

| model | variant | selection | draws | size_kb | threshold_mean | fn_mean | fn_max | recall_mean | far_mean | far_max | f1_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ton_near_iid_v2 | fp32_federated | fixed 0.1 | 1 | 732.44 | 0.1 | 1 | 1 | 99.994 | 54.57 | 54.57 | 92.49 |
| ton_near_iid_v2 | fp32_federated | fixed 0.2 | 1 | 732.44 | 0.2 | 4 | 4 | 99.975 | 7.82 | 7.82 | 98.84 |
| ton_near_iid_v2 | fp32_federated | fixed 0.3 | 1 | 732.44 | 0.3 | 19 | 19 | 99.883 | 3.61 | 3.61 | 99.41 |
| ton_near_iid_v2 | fp32_federated | val recall>=0.999 | 1 | 732.44 | 0.3165 | 23 | 23 | 99.858 | 3.51 | 3.51 | 99.41 |
| ton_near_iid_v2 | fp32_federated | val recall>=0.9995 | 1 | 732.44 | 0.2591 | 9 | 9 | 99.944 | 4.71 | 4.71 | 99.28 |
| ton_near_iid_v2 | fp32_federated | val recall>=0.9999 | 1 | 732.44 | 0.1105 | 1 | 1 | 99.994 | 23.79 | 23.79 | 96.58 |
| ton_near_iid_v2 | kdclip_fed | fixed 0.1 | 3 | 56.68 | 0.1 | 0.7 | 1 | 99.996 | 85.38 | 85.77 | 88.74 |
| ton_near_iid_v2 | kdclip_fed | fixed 0.2 | 3 | 56.68 | 0.2 | 14.3 | 17 | 99.912 | 6.31 | 6.51 | 99.03 |
| ton_near_iid_v2 | kdclip_fed | fixed 0.3 | 3 | 56.68 | 0.3 | 66 | 68 | 99.593 | 1.74 | 1.91 | 99.54 |
| ton_near_iid_v2 | kdclip_fed | val recall>=0.999 | 3 | 56.68 | 0.2435 | 30 | 32 | 99.815 | 3.68 | 3.82 | 99.36 |
| ton_near_iid_v2 | kdclip_fed | val recall>=0.9995 | 3 | 56.68 | 0.2148 | 17.7 | 20 | 99.891 | 5.61 | 5.99 | 99.12 |
| ton_near_iid_v2 | kdclip_fed | val recall>=0.9999 | 3 | 56.68 | 0.2031 | 12 | 14 | 99.926 | 6.61 | 6.82 | 98.99 |
| ton_near_iid_v2 | kdclip_fed_fix | fixed 0.1 | 3 | 56.69 | 0.1 | 0 | 0 | 100.0 | 91.96 | 95.46 | 87.98 |
| ton_near_iid_v2 | kdclip_fed_fix | fixed 0.2 | 3 | 56.69 | 0.2 | 13.3 | 15 | 99.918 | 6.27 | 6.53 | 99.04 |
| ton_near_iid_v2 | kdclip_fed_fix | fixed 0.3 | 3 | 56.69 | 0.3 | 54.3 | 58 | 99.665 | 2.12 | 2.24 | 99.52 |
| ton_near_iid_v2 | kdclip_fed_fix | val recall>=0.999 | 3 | 56.69 | 0.2526 | 30 | 33 | 99.815 | 3.63 | 3.88 | 99.37 |
| ton_near_iid_v2 | kdclip_fed_fix | val recall>=0.9995 | 3 | 56.69 | 0.2227 | 20.7 | 22 | 99.873 | 4.64 | 4.98 | 99.25 |
| ton_near_iid_v2 | kdclip_fed_fix | val recall>=0.9999 | 3 | 56.69 | 0.1927 | 8.3 | 10 | 99.949 | 7.12 | 8.01 | 98.93 |
| ton_near_iid_v2 | kdclip_fed_fixmax | fixed 0.1 | 3 | 56.69 | 0.1 | 0.7 | 1 | 99.996 | 85.47 | 85.71 | 88.72 |
| ton_near_iid_v2 | kdclip_fed_fixmax | fixed 0.2 | 3 | 56.69 | 0.2 | 12 | 14 | 99.926 | 6.49 | 6.82 | 99.01 |
| ton_near_iid_v2 | kdclip_fed_fixmax | fixed 0.3 | 3 | 56.69 | 0.3 | 64 | 66 | 99.605 | 1.74 | 1.91 | 99.54 |
| ton_near_iid_v2 | kdclip_fed_fixmax | val recall>=0.999 | 3 | 56.69 | 0.2461 | 28.3 | 31 | 99.825 | 3.97 | 4.63 | 99.33 |
| ton_near_iid_v2 | kdclip_fed_fixmax | val recall>=0.9995 | 3 | 56.69 | 0.2174 | 18.3 | 21 | 99.887 | 5.25 | 5.81 | 99.17 |
| ton_near_iid_v2 | kdclip_fed_fixmax | val recall>=0.9999 | 3 | 56.69 | 0.2005 | 11 | 16 | 99.932 | 6.6 | 7.03 | 98.99 |
| ton_near_iid_v2 | kdclip_fed_cle | fixed 0.1 | 3 | 56.69 | 0.1 | 1 | 1 | 99.994 | 85.0 | 85.63 | 88.78 |
| ton_near_iid_v2 | kdclip_fed_cle | fixed 0.2 | 3 | 56.69 | 0.2 | 19.7 | 23 | 99.879 | 5.75 | 5.91 | 99.09 |
| ton_near_iid_v2 | kdclip_fed_cle | fixed 0.3 | 3 | 56.69 | 0.3 | 66.3 | 76 | 99.591 | 1.89 | 1.99 | 99.52 |
| ton_near_iid_v2 | kdclip_fed_cle | val recall>=0.999 | 3 | 56.69 | 0.2227 | 24.3 | 28 | 99.85 | 4.93 | 5.35 | 99.2 |
| ton_near_iid_v2 | kdclip_fed_cle | val recall>=0.9995 | 3 | 56.69 | 0.2083 | 18.3 | 21 | 99.887 | 6.02 | 6.72 | 99.06 |
| ton_near_iid_v2 | kdclip_fed_cle | val recall>=0.9999 | 3 | 56.69 | 0.1901 | 12.7 | 17 | 99.922 | 7.42 | 7.9 | 98.87 |
| ton_near_iid_v2 | kdclip_fed_fix_cle | fixed 0.1 | 3 | 56.69 | 0.1 | 1 | 1 | 99.994 | 85.27 | 86.04 | 88.75 |
| ton_near_iid_v2 | kdclip_fed_fix_cle | fixed 0.2 | 3 | 56.69 | 0.2 | 17 | 19 | 99.895 | 6.44 | 6.66 | 99.0 |
| ton_near_iid_v2 | kdclip_fed_fix_cle | fixed 0.3 | 3 | 56.69 | 0.3 | 57 | 60 | 99.648 | 2.34 | 2.57 | 99.48 |
| ton_near_iid_v2 | kdclip_fed_fix_cle | val recall>=0.999 | 3 | 56.69 | 0.2188 | 22.7 | 25 | 99.86 | 5.05 | 6.58 | 99.19 |
| ton_near_iid_v2 | kdclip_fed_fix_cle | val recall>=0.9995 | 3 | 56.69 | 0.2005 | 17.7 | 21 | 99.891 | 6.26 | 7.24 | 99.02 |
| ton_near_iid_v2 | kdclip_fed_fix_cle | val recall>=0.9999 | 3 | 56.69 | 0.1914 | 14.3 | 17 | 99.912 | 7.3 | 7.47 | 98.88 |
