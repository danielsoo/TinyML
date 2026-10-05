# Recall-priority compression (means over fine-tuning draws; thresholds from client validation data)

| model | variant | selection | draws | size_kb | threshold_mean | fn_mean | fn_max | recall_mean | far_mean | far_max | f1_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ton_near_iid | fp32_federated | fixed 0.1 | 1 | 720.44 | 0.1 | 2 | 2 | 99.988 | 20.85 | 20.85 | 96.99 |
| ton_near_iid | fp32_federated | fixed 0.2 | 1 | 720.44 | 0.2 | 21 | 21 | 99.87 | 5.85 | 5.85 | 99.07 |
| ton_near_iid | fp32_federated | fixed 0.3 | 1 | 720.44 | 0.3 | 61 | 61 | 99.624 | 2.76 | 2.76 | 99.4 |
| ton_near_iid | fp32_federated | val recall>=0.999 | 1 | 720.44 | 0.0212 | 2 | 2 | 99.988 | 96.76 | 96.76 | 87.42 |
| ton_near_iid | fp32_federated | val recall>=0.9995 | 1 | 720.44 | 0.0 | 2 | 2 | 99.988 | 100.0 | 100.0 | 87.05 |
| ton_near_iid | fp32_federated | val recall>=0.9999 | 1 | 720.44 | 0.0 | 0 | 0 | 100.0 | 100.0 | 100.0 | 87.06 |
| ton_near_iid | hard | fixed 0.1 | 3 | 55.18 | 0.1 | 125 | 179 | 99.229 | 4.2 | 4.75 | 98.99 |
| ton_near_iid | hard | fixed 0.2 | 3 | 55.18 | 0.2 | 139.7 | 179 | 99.139 | 3.28 | 3.42 | 99.08 |
| ton_near_iid | hard | fixed 0.3 | 3 | 55.18 | 0.3 | 182 | 233 | 98.878 | 3.01 | 3.32 | 98.99 |
| ton_near_iid | hard | val recall>=0.999 | 3 | 55.18 | 0.0404 | 16 | 29 | 99.901 | 36.44 | 54.32 | 94.91 |
| ton_near_iid | hard | val recall>=0.9995 | 3 | 55.18 | 0.0078 | 4 | 12 | 99.975 | 72.18 | 100.0 | 90.37 |
| ton_near_iid | hard | val recall>=0.9999 | 3 | 55.18 | 0.0 | 0 | 0 | 100.0 | 100.0 | 100.0 | 87.06 |
| ton_near_iid | clipqat | fixed 0.1 | 3 | 55.19 | 0.1 | 54.7 | 69 | 99.663 | 3.33 | 3.65 | 99.34 |
| ton_near_iid | clipqat | fixed 0.2 | 3 | 55.19 | 0.2 | 69.7 | 79 | 99.57 | 2.85 | 3.53 | 99.36 |
| ton_near_iid | clipqat | fixed 0.3 | 3 | 55.19 | 0.3 | 80 | 89 | 99.507 | 2.33 | 2.45 | 99.41 |
| ton_near_iid | clipqat | val recall>=0.999 | 3 | 55.19 | 0.0378 | 21.3 | 25 | 99.868 | 7.02 | 8.38 | 98.9 |
| ton_near_iid | clipqat | val recall>=0.9995 | 3 | 55.19 | 0.0104 | 8.7 | 11 | 99.947 | 11.86 | 18.75 | 98.25 |
| ton_near_iid | clipqat | val recall>=0.9999 | 3 | 55.19 | 0.0 | 0 | 0 | 100.0 | 100.0 | 100.0 | 87.06 |
| ton_near_iid | kdclip | fixed 0.1 | 3 | 55.19 | 0.1 | 0 | 0 | 100.0 | 96.65 | 96.72 | 87.44 |
| ton_near_iid | kdclip | fixed 0.2 | 3 | 55.19 | 0.2 | 18 | 19 | 99.889 | 6.96 | 7.18 | 98.92 |
| ton_near_iid | kdclip | fixed 0.3 | 3 | 55.19 | 0.3 | 69 | 74 | 99.574 | 2.74 | 2.88 | 99.38 |
| ton_near_iid | kdclip | val recall>=0.999 | 3 | 55.19 | 0.2201 | 19 | 20 | 99.883 | 5.94 | 6.41 | 99.07 |
| ton_near_iid | kdclip | val recall>=0.9995 | 3 | 55.19 | 0.1953 | 15.7 | 19 | 99.903 | 8.19 | 10.7 | 98.75 |
| ton_near_iid | kdclip | val recall>=0.9999 | 3 | 55.19 | 0.1706 | 10 | 10 | 99.938 | 11.17 | 11.37 | 98.34 |
| ton_near_iid | kdclip_fed | fixed 0.1 | 3 | 55.19 | 0.1 | 0 | 0 | 100.0 | 96.55 | 96.72 | 87.45 |
| ton_near_iid | kdclip_fed | fixed 0.2 | 3 | 55.19 | 0.2 | 19 | 19 | 99.883 | 6.33 | 6.43 | 99.01 |
| ton_near_iid | kdclip_fed | fixed 0.3 | 3 | 55.19 | 0.3 | 80.7 | 82 | 99.503 | 2.29 | 2.34 | 99.41 |
| ton_near_iid | kdclip_fed | val recall>=0.999 | 3 | 55.19 | 0.1992 | 19.7 | 21 | 99.879 | 6.45 | 7.34 | 98.99 |
| ton_near_iid | kdclip_fed | val recall>=0.9995 | 3 | 55.19 | 0.1784 | 18 | 19 | 99.889 | 8.55 | 8.94 | 98.69 |
| ton_near_iid | kdclip_fed | val recall>=0.9999 | 3 | 55.19 | 0.1589 | 7.3 | 10 | 99.955 | 12.6 | 13.32 | 98.14 |
| ton_near_iid | kdclip_p30 | fixed 0.1 | 3 | 97.74 | 0.1 | 0 | 0 | 100.0 | 96.38 | 96.85 | 87.47 |
| ton_near_iid | kdclip_p30 | fixed 0.2 | 3 | 97.74 | 0.2 | 20.7 | 25 | 99.873 | 6.31 | 6.64 | 99.01 |
| ton_near_iid | kdclip_p30 | fixed 0.3 | 3 | 97.74 | 0.3 | 73 | 76 | 99.55 | 2.53 | 2.68 | 99.4 |
| ton_near_iid | kdclip_p30 | val recall>=0.999 | 3 | 97.74 | 0.2096 | 21.3 | 25 | 99.868 | 5.79 | 6.14 | 99.08 |
| ton_near_iid | kdclip_p30 | val recall>=0.9995 | 3 | 97.74 | 0.1966 | 18.3 | 19 | 99.887 | 6.93 | 8.03 | 98.92 |
| ton_near_iid | kdclip_p30 | val recall>=0.9999 | 3 | 97.74 | 0.1667 | 11 | 11 | 99.932 | 10.72 | 11.1 | 98.4 |
