# Results

Test set: 1450 images from blocks of frames never seen in training (80.5% detected by MediaPipe).

## End-to-end (every test image counts)

| model | accuracy | acc_ci95_low | acc_ci95_high | precision_macro | recall_macro | f1_macro | ece |
|---|---|---|---|---|---|---|---|
| knn_raw | 0.7503 | 0.6981 | 0.7987 | 0.8654 | 0.7503 | 0.7817 | 0.1977 |
| knn | 0.7724 | 0.7196 | 0.8233 | 0.8907 | 0.7724 | 0.8063 | 0.1956 |
| mlp_raw | 0.6262 | 0.5670 | 0.6855 | 0.7506 | 0.6262 | 0.6297 | 0.5791 |
| mlp | 0.6834 | 0.6280 | 0.7413 | 0.7928 | 0.6834 | 0.6950 | 0.4592 |
| gnn | 0.6462 | 0.5913 | 0.7054 | 0.7597 | 0.6462 | 0.6531 | 0.6827 |
| cnn | 0.0345 | 0.0129 | 0.0586 | 0.0012 | 0.0345 | 0.0023 | 0.7879 |
| mobilenet | 0.6648 | 0.6143 | 0.7121 | 0.7294 | 0.6648 | 0.6567 | 0.0549 |

## Common subset (hand detected, identical inputs for all models)

| model | accuracy | acc_ci95_low | acc_ci95_high | precision_macro | recall_macro | f1_macro | ece |
|---|---|---|---|---|---|---|---|
| knn_raw | 0.8903 | 0.8586 | 0.9183 | 0.8594 | 0.8546 | 0.8499 | 0.0452 |
| knn | 0.9177 | 0.8902 | 0.9425 | 0.8848 | 0.8809 | 0.8788 | 0.0426 |
| mlp_raw | 0.7361 | 0.6810 | 0.7899 | 0.7446 | 0.7074 | 0.6870 | 0.5190 |
| mlp | 0.8072 | 0.7577 | 0.8547 | 0.7869 | 0.7820 | 0.7570 | 0.3702 |
| gnn | 0.7609 | 0.7043 | 0.8124 | 0.7537 | 0.7303 | 0.7023 | 0.6478 |
| cnn | 0.0009 | 0.0000 | 0.0027 | 0.0000 | 0.0345 | 0.0001 | 0.8213 |
| mobilenet | 0.6341 | 0.5774 | 0.6890 | 0.6967 | 0.6597 | 0.6219 | 0.0604 |

## Size and training cost

| model | parameters | model_file_mb | train_seconds | detected_fraction |
|---|---|---|---|---|
| knn_raw | 296220 | 1.23 | 0.01 | 0.80 |
| knn | 464078 | 1.90 | 0.01 | 0.80 |
| mlp_raw | 53526 | 0.69 | 1.55 | 0.80 |
| mlp | 62298 | 0.79 | 2.23 | 0.80 |
| gnn | 97226 | 1.24 | 2.75 | 0.80 |
| cnn | 1211773 | 14.67 | 153.11 | 1.00 |
| mobilenet | 2295133 | 23.56 | 46.61 | 1.00 |

## Confidence

| model | mean_conf_correct | mean_conf_wrong | frac_wrong_below_0.9 | frac_correct_above_0.9 |
|---|---|---|---|---|
| knn_raw | 0.9508 | 0.8809 | 0.3011 | 0.8327 |
| knn | 0.9792 | 0.9303 | 0.1818 | 0.9375 |
| mlp_raw | 0.2850 | 0.5121 | 0.5683 | 0.0540 |
| mlp | 0.5074 | 0.6322 | 0.4902 | 0.0535 |
| gnn | 0.1683 | 0.5016 | 0.5439 | 0.0523 |
| cnn | 0.8207 | 0.8224 | 1.0000 | 0.0000 |
| mobilenet | 0.8156 | 0.5263 | 0.9650 | 0.5145 |

## Most frequent mistakes (end-to-end)

| model | true | predicted | count |
|---|---|---|---|
| knn_raw | A | nothing | 28 |
| knn_raw | N | nothing | 22 |
| knn_raw | P | Q | 17 |
| knn_raw | B | nothing | 16 |
| knn_raw | D | nothing | 16 |
| knn_raw | Q | nothing | 16 |
| knn_raw | E | nothing | 14 |
| knn_raw | R | nothing | 11 |
| knn_raw | M | nothing | 11 |
| knn_raw | N | M | 11 |
| knn | A | nothing | 28 |
| knn | N | nothing | 22 |
| knn | B | nothing | 16 |
| knn | D | nothing | 16 |
| knn | Q | nothing | 16 |
| knn | E | nothing | 14 |
| knn | P | Q | 12 |
| knn | C | nothing | 11 |
| knn | R | nothing | 11 |
| knn | M | nothing | 11 |
| mlp_raw | P | Q | 34 |
| mlp_raw | A | nothing | 28 |
| mlp_raw | U | K | 28 |
| mlp_raw | V | K | 25 |
| mlp_raw | N | nothing | 22 |
| mlp_raw | R | K | 22 |
| mlp_raw | T | L | 17 |
| mlp_raw | D | nothing | 16 |
| mlp_raw | Q | nothing | 16 |
| mlp_raw | B | nothing | 16 |
| mlp | U | V | 48 |
| mlp | A | nothing | 28 |
| mlp | S | E | 25 |
| mlp | R | V | 24 |
| mlp | N | nothing | 22 |
| mlp | P | Q | 21 |
| mlp | Q | nothing | 16 |
| mlp | B | nothing | 16 |
| mlp | D | nothing | 16 |
| mlp | E | nothing | 14 |
| gnn | P | Q | 40 |
| gnn | A | nothing | 28 |
| gnn | X | E | 28 |
| gnn | N | nothing | 22 |
| gnn | R | U | 21 |
| gnn | S | E | 20 |
| gnn | N | M | 19 |
| gnn | D | nothing | 16 |
| gnn | B | nothing | 16 |
| gnn | Q | nothing | 16 |
| cnn | V | nothing | 50 |
| cnn | U | nothing | 50 |
| cnn | Q | nothing | 50 |
| cnn | X | nothing | 50 |
| cnn | space | nothing | 50 |
| cnn | P | nothing | 50 |
| cnn | Y | nothing | 50 |
| cnn | del | nothing | 50 |
| cnn | W | nothing | 50 |
| cnn | S | nothing | 50 |
| mobilenet | S | X | 33 |
| mobilenet | U | R | 27 |
| mobilenet | W | K | 27 |
| mobilenet | R | U | 23 |
| mobilenet | V | K | 19 |
| mobilenet | J | Z | 17 |
| mobilenet | P | Q | 12 |
| mobilenet | W | V | 12 |
| mobilenet | M | N | 10 |
| mobilenet | F | D | 10 |

## Pairwise significance (McNemar)

| model_a | model_b | acc_a | acc_b | p_value |
|---|---|---|---|---|
| knn_raw | knn | 0.7503 | 0.7724 | 0.0003778 |
| knn_raw | mlp_raw | 0.7503 | 0.6262 | 6.096e-36 |
| knn_raw | mlp | 0.7503 | 0.6834 | 3.476e-13 |
| knn_raw | gnn | 0.7503 | 0.6462 | 6.683e-27 |
| knn_raw | cnn | 0.7503 | 0.03448 | 1.767e-310 |
| knn_raw | mobilenet | 0.7503 | 0.6648 | 3.253e-07 |
| knn | mlp_raw | 0.7724 | 0.6262 | 6.412e-46 |
| knn | mlp | 0.7724 | 0.6834 | 9.267e-27 |
| knn | gnn | 0.7724 | 0.6462 | 6.073e-38 |
| knn | cnn | 0.7724 | 0.03448 | 4.241e-320 |
| knn | mobilenet | 0.7724 | 0.6648 | 1.192e-10 |
| mlp_raw | mlp | 0.6262 | 0.6834 | 7.646e-09 |
| mlp_raw | gnn | 0.6262 | 0.6462 | 0.07932 |
| mlp_raw | cnn | 0.6262 | 0.03448 | 2.24e-256 |
| mlp_raw | mobilenet | 0.6262 | 0.6648 | 0.0249 |
| mlp | gnn | 0.6834 | 0.6462 | 0.0003313 |
| mlp | cnn | 0.6834 | 0.03448 | 2.539e-281 |
| mlp | mobilenet | 0.6834 | 0.6648 | 0.2766 |
| gnn | cnn | 0.6462 | 0.03448 | 4.313e-265 |
| gnn | mobilenet | 0.6462 | 0.6648 | 0.2865 |
| cnn | mobilenet | 0.03448 | 0.6648 | 1.444e-275 |
