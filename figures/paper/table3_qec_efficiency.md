# Table 3: QEC Data Efficiency and Generalization Results

## Data Efficiency Results
|   composition |   real_pct |   synthetic_pct |   accuracy_mean |   data_efficiency |
|--------------:|-----------:|----------------:|----------------:|------------------:|
|         100_0 |        100 |               0 |        0.994889 |           1       |
|         50_50 |         50 |              50 |        0.993333 |           2       |
|         30_70 |         30 |              70 |        0.993333 |           3.33333 |
|         10_90 |         10 |              90 |        0.993333 |          10       |

## Generalization Results
| noise_type        |   baseline_accuracy |   physics_informed_accuracy |   accuracy_difference |
|:------------------|--------------------:|----------------------------:|----------------------:|
| phase_damping     |            0.989333 |                    0.986667 |          -0.00266665  |
| depolarizing      |            0.994222 |                    0.988444 |          -0.00577778  |
| amplitude_damping |            0.988444 |                    0.988889 |           0.000444412 |
| combined          |            0.998222 |                    0.996889 |          -0.00133336  |