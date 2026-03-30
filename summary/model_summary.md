# Model Summary

| Model | Model Type | Granularity | Features | Mean R² | Std R² | Parameters |
|---|---|---|---|---:|---:|---:|
| LSTMHUni | Single LSTM | hourly | Uni | 90.16 | 2.2 | 68608 |
| LSTMQUni | Single LSTM | quarter-hourly | Uni | 84.4 | 5.16 | 68608 |
| LSTMHT | Single LSTM | hourly | T | 90.66 | 2.08 | 69120 |
| LSTMQT | Single LSTM | quarter-hourly | T | 89.81 | 3.13 | 69120 |
| LSTMHTC | Single LSTM | hourly | TC | 91.5 | 1.65 | 71168 |
| LSTMQTC | Single LSTM | quarter-hourly | TC | 84.68 | 5.43 | 71168 |
| BiHUni | Bidirectional LSTM | hourly | Uni | 90.0 | 1.89 | 137216 |
| BiQUni | Bidirectional LSTM | quarter-hourly | Uni | 87.65 | 3.19 | 137216 |
| BiHT | Bidirectional LSTM | hourly | T | 91.38 | 1.85 | 138240 |
| BiQT | Bidirectional LSTM | quarter-hourly | T | 91.44 | 2.56 | 138240 |
| BiHTC | Bidirectional LSTM | hourly | TC | 90.98 | 1.5 | 142336 |
| BiQTC | Bidirectional LSTM | quarter-hourly | TC | 87.92 | 3.12 | 142336 |
| StackedHUni | Stacked LSTM | hourly | Uni | 90.32 | 1.81 | 131584 |
| StackedQUni | Stacked LSTM | quarter-hourly | Uni | 86.46 | 3.06 | 131584 |
| StackedHT | Stacked LSTM | hourly | T | 90.82 | 1.78 | 131584 |
| StackedQT | Stacked LSTM | quarter-hourly | T | 85.75 | 4.14 | 131584 |
| StackedHTC | Stacked LSTM | hourly | TC | 92.03 | 1.27 | 131584 |
| StackedQTC | Stacked LSTM | quarter-hourly | TC | 86.46 | 3.06 | 131584 |
