# fourroom: % arena cells explored

| condition | n | mean | SE | % time in motion | rooms reached (of 4) |
|---|---|---|---|---|---|
| pattern:noise_gauss | 6 | 15.89 | 1.60 | 94 | 2.3 |
| v2:fractal_nolearn | 6 | 15.23 | 0.91 | 93 | 2.0 |
| v2:fractal | 6 | 14.91 | 1.51 | 93 | 2.7 |
| v2:fractal_shuffled | 6 | 14.91 | 1.71 | 93 | 2.8 |
| pattern:levy | 6 | 14.65 | 1.31 | 95 | 2.8 |
| v2:none | 6 | 14.06 | 1.40 | 93 | 2.7 |
| pattern:fractal | 6 | 13.67 | 1.06 | 95 | 2.5 |
| pattern:fractal_shuffled | 6 | 12.04 | 0.89 | 94 | 2.5 |
| random_macro | 6 | 11.00 | 1.15 | 95 | 2.3 |
| pattern:lorenz | 6 | 10.16 | 1.51 | 95 | 2.0 |
| direct:fractal_shuffled | 6 | 9.05 | 1.55 | 92 | 2.0 |
| direct:fractal | 6 | 7.42 | 1.08 | 92 | 1.5 |
| random_actions | 6 | 6.58 | 0.59 | 97 | 1.2 |

One-way ANOVA across 13 conditions: F = 5.94, p = 7.42e-07

| comparison | diff | Tukey HSD p |
|---|---|---|
| pattern:fractal vs pattern:fractal_shuffled | +1.63 | 0.999 |
| pattern:fractal vs pattern:noise_gauss | -2.21 | 0.991 |
| pattern:fractal vs direct:fractal | +6.25 | 0.0525 |
| direct:fractal vs direct:fractal_shuffled | -1.63 | 0.999 |
| pattern:fractal vs random_macro | +2.67 | 0.959 |
| pattern:fractal vs random_actions | +7.10 | 0.0138 |
| pattern:lorenz vs pattern:noise_gauss | -5.73 | 0.108 |
| pattern:noise_gauss vs random_macro | +4.88 | 0.291 |
| random_macro vs random_actions | +4.43 | 0.442 |
| v2:fractal vs v2:none | +0.85 | 1 |
| v2:fractal vs v2:fractal_shuffled | +0.00 | 1 |
| v2:fractal vs v2:fractal_nolearn | -0.33 | 1 |
| v2:fractal vs pattern:fractal | +1.24 | 1 |
| v2:none vs random_macro | +3.06 | 0.897 |
| pattern:fractal vs pattern:levy | -0.98 | 1 |