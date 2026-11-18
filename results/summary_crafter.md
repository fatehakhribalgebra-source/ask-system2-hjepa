# crafter: achievements unlocked / episode

| condition | n | mean | SE | % time in motion | tiles visited | steps survived |
|---|---|---|---|---|---|---|
| random_actions | 10 | 2.70 | 0.37 | 17 | 17 | 217 |
| pattern:levy | 10 | 2.50 | 0.50 | 58 | 41 | 199 |
| pattern:lorenz | 10 | 2.00 | 0.33 | 62 | 41 | 164 |
| pattern:fractal | 10 | 1.90 | 0.57 | 63 | 49 | 155 |
| pattern:noise_gauss | 10 | 1.90 | 0.48 | 56 | 38 | 158 |
| random_macro | 10 | 1.70 | 0.37 | 62 | 44 | 135 |
| direct:fractal | 10 | 1.60 | 0.31 | 64 | 42 | 139 |
| pattern:fractal_shuffled | 10 | 1.50 | 0.34 | 63 | 43 | 139 |
| fl_direct:fractal | 10 | 0.00 | 0.00 | 37 | 26 | 163 |
| fl_direct:fractal_shuffled | 10 | 0.00 | 0.00 | 43 | 28 | 161 |

One-way ANOVA across 10 conditions: F = 5.98, p = 1.58e-06

| comparison | diff | Tukey HSD p |
|---|---|---|
| pattern:fractal vs pattern:fractal_shuffled | +0.40 | 0.999 |
| pattern:fractal vs pattern:noise_gauss | +0.00 | 1 |
| pattern:fractal vs direct:fractal | +0.30 | 1 |
| pattern:fractal vs random_macro | +0.20 | 1 |
| pattern:fractal vs random_actions | -0.80 | 0.882 |
| pattern:lorenz vs pattern:noise_gauss | +0.10 | 1 |
| pattern:noise_gauss vs random_macro | +0.20 | 1 |
| random_macro vs random_actions | -1.00 | 0.673 |
| fl_direct:fractal vs fl_direct:fractal_shuffled | +0.00 | 1 |
| pattern:fractal vs pattern:levy | -0.60 | 0.979 |

## Crafter score (geometric mean of achievement success rates)

| condition | score % | top achievements |
|---|---|---|
| direct:fractal (seed 0) | 0.91 | collect_wood 80%, collect_sapling 40%, place_table 20%, collect_drink 20% |
| direct:fractal (seed 1) | 0.92 | collect_wood 60%, collect_sapling 60%, collect_drink 20%, place_plant 20% |
| fl_direct:fractal (seed 0) | 0.00 |  |
| fl_direct:fractal (seed 1) | 0.00 |  |
| fl_direct:fractal_shuffled (seed 0) | 0.00 |  |
| fl_direct:fractal_shuffled (seed 1) | 0.00 |  |
| pattern:fractal (seed 0) | 1.41 | collect_wood 60%, collect_sapling 60%, place_table 40%, collect_drink 40% |
| pattern:fractal (seed 1) | 0.91 | collect_wood 40%, collect_sapling 40%, collect_drink 40%, place_plant 20% |
| pattern:fractal_shuffled (seed 0) | 1.19 | collect_wood 80%, collect_drink 40%, place_table 20%, collect_sapling 20% |
| pattern:fractal_shuffled (seed 1) | 0.64 | collect_wood 60%, collect_drink 40%, collect_sapling 20% |
| pattern:levy (seed 0) | 1.51 | collect_wood 100%, collect_sapling 60%, collect_drink 60%, place_table 40% |
| pattern:levy (seed 1) | 1.03 | collect_wood 80%, place_table 40%, collect_sapling 40%, collect_drink 40% |
| pattern:lorenz (seed 0) | 1.35 | collect_wood 100%, collect_sapling 80%, place_plant 40%, place_table 20% |
| pattern:lorenz (seed 1) | 0.67 | collect_wood 60%, collect_drink 60%, collect_sapling 20% |
| pattern:noise_gauss (seed 0) | 1.30 | collect_wood 60%, place_table 40%, collect_sapling 40%, collect_drink 40% |
| pattern:noise_gauss (seed 1) | 0.97 | collect_wood 80%, collect_sapling 40%, collect_drink 40%, place_plant 20% |
| random_actions (seed 0) | 1.42 | wake_up 100%, collect_wood 80%, collect_sapling 40%, place_plant 40% |
| random_actions (seed 1) | 1.35 | wake_up 100%, collect_wood 80%, collect_sapling 40%, collect_drink 20% |
| random_macro (seed 0) | 0.94 | collect_sapling 60%, collect_wood 40%, place_plant 40%, place_table 20% |
| random_macro (seed 1) | 1.00 | collect_wood 60%, collect_sapling 40%, collect_drink 40%, place_plant 40% |