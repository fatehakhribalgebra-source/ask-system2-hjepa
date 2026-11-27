# E0: run-to-run spread (deterministic repeats, one machine)

## gd (13 tasks)

| task | rep0 | rep1 | rep2 | flips |
|---|---|---|---|---|
| 0 | x | 297 | 314 | yes |
| 1 | x | 323 | 308 | yes |
| 2 | 321 | x | 251 | yes |
| 3 | 348 | x | x | yes |
| 4 | 179 | x | 263 | yes |
| 5 | x | 304 | x | yes |
| 6 | 350 | x | 340 | yes |
| 7 | 268 | 326 | x | yes |
| 8 | x | x | 250 | yes |
| 9 | 297 | 302 | 298 |  |
| 10 | 151 | 221 | 296 |  |
| 11 | 197 | x | x | yes |
| 12 | x | x | x |  |

- solved per repeat: rep0: 8/13, rep1: 6/13, rep2: 8/13
- success 56.4 +- 5.1 % (mean +- SE over 3 repeats)
- tasks that flip between repeats: 10/13
- same seed and repeat run twice: 13/13 tasks identical (success and steps)

## rgm_paper (13 tasks)

| task | rep0 | rep1 | rep2 | flips |
|---|---|---|---|---|
| 0 | 261 | 293 | 142 |  |
| 1 | x | 172 | 208 | yes |
| 2 | 321 | 321 | 217 |  |
| 3 | x | x | x |  |
| 4 | 218 | 207 | 306 |  |
| 5 | x | 174 | 170 | yes |
| 6 | 247 | 309 | 196 |  |
| 7 | 222 | 256 | 253 |  |
| 8 | 284 | 170 | 139 |  |
| 9 | 209 | 295 | 309 |  |
| 10 | 277 | 195 | 158 |  |
| 11 | 227 | 193 | x | yes |
| 12 | 142 | x | 263 | yes |

- solved per repeat: rep0: 10/13, rep1: 11/13, rep2: 11/13
- success 82.1 +- 2.6 % (mean +- SE over 3 repeats)
- tasks that flip between repeats: 4/13

## rgm_paper_ttt (13 tasks)

| task | rep0 | rep1 | rep2 | flips |
|---|---|---|---|---|
| 0 | x | 178 | 151 | yes |
| 1 | 189 | 276 | 211 |  |
| 2 | 323 | x | x | yes |
| 3 | x | 222 | 228 | yes |
| 4 | x | 320 | 311 | yes |
| 5 | x | 139 | 180 | yes |
| 6 | 229 | 301 | 224 |  |
| 7 | 164 | 340 | 310 |  |
| 8 | 159 | 242 | x | yes |
| 9 | 169 | 291 | 241 |  |
| 10 | 125 | 148 | x | yes |
| 11 | 182 | 149 | 205 |  |
| 12 | 128 | x | 282 | yes |

- solved per repeat: rep0: 9/13, rep1: 11/13, rep2: 10/13
- success 76.9 +- 4.4 % (mean +- SE over 3 repeats)
- tasks that flip between repeats: 8/13

## Speed: gd vs rgm_paper (paired by task and repeat, 39 pairs)

- both solved: 21 pairs; median steps gd 297 vs rgm_paper 218; mean 279 vs 228
- faster in those pairs: gd 5, rgm_paper 15, ties 1 (sign test p = 0.041)
- median step difference gd - rgm_paper: +46
- time to goal with a failure counted as 350 steps: gd 312, rgm_paper 251 (paired difference +60 +- 12 SE)

## Speed: gd vs rgm_paper_ttt (paired by task and repeat, 39 pairs)

- both solved: 17 pairs; median steps gd 302 vs rgm_paper_ttt 211; mean 287 vs 218
- faster in those pairs: gd 3, rgm_paper_ttt 14, ties 0 (sign test p = 0.013)
- median step difference gd - rgm_paper_ttt: +73
- time to goal with a failure counted as 350 steps: gd 312, rgm_paper_ttt 250 (paired difference +61 +- 14 SE)

## Speed: rgm_paper vs rgm_paper_ttt (paired by task and repeat, 39 pairs)

- both solved: 26 pairs; median steps rgm_paper 224 vs rgm_paper_ttt 218; mean 232 vs 222
- faster in those pairs: rgm_paper 12, rgm_paper_ttt 14, ties 0 (sign test p = 0.845)
- median step difference rgm_paper - rgm_paper_ttt: +6
- time to goal with a failure counted as 350 steps: rgm_paper 251, rgm_paper_ttt 250 (paired difference +1 +- 15 SE)
