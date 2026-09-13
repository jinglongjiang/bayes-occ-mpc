# Selected-plan risk contribution audit

Potential relevance threshold passed; this is not a decision-benefit or novelty result.

All 600 recorded Bayes layouts were replayed with original seeds and separate warm starts. Every first command and executed robot position matched within 1e-9. No new navigation outcomes were generated.

| Configuration | Control states | Geometric opportunity states | Potential binding states | Fraction of all states |
|---|---:|---:|---:|---:|
| All | 27978 | 12526 | 4992 | 17.84% |
| baseline_circle | 3927 | 662 | 272 | 6.93% |
| baseline_square | 3897 | 1018 | 300 | 7.70% |
| dense_circle | 4622 | 2065 | 1130 | 24.45% |
| dense_square | 5145 | 3470 | 1448 | 28.14% |
| large_circle | 5923 | 2966 | 1211 | 20.45% |
| large_square | 4464 | 2345 | 631 | 14.14% |

Binding here means >=1 percentage point maximum per-step collision-probability reduction after removing ALL geometrically eligible tracks from the selected plan. This is an optimistic diagnostic, not a valid posterior update, an actual constraint switch, or demonstrated navigation benefit.

7 versus 15 point quadrature classification disagreements: 432/27978.

Sensitivity thresholds 0.5 and 2 percentage points, all denominators, and descriptive seed-block intervals are retained in summary.json and interpretation.json. No efficacy significance test or additional sample selection is performed.
