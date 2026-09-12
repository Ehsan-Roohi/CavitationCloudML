# Frozen cavitation robustness and archive-transfer evaluation

All metrics compare to clean native alpha>=0.20 weak labels, not expert truth. Cases 19/24/23 were exposed development cases. Training cases are excluded from the table below. Seed averaging precedes case averaging.

| Condition | Method | Attached Dice | Cloud Dice |
|---|---|---:|---:|
| clean 0 | neural | 0.971 | 0.638 |
| clean 0 | raster4 | 1.000 | 1.000 |
| clean 0 | raster8 | 0.998 | 0.944 |
| coarse 2 | neural | 0.968 | 0.636 |
| coarse 2 | raster4 | 0.970 | 0.828 |
| coarse 2 | raster8 | 0.970 | 0.828 |
| coarse 4 | neural | 0.954 | 0.588 |
| coarse 4 | raster4 | 0.943 | 0.640 |
| coarse 4 | raster8 | 0.940 | 0.478 |
| missing 0.1 | neural | 0.957 | 0.548 |
| missing 0.1 | neural_filled | 0.970 | 0.638 |
| missing 0.1 | raster4 | 0.947 | 0.919 |
| missing 0.1 | raster4_filled | 0.997 | 0.972 |
| missing 0.1 | raster8 | 0.946 | 0.894 |
| missing 0.1 | raster8_filled | 0.995 | 0.926 |
| missing 0.3 | neural | 0.913 | 0.334 |
| missing 0.3 | neural_filled | 0.968 | 0.636 |
| missing 0.3 | raster4 | 0.801 | 0.429 |
| missing 0.3 | raster4_filled | 0.989 | 0.918 |
| missing 0.3 | raster8 | 0.821 | 0.753 |
| missing 0.3 | raster8_filled | 0.988 | 0.888 |
| noise 0.02 | neural | 0.970 | 0.636 |
| noise 0.02 | raster4 | 0.990 | 0.762 |
| noise 0.02 | raster4_smooth | 0.987 | 0.913 |
| noise 0.02 | raster8 | 0.987 | 0.680 |
| noise 0.02 | raster8_smooth | 0.985 | 0.857 |
| noise 0.05 | neural | 0.969 | 0.623 |
| noise 0.05 | raster4 | 0.979 | 0.496 |
| noise 0.05 | raster4_smooth | 0.980 | 0.738 |
| noise 0.05 | raster8 | 0.974 | 0.484 |
| noise 0.05 | raster8_smooth | 0.979 | 0.676 |
| noise 0.1 | neural | 0.965 | 0.598 |
| noise 0.1 | raster4 | 0.960 | 0.020 |
| noise 0.1 | raster4_smooth | 0.972 | 0.622 |
| noise 0.1 | raster8 | 0.951 | 0.016 |
| noise 0.1 | raster8_smooth | 0.971 | 0.616 |

## Interpretation
Clean fields do not establish an advantage over direct connectivity. Neural resistance to unfiltered noise must be compared with the smoothed controls, not only an unfiltered threshold. Missing-data results include identical nearest-observed imputation for both methods. These are synthetic measurement corruptions, not a new CFD regime or experimental validation.

## Previously unused archive cases
- Case 8: attached/cloud Dice 0.979/0.861; 749 cloud reference pixels. Raster4 Dice is 1.0/1.0 in both cases.
- Case 20: attached/cloud Dice 0.976/0.314; 383 cloud reference pixels. Raster4 Dice is 1.0/1.0 in both cases.

The model was frozen before fields for cases 8 and 20 were opened; case IDs were registered in advance. These are unseen-case transfer checks within the same archive, not independently annotated truth. Case 20 exposes poor cloud transfer. Do not replace this result with a selected favorable frame.

No model retraining or threshold tuning was performed. Strong control Gaussian sigma=0.7 pixels was fixed before its evaluation, not optimized per case. Geometry is kept exact in all corruptions. Raw reference connectivity and raster inference baseline differ explicitly.