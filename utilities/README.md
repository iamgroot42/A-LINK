### To generate DFW Matrix

`python generateMatrixDFW.py <model_path_prefix> <matrix_output_path>`

### Compute RPF, FPR rates using above matrix

`python ROC_precompute.py <matrix_output_path> <output_fpr_tpr> <DFW_case>`

### Compute statistics (and generate graph) using TPR, FPR

`python getStats.py <output_fpr_tpr>`