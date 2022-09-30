# 2nd place solution in the Efficiency Track
## Kaggle Feedback Prize - Predicting Effective Arguments

We followed a multi-stage approach to training the winning model, so the exact score may be hard to reproduce. I will highlight below the key elements of the solution:

MLM pre-training: notebooks/HF-pret-7.py
1st stage models: notebooks/HF-43.ipynb
2nd stage model distillation on pseudolabels: notebooks/HF-56-pseudolabels.ipynb
2nd stage model finetuning: notebooks/HF-43pseudo2.ipynb
