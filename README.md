# Retired client project Description
Take-home project: the overall structure contains two parts. For the classification model, it is located in cluster_tabular_models; for the segmentation and data inspection, it is in the main Jupyter notebook file. The configs are in the census_foundation_models.json

The folder structure should be

```
retailed_client
├── build(created by setup.py)
├── model
├── cluster_tabular_models
├── requirements
├── census-bureau.columns(not provided here)
├── census-bureau.data(not provided here)
├── clustering_analysis.ipynb
├── data_inspect.ipynb
└── setup.py
```

# Installation:
Follow step by step.
```bash
conda create -n census-cluster python=3.11
conda activate census-cluster
pip install --upgrade pip setuptools wheel
pip install -e ".[full]"
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

````
The CUDA driver is 12.4, and the GPU is RTX-A6000; the peak GPU usage is ~12GB, so you may need to change the batch size as well as change the PyTorch version if your hardware does not support CUDA 12.4.

*If you are running the code in a non-interactive environment, please activate your environment and*
To authenticate in a non-interactive environment:
  1. Open https://ux.priorlabs.ai in a browser and log in (or register)
  2. Accept the license on the Licenses tab
  3. Download the chpt from the given HuggingFace link.
  4. Put the ckpt into ~/.cache/tabpfn/


# Train
In the file cluster_tabular_models, run
```bash
source activate census-cluster
python scripts/train_census_foundation_models.py --config configs/census_foundation_models.json
```

# Eval
A Jupyter Notebook file in 
```
cluster_tabular_models/notebooks/evaluate_censusu_foundation_models.ipynb
```
# Visualization tutorial
Data inspection and segmentation (clustering) are finished in 
```bash
clustering_analysis.ipynb
```
