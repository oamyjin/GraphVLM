# LLaGA

## Environment setup

```
conda create -n llaga python=3.10
conda activate llaga
cd Augmenter/LLaGA

pip3 install torch  --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

pip install -r requirement.txt
```