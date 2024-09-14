# Inferring Pluggable Types with Machine Learning

Deep Learning-based prediction of @Nullable nodes in Java

2hv: Convert the training data into a graph.

spectral_2hv: Cluster the training data.

GTN_alltypes: Train the GTN model.

reann_cond_pairs: Reannotate input repositories with the GTN model.

nullgcn: Train the GCN model.

cinnabar: Reannotate input repositories with the GCN model.

magreann_copy: Reannotate repositoes with the Magicoder LLM assuming the server is running on the local machine.

GTN_perc: Train the GTN model by percentage.

reann_perc: Reannotate repositories by percentage.

TDG: TDG GCN

---

Prerequisites for GTN:

OS: Tested exclusively on Ubuntu 22.04.

Graphics card: Tested on a system with a 16 GB GPU and CUDA installed.

conda create --name nullgtn python=3.8 openjdk=11.0 maven -c anaconda -c conda-forge
conda activate nullgtn
git clone https://github.com/thoriumrobot/nullgtn-artifact
cd nullgtn-artifact
pip install -r pip_new.txt
pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html