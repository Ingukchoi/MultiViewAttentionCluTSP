# **A Multi-View Attention-Based Encoder-Decoder Framework for Clustered Traveling Salesman Problem**

This repository is the official implementation of the paper "**A Multi-View Attention-Based Encoder-Decoder Framework for Clustered Traveling Salesman Problem**". *IEEE Robotics and Automation Letters*, 2025.

# **Quick Start**

### **Dependencies:**
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-6A0DAD?logo=pytorch&logoColor=white)](https://pytorch-geometric.readthedocs.io/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

- `Python=3.10.14`
- `torch==2.2.2`
- `torch_geometric==2.5.2`
- `numpy==1.24.3`
- `pytz==2024.1`
- `sklearn==1.4.2`

### **Training the model:**

- Run `train.py`. The current code uses the same hyperparameter settings as those described in the paper.

### **Testing the model:**

---

- Run `test.py`. You can modify the `n_node` (number of nodes) and `n_cluster`(number of clusters) parameters to evaluate the model on various datasets. It's set to use the our main model in the result folder, but you can easily switch to a model you've trained.

### **Acknowledgements:**

---

- Our code execution is based on the [POMO](https://github.com/yd-kwon/POMO). We thank them for their contribution!
