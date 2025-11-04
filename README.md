# 🧭 MACHINE LEARNING → DEEP LEARNING — COMPLETE END-TO-END PATH

> **Goal:** Build a complete understanding of machine learning and deep learning — by implementing every key concept *from scratch first*, then re-implementing it with real-world libraries.

> **Methodology:**  
> 1️⃣ Write algorithms manually (NumPy only).  
> 2️⃣ Rebuild them using correct ML/DL libraries.  
> 3️⃣ Compare, validate, and document insights.  
> 4️⃣ Move on only when results and intuition match.

---

## ⚙️ 0. SETUP

**Environment**  
- Python 3.10+  
- Create virtual environment `ml_journey`.  
- Install base packages:

```bash
pip install numpy pandas matplotlib seaborn jupyter scikit-learn torch torchvision
```

**Folder pattern**

```
XX_topic_name/
 ├── from_scratch/
 ├── library_impl/
 ├── comparison.ipynb
 └── notes.md
```

---

## 🧩 1. INTRODUCTION

**Objective:** Understand what “learning from data” actually means.

**Tasks**  
- Read `00_introduction/what_is_ml.md`.  
- Sketch pseudo-code:
  ```
  data → model → loss → optimization → update
  ```
- No code yet — just intuition and examples.

---

## 🧮 2. LINEAR REGRESSION

**Goal:** Master the “wx + b” relation and gradient descent loop.

**Tasks**  
1. **From Scratch (NumPy)**  
   - Implement linear regression with manual gradient descent.  
   - File: `linear_regression_numpy.py`.  
   - Output: plotted fitted line vs true data.  
2. **Library Implementation (scikit-learn)**  
   - Use `LinearRegression`.  
   - Compare coefficients and intercept.  
3. **Validate**  
   - Ensure both implementations give the same predictions.

---

## 🧮 3. LOGISTIC REGRESSION

**Goal:** Understand probabilistic binary classification.

**Tasks**  
1. Implement sigmoid, binary cross-entropy, and gradient updates manually (`logistic_regression_numpy.py`).  
2. Validate with `sklearn.linear_model.LogisticRegression`.  
3. Visualize decision boundary.  
4. Explain log-odds connection.

---

## ⚡ 4. PERCEPTRON

**Goal:** Transition from regression to a learnable neuron.

**Tasks**  
1. Build single-neuron perceptron manually (`perceptron_numpy.py`).  
2. Train on simple 2D datasets (AND/OR gates).  
3. Demonstrate XOR failure (linear separability).  
4. No library use here.

---

## 🧠 5. MULTI-LAYER PERCEPTRON (MLP)

**Goal:** Learn forward pass and backprop mechanics.

**Tasks**  
1. Implement 2-layer network manually (`mlp_two_layer_numpy.py`): forward, ReLU/tanh, backprop via chain rule.  
2. Validate with PyTorch (`nn.Linear`, `nn.ReLU`, `torch.optim.SGD`).  
3. Compare gradients (manual vs autograd).

---

## 🌳 6. DECISION TREE

**Goal:** Understand rule-based learning.

**Tasks**  
1. Implement a tiny decision tree manually (entropy or Gini).  
2. Compare with `sklearn.tree.DecisionTreeClassifier`.  
3. Visualize tree using `export_graphviz`.

---

## 🌲 7. RANDOM FOREST & ENSEMBLES

**Goal:** Learn ensemble averaging and variance reduction.

**Tasks**  
1. Train `RandomForestClassifier`.  
2. Explore `feature_importances_`.  
3. Optionally simulate bagging manually.

---

## ⚔️ 8. SUPPORT VECTOR MACHINE (SVM)

**Goal:** Margin maximization and kernel intuition.

**Tasks**  
1. Visualize support vectors with `sklearn.svm.SVC`.  
2. Implement linear margin loss manually on toy data.  
3. Compare linear vs RBF kernels.

---

## 📏 9. KNN & NAIVE BAYES

**Goal:** Grasp instance-based and probabilistic reasoning.

**Tasks**  
1. Implement simple `KNN` manually (distance + vote).  
2. Compare with `KNeighborsClassifier`.  
3. Implement Gaussian Naive Bayes manually; validate with `GaussianNB`.

---

## 🚀 10. GRADIENT BOOSTING & ENSEMBLES

**Goal:** Understand “learning from residuals”.

**Tasks**  
1. Conceptual loop explanation of gradient boosting.  
2. Train models using `xgboost.XGBClassifier` and `lightgbm.LGBMClassifier`.  
3. Plot feature importances and training curves.

---

## 🧩 11. CONVOLUTIONAL NEURAL NETWORKS (CNN) — BASICS

**Goal:** Learn local connectivity & weight sharing.

**Tasks**  
1. **Manual (NumPy):**  
   - Write `conv2d_from_scratch.py` (single kernel convolution).  
   - Add simple max-pooling.  
   - Visualize feature maps.  
2. **PyTorch:**  
   - Rebuild same layer with `nn.Conv2d` + `nn.MaxPool2d`.  
3. Compare outputs numerically.

---

## 🧠 12. MINI CNN MODEL (MNIST)

**Goal:** Build complete image classifier.

**Tasks**  
1. Dataset: `torchvision.datasets.MNIST`.  
2. Network: Conv → ReLU → Pool → FC.  
3. Train for 5 epochs with Adam optimizer.  
4. Plot accuracy & loss curves.  
5. Save model weights (`torch.save`).

---

## 🕒 13. RNN / LSTM / SEQUENCE MODELS

**Goal:** Handle sequential dependencies.

**Tasks**  
1. Create synthetic time-series data.  
2. Implement simple recurrent loop manually (NumPy).  
3. Train LSTM in PyTorch (`nn.LSTM` + `nn.Linear`).  
4. Compare sequence prediction outputs.

---

## 🧭 14. ATTENTION & TRANSFORMER CORE

**Goal:** Understand self-attention mechanism.

**Tasks**  
1. Implement scaled dot-product attention manually (NumPy).  
   - Compute Q, K, V, softmax(QKᵀ/√dₖ) V.  
2. Build minimal Transformer block in PyTorch using `nn.MultiheadAttention`.  
3. Compare manual and framework attention outputs.

---

## 🧰 15. TRANSFER LEARNING & FINE-TUNING

**Goal:** Apply pre-trained networks.

**Tasks**  
1. Load `torchvision.models.resnet18(pretrained=True)`.  
2. Replace final FC layer for new dataset (e.g. CIFAR-10 subset).  
3. Fine-tune last layers only.  
4. Save and evaluate model.

---

## 🎨 16. AUTOENCODERS / VAE / GAN

**Goal:** Learn representation and generative modeling.

**Tasks**  
1. Implement simple Autoencoder in PyTorch (`nn.Linear` encoder-decoder).  
2. Extend to Variational Autoencoder (VAE).  
3. Build basic GAN (Generator + Discriminator) on MNIST.  
4. Visualize reconstructions and generated samples.

---

## 🧱 17. DEPLOYMENT & MLOPS BASICS

**Goal:** Bring models to real world.

**Tasks**  
1. Save model (`torch.save` or `joblib.dump`).  
2. Build inference API using FastAPI or Flask.  
3. (Optional) Create demo interface with Streamlit.  
4. Document reproducible pipeline.

---

## 🔚 COMPLETION OUTCOME

By following all stages sequentially, you will:
- Understand every major ML/DL algorithm intuitively and mathematically.  
- Be able to implement and debug all core models.  
- Confidently transition from NumPy to sklearn → PyTorch.  
- Apply and deploy trained models in production.

---

## 🧭 Implementation Summary

| Phase | Implementation | Libraries |
|-------|----------------|------------|
| Fundamental ML | From Scratch | NumPy |
| Classic ML | Framework Validation | scikit-learn |
| Shallow Neural Nets | Scratch → Autograd | NumPy / PyTorch |
| CNN, RNN, Transformer | Practical Framework | PyTorch |
| Deployment | Production Stack | FastAPI / Streamlit |

---

## ✏️ Guideline

> **"Code it. Compare it. Understand it. Then automate it."**  
> Follow the path step by step — do not skip the manual phase.  
> Every algorithm you write once by hand becomes permanent intuition.

