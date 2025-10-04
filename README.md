
# 🧠 Artificial Neural Networks (ANN) & Deep Learning Practice

This repository contains a progressive collection of TensorFlow/Keras notebooks demonstrating the **core to advanced concepts in Deep Learning** — from perceptrons and backpropagation to CNNs, RNNs, and Transfer Learning architectures like VGG and LeNet-5.

Each notebook is fully runnable, following modern TensorFlow best practices with clear explanations, comments, and training visualizations.

---

## 📚 Table of Contents

| # | Notebook | Description |
|---|-----------|-------------|
| 1 | [tensorflow_basics_1.ipynb](https://github.com/le-patrice/ANN/blob/main/tensorflow_basics_1.ipynb) | Introduction to TensorFlow tensors, operations, and computational graphs. |
| 2 | [tensorflow_basics_mnist_with_functional_api.ipynb](https://github.com/le-patrice/ANN/blob/main/tensorflow_basics_mnist_with_functional_api.ipynb) | Building your first MNIST classifier using the **Functional API**. |
| 3 | [backpropagation.ipynb](https://github.com/le-patrice/ANN/blob/main/backpropagation.ipynb) | Manual implementation of **Artificial Neural Networks (ANN)** with backpropagation and varying activation functions. |
| 4 | [dfnn_on_mnist_cifar10.ipynb](https://github.com/le-patrice/ANN/blob/main/dfnn_on_mnist_cifar10.ipynb) | Deep Feed Forward Networks (DFNN) applied to MNIST & CIFAR-10, recording accuracy vs epochs. |
| 5 | [cnn_for_multi-cass.ipynb](https://github.com/le-patrice/ANN/blob/main/cnn_for_multi-cass.ipynb) | CNN for **multi-category image classification**, demonstrating convolution + pooling layers. |
| 6 | [cnn_4_conv_layers.ipynb](https://github.com/le-patrice/ANN/blob/main/cnn_4_conv_layers.ipynb) | CNN with **4+ convolution layers** on Fashion-MNIST/CIFAR-10; includes accuracy/time comparisons on CPU vs GPU. |
| 7 | [cnn_with_batch_normalization.ipynb](https://github.com/le-patrice/ANN/blob/main/cnn_with_batch_normalization.ipynb) | Demonstrates **Batch Normalization** and **Padding** concepts in CNN design. |
| 8 | [cnn_with_regularization.ipynb](https://github.com/le-patrice/ANN/blob/main/cnn_with_regularization.ipynb) | Regularization techniques: **L1, L2, Dropout**, and **L1+Dropout** comparison for CNN generalization. |
| 9 | [cnn_with_data_augmentation.ipynb](https://github.com/le-patrice/ANN/blob/main/cnn_with_data_augmentation.ipynb) | CNN with **Data Augmentation** on CIFAR-10 (RandomFlip, Rotation, Zoom, Contrast). |
| 10 | [cifar_10_cnn_practice.ipynb](https://github.com/le-patrice/ANN/blob/main/cifar_10_cnn_practice.ipynb) | End-to-end **CIFAR-10 CNN classifier** with visualization, confusion matrix, and accuracy tracking. |
| 11 | [lenet-5.ipynb](https://github.com/le-patrice/ANN/blob/main/lenet-5.ipynb) | Implementation of the **LeNet-5** architecture on MNIST and Fashion-MNIST (Tanh & ReLU variants). |
| 12 | [vgg-16.ipynb](https://github.com/le-patrice/ANN/blob/main/vgg-16.ipynb) | Transfer Learning using **VGG-16/VGG-19** for CIFAR-10 and Fashion-MNIST classification. |
| 13 | [rnn_with_sentiment.ipynb](https://github.com/le-patrice/ANN/blob/main/rnn_with_sentiment.ipynb) | **Recurrent Neural Networks (RNN/LSTM/GRU)** for sentiment analysis on IMDB movie reviews. |

---

## 🧩 Structure Overview

```markdown

ANN/
│
├── tensorflow_basics_1.ipynb
├── tensorflow_basics_mnist_with_functional_api.ipynb
├── backpropagation.ipynb
├── dfnn_on_mnist_cifar10.ipynb
├── cnn_for_multi-cass.ipynb
├── cnn_4_conv_layers.ipynb
├── cnn_with_batch_normalization.ipynb
├── cnn_with_regularization.ipynb
├── cnn_with_data_augmentation.ipynb
├── cifar_10_cnn_practice.ipynb
├── lenet-5.ipynb
├── vgg-16.ipynb
└── rnn_with_sentiment.ipynb
```


---

## 🧠 Datasets Used

| Dataset | Description | Source |
|----------|--------------|--------|
| **MNIST** | Handwritten digits (28×28 grayscale). | `tf.keras.datasets.mnist` |
| **Fashion-MNIST** | Zalando clothing dataset. | `tf.keras.datasets.fashion_mnist` |
| **CIFAR-10** | 10-class RGB image dataset (32×32). | `tf.keras.datasets.cifar10` |
| **IMDB Reviews** | 25,000 labeled movie reviews (binary sentiment). | `tf.keras.datasets.imdb` |

---

## 🧰 Technologies & Frameworks
- Python 3.x  
- TensorFlow 2.x  
- Keras (Functional + Sequential APIs)  
- NumPy, Matplotlib, Seaborn  
- Google Colab / Kaggle Notebooks  

---

## 🖼️ Adding Images / Visuals

To include model diagrams, training curves, or sample outputs:
1. Create an `/images` folder in the repo root:

```markdown
ANN/
├── images/
│   ├── model_architecture.png
│   ├── training_curve.png
│   └── sample_predictions.png
```

2. Reference them in Markdown:
```markdown
![Model Architecture](images/model_architecture.png)
![Training Accuracy Curve](images/training_curve.png)
```

---

## 🚀 How to Run

1. Clone the repo

   ```bash
   git clone https://github.com/le-patrice/ANN.git
   cd ANN
   ```
2. Open any notebook in Jupyter, VS Code, or Google Colab.
3. Install requirements if needed:

   ```bash
   pip install tensorflow numpy matplotlib seaborn
   ```
4. Run all cells and observe outputs / plots.

---

## 📊 Expected Results (Typical Accuracies)

| Model                | Dataset       | Accuracy (Approx.) |
| -------------------- | ------------- | ------------------ |
| ANN Backprop         | MNIST         | ~97%               |
| DFNN                 | CIFAR-10      | ~65–70%            |
| CNN (4 Layers)       | Fashion-MNIST | ~90%               |
| CNN + BatchNorm      | CIFAR-10      | ~75%               |
| CNN + Regularization | Fashion-MNIST | ~91%               |
| CNN + Augmentation   | CIFAR-10      | ~78–80%            |
| LeNet-5              | MNIST         | ~98%               |
| VGG-16 Transfer      | CIFAR-10      | ~83–85%            |
| RNN (LSTM)           | IMDB          | ~87–90%            |

---

## 🧾 Notes

* All models are built **from scratch using TensorFlow/Keras**, not high-level wrappers.
* Code and structure follow a **pedagogical order** — each notebook builds upon previous concepts.
* The repo can serve as a **teaching or self-learning resource** for students mastering deep learning pipelines.

---

## 🏆 Author

**Asiimwe Patrick (le-patrice)**
AI/ML Engineer | Researcher | Mentor
📍 Kampala, Uganda
🔗 [GitHub Profile](https://github.com/le-patrice)

---

## ⭐ Contributions

Pull requests are welcome for improvements, dataset additions, or notebook cleanups.
Please ensure your contributions maintain clarity and reproducibility.

---

## 🪶 License

This project is open-source under the **MIT License**.



