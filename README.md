# Forge (.NET)

**A General-Purpose Differentiable Compute Graph Library.**

`Forge.NET` is a lightweight, domain-agnostic engine designed to model, execute, and optimize complex relational systems. It has evolved from a scalar educational tool into a **production-grade Tensor engine**, capable of training neural networks and solving graph topology problems via vectorized operations.

---

## 📐 Organizing Principle

The core philosophy of Forge is **Data-Oriented Design (DOD).**
In the v1 architecture, we treated every number as an object (`Value`). This killed performance via Garbage Collection.
In **Forge v2**, we treat memory as contiguous blocks (`Tensor`), enabling CPU cache coherence and vectorized math.

Forge is organized into three distinct layers of abstraction:

### 1. Core (The Primitives)
The foundational unit of computation is the `Tensor`.
* **Role:** Handles N-Dimensional arrays, `Stride` manipulation, and `Storage` views.
* **Function:** Implements **Broadcasting** (Implicit expansion) and **Vectorized Backpropagation** to support batch processing efficiently.

### 2. Neural (The Modules)
The structural arrangement of Tensors into learnable blocks.
* **Role:** Organizes `Tensor` weights into `Layers` (implementing `IModule`) and `Sequential` blocks.
* **Function:** Replaces the loop-heavy `Neuron` class with matrix multiplication (`Linear` layers) and efficient activation functions (`ReLU`, `Tanh`).

### 3. Graph (The Topology)
The relational engine that models connections beyond simple layers.
* **Role:** Defines arbitrary `Node<T>` and `Edge<T>` relationships.
* **Function:** Enables graph convolution, community detection (e.g., Connected Components), and message passing between entities that do not fit into a standard tensor grid.

---

## 🎯 Target Consumers

Because Forge adheres to a strict "Generalist Rule"—containing no domain-specific logic—it serves as the computational backbone for widely divergent applications:

* **Generative AI:**
    Used to construct character-level language models and experiment with Transformer attention mechanisms (using `MatMul` and `Softmax`) from first principles.

* **DevOps Intelligence:**
    Used to build "Code Intelligence" systems that model software repositories as graphs. It employs Vector Space Models to calculate the semantic distance between Feature Requirements (Tickets) and Source Code implementation, enabling automated impact analysis and self-healing documentation.

* **Systems Biology:**
    Used to model Gene Regulatory Networks (GRNs), leveraging the Tensor engine to process single-cell expression data (scRNA-seq) as large, sparse matrices.

---

## 📦 Installation

Forge is designed to be installed as a standalone dependency to ensure a clean separation between the engine (Forge) and the application logic.

```bash
# Add as a project reference
dotnet add package Forge

# Or, if building from local source:
dotnet pack -c Release
dotnet add package Forge --source "C:\Your\Local\NuGet"

```

---

## 🛡️ License & Acknowledgments

**License:** MIT

**Acknowledgments:**

* **Andrej Karpathy:** For `micrograd`, which served as the conceptual foundation for the original scalar engine (v1).
* **PyTorch Internals:** The current Tensor engine (v2) is modeled after the `Storage` vs `View` architecture of Torch, specifically the use of **Strides** to handle broadcasting and transpositions without memory allocation.