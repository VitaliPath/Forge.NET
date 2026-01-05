# Forge (.NET)

**A General-Purpose Differentiable Compute Graph Library.**

`Forge.NET` is a lightweight, domain-agnostic engine designed to model, execute, and optimize complex relational systems. While it provides the primitives necessary for deep learning (scalar automatic differentiation), it is architected to support any problem domain that can be represented as a directed graph of differentiable operations—from neural networks to software dependency graphs and gene regulatory networks.

---

## 📐 Organizing Principle

The core philosophy of Forge is that **Structure and Computation are inseparable.** Most machine learning libraries treat the "Graph" as a static artifact that only exists to facilitate matrix multiplication. Forge treats the Graph as a first-class citizen. It is organized into three distinct layers of abstraction:

### 1. Core (The Atoms)
The foundational unit of computation is the `Value` (or `Value<T>`).
* **Role:** Handles scalar state, gradient storage, and local backpropagation.
* **Function:** Serves as the atomic node in the differentiable graph.

### 2. Neural (The Assembly)
The structural arrangement of Atoms into standard learning patterns.
* **Role:** organizes `Value` atoms into `Neurons`, `Layers` (implementing `IModelLayer`), and `MLPs`.
* **Function:** Provides the "Feed-Forward" architecture required for standard optimization tasks.

### 3. Graph (The Topology)
The relational engine that models connections beyond simple layers.
* **Role:** Defines arbitrary `Node` and `Edge` relationships.
* **Function:** Enables graph convolution, dependency mapping, and message passing between entities that do not fit into a standard tensor grid.

---

## 🎯 Target Consumers

Because Forge adheres to a strict "Generalist Rule"—containing no domain-specific logic—it serves as the computational backbone for widely divergent applications:

* **Generative AI:**
    Used to construct character-level language models and experiment with Transformer attention mechanisms from first principles.
    
* **DevOps Intelligence:**
    Used to model software repositories as graphs, calculating the "distance" between Ticket Requirements and Source Code Files to identify stale documentation or high-risk commits.
    
* **Systems Biology:**
    Used to model Gene Regulatory Networks (GRNs), treating genes as nodes and transcription factors as edges to predict cellular state changes under perturbation.

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
The scalar automatic differentiation engine (`Forge.Core`) is heavily inspired by **Andrej Karpathy's** `micrograd`. We gratefully acknowledge his educational work in demystifying backpropagation, which serves as the reference implementation for the Core atom of this system.