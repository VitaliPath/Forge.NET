# Ticket Title Format
`{{Module}} - {{BriefDescription}}`
* **Example:** `Neural - Implement CrossEntropyLoss`
* **Example:** `Core - Optimize Tensor Strides for Broadcasting`

---

# Overview & Deep Dive (The "Why")

* **Goal:** A comprehensive explanation of the concept, accessible to a novice.
* **Requirements:**
    * **The Problem:** What issue does this solve? (e.g., "Sigmoid saturates at the extremes, killing gradients.")
    * **The Solution:** How does this math fix it? (e.g., "ReLU provides a constant gradient of 1.0 for positive inputs.")
    * **Academic Context:** Cite the origin. (e.g., "First popularized for Deep Learning by Nair & Hinton (2010) and AlexNet (2012).")
    * **Intuition:** Provide a mental model or analogy.

# Mathematical Specification

* **Formula:** LaTeX representation of the Forward Pass.
    * $f(x) = \dots$
* **Derivative:** LaTeX representation of the Backward Pass (Gradient).
    * $\frac{\partial L}{\partial x} = \dots$
* **Constraints:** Dimension invariance, domain restrictions (e.g., $x > 0$ for Log).

# Acceptance Criteria / Verification

* **State the observable outcome.**
* *Example:* "Tensor shape must remain invariant `(B, T, C)` -> `(B, T, C)`."
* *Example:* "Must handle broadcasting if input bias is `(1, C)`."

# Validation Scenarios

* **Test Cases:** List specific numerical inputs and expected outputs.
* **Edge Cases:** Zeros, Negatives, NaNs, Dimension Mismatches.

# Developer Implementation

**Repository**
`Forge.NET`

### Computational Logic

* **The Operation:** Define the vectorization strategy.
    * *Example:* "Use `Tensor.Map` for element-wise operations to avoid allocation."

### Pitfalls & Performance

* **Memory:** Does this create new arrays or view existing memory?
* **Broadcasting:** Are there implicit shape expansions?

---

# KinetiGraph Topology (Root Fields)
*These fields define where this node lives in the Knowledge Graph.*

* **Component:** The specific file or class being modified. (e.g., `Tensor.cs`, `Relu.cs`)
* **Subsystem:** The architectural layer. (e.g., `Core`, `Neural`, `Algorithms`)

# Process Metadata (Fields Object)
*These fields track the administrative state.*

* **Module:** The Namespace/Project. (e.g., `Forge.Core`, `Forge.Neural`)
* **Complexity:** (1-5) - *1 = Trivial, 5 = Architecture Change*
* **Estimation:** (Hours/Minutes)