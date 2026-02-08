# 🎫 Ticket Template: Forge.NET

## Ticket Summary Format

`{{Component}} - {{BriefDescription}}`

* **Example:** `Activation - Implement CrossEntropyLoss`
* **Example:** `Optimizer - Add Adam Weight Decay (AdamW)`

---

# Description (Markdown Content)

## Overview & Deep Dive (The "Why")

* **The Problem:** What conceptual or technical issue does this solve? (e.g., "The vanishing gradient problem in deep networks.")
* **The Solution:** How does this specific mathematical approach fix it?
* **Intuition:** Provide a mental model, analogy, or visualization of the operation.
* **Academic Context:** Provide a citation (Priority: Open Access/arXiv).

## Mathematical Specification

* **Forward Pass:**
* 


* **Backward Pass (Gradient):**
* 


* **Constraints:** Dimension invariance, domain restrictions, or broadcasting rules.

## Acceptance Criteria / How to Reproduce

* [ ] **Requirement:** Describe the observable outcome (e.g., "Gradient check passes with ").
* [ ] **Requirement:** Describe expected behavior for edge cases.

## Use Cases / Validation Scenarios

* **Numerical Test:** `Input: [x, y] -> Expected Output: [a, b]`
* **Edge Cases:** Handling of zeros, negatives, NaNs, or dimension mismatches.

## Developer Implementation

**Repository:** `Forge.NET`

### Computational Logic

* **The Operation:** Define the vectorization or implementation strategy (e.g., "Use SIMD-accelerated loops" or "Implement as an in-place map to reduce memory overhead").

### Pitfalls & Performance

* **Memory:** Does this create new buffers or return a view?
* **Stability:** Potential for overflow/underflow (e.g., "Use the Log-Sum-Exp trick for numerical stability").

### Existing Process Examples

* List any existing code in the repository that performs a similar process to avoid re-inventing the wheel.

### Architectural Concerns

* Does this change the dependency graph or library topology?

---

# Metadata (Root JSON Fields)

* **Component:** The logical primitive or concept (e.g., `Tensor`, `Activation`, `Optimizer`).
* **Subsystem:** The architectural layer (e.g., `Core`, `Neural`, `Algorithms`).
* **Complexity:** (1-5) *1 = Trivial, 5 = Major Architecture Change*.
* **Estimation:** Estimated time for dev, testing, and documentation.
* **State:** `Open`, `In Progress`, or `Completed`.