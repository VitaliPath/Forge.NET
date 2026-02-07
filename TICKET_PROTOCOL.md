# 🎫 Forge Ticket Protocol

**Goal:** Treat mathematical operators and algorithmic primitives as "Feature Requests." All work must be tracked to enable automated documentation and structural analysis of the library's topology.

## 1. The Workflow Rules

1. **No Ticket, No Code:** You cannot modify `src/` without an active ticket in the external tracking system.
2. **Commit Discipline:** Every commit message must start with the Ticket ID.
* *Format:* `FORGE-0000: <Type> - <Brief Description>`
* *Example:* `FORGE-0012: Feat - Implement Softmax backward pass`


3. **The Ontology Link:** Every ticket must define the **Subsystem** (e.g., `Neural`) and **Component** (e.g., Tensor, Activation, Optimizer) it affects. This enables standard graph tooling to map the library's internal dependencies.
4. **Atomicity:** Tickets should map to a single mathematical concept or algorithmic unit.
5. **The Expert Review Standard (The Deep Dive):**
* **Novice-Ready:** The Overview must provide enough context that a developer unfamiliar with the specific math could implement it.
* **First Principles:** Explain the *Why*, not just the *What*.
* **Citations:** Must include a reference to a specific, consumable source. **Prioritize Open Access academic papers (e.g., arXiv, PMLR)** that are free for the public. The goal is a 15-minute "Deep Dive" for the developer, not a semester course.



## 2. Ticket Schema (JSON)

We map the "Expert Review" fields to our JSON structure. `Component` and `Subsystem` are root fields to allow for direct node-mapping in graph-based documentation systems.

```json
{
  "id": "FORGE-0001",
  "title": "Neural - Implement ReLU Activation",
  "overview": "Deep dive into the Vanishing Gradient problem (Sigmoid/Tanh) and how Rectified Linear Units (Nair & Hinton, 2010) solve this via constant gradients for positive inputs.",
  "mathematical_context": "f(x) = max(0, x). Gradient is 1 if x > 0, else 0.",
  "acceptance_criteria": [
    "Forward pass returns 0 for negative inputs",
    "Backward pass masks gradients where input was negative"
  ],
  "testing_scenarios": [
    "Verify inputs [-1, 0, 1] -> [0, 0, 1]",
    "Gradient check against numerical approximation"
  ],
  "citation": {
    "title": "Rectified Linear Units Improve Restricted Boltzmann Machines",
    "authors": "Vinod Nair, Geoffrey E. Hinton",
    "year": 2010,
    "url": "https://www.cs.toronto.edu/~fritz/absps/relu_icml.pdf",
    "source_type": "Open Access Paper"
  },
  "Component": "Activation",
  "Subsystem": "Neural",
  "fields": {
    "complexity": "1",
    "estimation": "30m"
  },
  "state": "In Progress",
  "created": 1768600000000
}

```

## 3. Ticket Lifecycle

While tickets are managed outside the repository, they follow a strict progression to maintain the integrity of the Knowledge Graph:

* **Open:** Backlog items awaiting prioritization.
* **In Progress:** Active development (The Hot Path).
* **Closed:** Completed work merged into the Knowledge Graph.