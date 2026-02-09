# 🎫 Forge Ticket Protocol

**Goal:** Treat mathematical operators and algorithmic primitives as "Feature Requests." All work must be tracked to enable automated documentation and structural analysis of the library's topology.

## 1. The Workflow Rules

1. **No Ticket, No Code:** You cannot modify `src/` without an active ticket in the external tracking system.
2. **Commit Discipline:** Every commit message must start with the Ticket ID and the Component name.
* **Format:** `FORGE-0000: <Component> - <Brief Description>`
* **Example:** `FORGE-0012: Activation - Implement Softmax backward pass`


3. **The Ontology Link:** Every ticket must define the **Subsystem** (e.g., `Neural`) and **Component** (e.g., `Tensor`, `Activation`, `Optimizer`) it affects. This enables standard graph tooling to map the library's internal dependencies.
4. **Atomicity:** Tickets should map to a single mathematical concept or algorithmic unit.
5. **The Expert Review Standard (The Deep Dive):**
* **Novice-Ready:** The Overview must provide enough context that a developer unfamiliar with the specific math could implement it.
* **First Principles:** Explain the *Why*, not just the *What*.
* **Citations:** Must include a reference to a specific, consumable source. **Prioritize Open Access academic papers (e.g., arXiv, PMLR)** that are free for the public.



---

## 2. Ticket Schema (JSON)

We mimic the standard ingestion structure used for enterprise ticketing systems. Technical depth, including mathematical specifications and implementation details, is stored within a single Markdown-formatted `description` string to maintain compatibility with automation tools and search indexers.

### Root JSON Properties

| Property | Description |
| --- | --- |
| `Id` | Unique identifier (e.g., `FORGE-0001`) |
| `Summary` | Ticket title following the `{{Component}} - {{BriefDescription}}` format |
| `Description` | The full Markdown body containing the Expert Review sections |
| `Component` | The logical primitive, concept, or feature being modified |
| `Subsystem` | The architectural layer (e.g., `Core`, `Neural`, `Algorithms`) |
| `Complexity` | Numeric value (1-5) representing implementation difficulty |
| `State` | The current lifecycle phase (e.g., `Open`, `In Progress`, `Closed`) |
| `Estimation` | Total time required for dev, test, and documentation |

### JSON Example

```json
{
  "Id": "FORGE-0001",
  "Summary": "Activation - Implement ReLU",
  "Description": "## Overview & Deep Dive\n\nDeep dive into the Vanishing Gradient problem (Sigmoid/Tanh) and how Rectified Linear Units (Nair & Hinton, 2010) solve this via constant gradients for positive inputs.\n\n## Mathematical Specification\n\nForward: $f(x) = \\max(0, x)$. \nBackward: $\\frac{\\partial L}{\\partial x} = 1$ if $x > 0$, else $0$.\n\n## Acceptance Criteria / How to Reproduce\n\n* [ ] Forward pass returns 0 for negative inputs\n* [ ] Backward pass masks gradients where input was negative\n\n## Developer Implementation\n\n**Repository**\n`Forge.NET`\n\n### Computational Logic\n\nUse element-wise comparison logic; ensure gradient is 0 at the discontinuity (x=0) for stability.",
  "Component": "Activation",
  "Subsystem": "Neural",
  "Complexity": "1",
  "Estimation": "30m",
  "State": "In Progress",
  "created": 1768600000000
}

```

> **Ingestion Note:** The `description` property is the primary source of technical truth. While metadata like `Component` and `Subsystem` are hoisted to root-level properties for structural analysis, the internal technical breakdown (Business Logic, Pitfalls, Criteria) must remain in the Markdown string to ensure the ticket can be ingested and rendered consistently across all management platforms.

---

## 3. Ticket Lifecycle

While tickets are managed outside the repository, they follow a strict progression to maintain the integrity of the Knowledge Graph:

* **Open:** Backlog items awaiting prioritization.
* **In Progress:** Active development (The Hot Path).
* **Closed:** Completed work merged into the Knowledge Graph.