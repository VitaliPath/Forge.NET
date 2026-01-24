# 🎫 Forge Ticket Protocol

**Goal:** Treat mathematical operators and graph algorithms as "Feature Requests." All work must be tracked to enable automated documentation and structural analysis via KinetiGraph.

## 1. The Workflow Rules

1.  **No Ticket, No Code:** You cannot modify `src/` without an active ticket in the `tickets/in_progress/` directory.
2.  **Commit Discipline:** Every commit message must start with the Ticket ID.
    * *Format:* `FORGE-0000: <Type> - <Brief Description>`
    * *Example:* `FORGE-0012: Feat - Implement Softmax backward pass`
3.  **The Ontology Link:** Every ticket must define the **Subsystem** (e.g., `Neural`) and **Component** (e.g., `Tensor.cs`) it affects. This enables KinetiGraph to build topology maps.
4.  **Atomicity:** Tickets should map to a single mathematical concept or algorithmic unit.
5.  **The Expert Review Standard (The Deep Dive):**
    * **Novice-Ready:** The Overview must provide enough context that a developer unfamiliar with the specific math could implement it.
    * **First Principles:** Explain the *Why*, not just the *What*. (e.g., "Why do we need this Activation Function?").
    * **Citations:** Must include references to relevant academic papers or standard literature.

## 2. Ticket Schema (JSON)

We map the "Expert Review" fields to our JSON structure.
* **Note:** `Component` and `Subsystem` are root fields to allow for direct node-mapping in graph-based documentation systems.

```json
{
  "id": "FORGE-0000",
  "title": "Neural - Implement ReLU Activation",
  "overview": "Deep dive into the Vanishing Gradient problem (Sigmoid/Tanh) and how Rectified Linear Units (Nair & Hinton, 2010) solve this via constant gradients...",
  "mathematical_context": "f(x) = max(0, x). Gradient is 1 if x > 0, else 0.",
  "acceptance_criteria": [
    "Forward pass returns 0 for negative inputs",
    "Backward pass masks gradients where input was negative"
  ],
  "testing_scenarios": [
    "Verify inputs [-1, 0, 1] -> [0, 0, 1]",
    "Gradient check against numerical approximation"
  ],
  "Component": "Relu.cs",
  "Subsystem": "Neural",
  "fields": {
    "complexity": "1",
    "estimation": "30m"
  },
  "state": "In Progress",
  "created": 1768600000000
}

```

## 3. Directory Structure

* `tickets/templates/`: Stores `TICKET_TEMPLATE.md`.
* `tickets/open/`: Queue (Backlog).
* `tickets/in_progress/`: Active work (The Hot Path).
* `tickets/closed/`: Completed work (The Knowledge Graph).