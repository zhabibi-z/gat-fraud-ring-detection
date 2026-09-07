# Graph Attention Networks for Fraud-Ring Detection

A Graph Attention Network implemented from scratch — no PyTorch Geometric — that separates
camouflaged fraud from legitimate accounts in a synthetic payment graph.

Runs on CPU in roughly 2–4 minutes. **The dataset is 100% synthetic** and generated inside the
notebook; nothing is downloaded, and no external or proprietary data is used.

**The punchline:** camouflaged fraud accounts look legitimate on their own features — an MLP
trained on the same node features cannot tell them apart. The ring structure is what gives them
away, and attention over neighbours is what exposes it.

## Quickstart

```bash
pip install -r requirements.txt
jupyter notebook 02_gat_fraud_ring_detection.ipynb
```

## Contents

1. **Synthetic payment graph** — accounts, transactions, and planted fraud rings with camouflage.
2. **Transductive split** — train/val/test masks over a single shared graph.
3. **GAT** — multi-head masked attention built directly on the adjacency structure.
4. **Training** — with the MLP tabular baseline trained on identical node features.
5. **Test slices** — scored separately on camouflaged vs. obvious fraud.
6. **Did it use the ring?** — attention-weight analysis checking the model relies on graph
   structure rather than node features alone.
7. **Notes** — caveats and where the approach breaks down.

## Built with

PyTorch, NetworkX, NumPy, pandas, scikit-learn, matplotlib, and seaborn.
