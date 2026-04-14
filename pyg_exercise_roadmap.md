# PyG Practice Roadmap (Beginner → Advanced)

This roadmap turns your graph learning book into a practical, hands-on PyTorch Geometric ladder.  
Each exercise has a **goal**, a **build target**, and a **one-line success criterion**.

## Stage 1 — Graph Fundamentals and Classical Baselines

### 1) Hand-build tiny `Data` objects
- Build: one undirected, one directed, one weighted graph in PyG `Data`.
- Include: `x`, `edge_index`, optional `edge_attr`, and boolean masks.
- **Success criterion:** you can print and verify all tensor shapes and run one forward pass of a trivial MLP on node features.

### 2) Build a typed mini graph with `HeteroData`
- Build: two node types and at least two edge types.
- Include: per-type features and typed connectivity.
- **Success criterion:** `HeteroData` object is valid and each node/edge type reports expected counts.

### 3) Convert adjacency ↔ edge list and test permutation invariance
- Implement conversion from adjacency matrix to `edge_index` and back.
- Permute node ordering and remap labels/features consistently.
- **Success criterion:** task labels and graph-level properties are unchanged after node index permutation.

### 4) Classical node statistics as features
- Compute: degree, clustering coefficient, eigenvector centrality, and one ego-graph motif count.
- Concatenate with original node features.
- **Success criterion:** an MLP/logistic baseline with enriched features beats raw-feature baseline on validation accuracy.

### 5) Overlap heuristics for link prediction
- Implement: Common Neighbors, Jaccard, Adamic-Adar, Resource Allocation.
- Evaluate on held-out positive/negative edges.
- **Success criterion:** you report AUC/AP and at least one heuristic clearly outperforms random scoring.

## Stage 2 — Spectral and Shallow Embeddings

### 6) Laplacian spectral embedding + clustering
- Build adjacency, degree, unnormalized + normalized Laplacian.
- Extract first nontrivial eigenvectors and run k-means in 2D embedding space.
- **Success criterion:** cluster assignments are meaningfully correlated with known node groups (or synthetic communities).

### 7) Shallow encoder-decoder embedding (matrix reconstruction)
- Learn node embeddings as trainable parameters.
- Decode edges with dot product and train with negative sampling.
- **Success criterion:** reconstruction loss drops steadily and link AUC exceeds heuristic baseline on same split.

### 8) Random-walk / DeepWalk-style embeddings
- Generate random walks and context windows.
- Train skip-gram style embedding lookup model.
- **Success criterion:** nearest-neighbor nodes in embedding space share labels/roles more often than chance.

## Stage 3 — First Message-Passing Models

### 9) First real GNN for node classification (`GCNConv` or `SAGEConv`)
- Train with masked loss (`train_mask`, `val_mask`, `test_mask`).
- Include self-loops + normalization handling.
- **Success criterion:** validation performance improves over non-graph MLP baseline.

### 10) Aggregator ablation: GCN vs SAGE vs GAT
- Compare: GCN, GraphSAGE (mean), max/sum aggregation, and GAT.
- Keep splits and optimization budget fixed.
- **Success criterion:** produce a clean comparison table (accuracy/F1 + runtime) and identify one best tradeoff.

### 11) Update-function ablation
- Add and compare: plain update, concat skip, residual, GRU/gated update, and Jumping Knowledge.
- Probe shallow vs deeper stacks.
- **Success criterion:** you can show which update style delays depth degradation on your dataset.

### 12) Oversmoothing and depth sensitivity study
- Train 2/4/8/16-layer variants on same data.
- Track embedding variance and class separation over depth.
- **Success criterion:** you provide a plot/table showing where oversmoothing starts and which architecture resists it best.

## Stage 4 — Graph-Level and Edge-Level Tasks

### 13) Graph classification with pooling strategies
- Compare global mean/add/max, attention pooling, and Set2Set-style pooling.
- Use small-graph dataset (or synthetic graphs with labels).
- **Success criterion:** pooled-graph model outperforms a simple handcrafted-feature baseline and you identify when each pooling type helps.

### 14) GNN encoder + decoder for link prediction
- Encode nodes with GCN/SAGE.
- Decode with dot product and MLP-on-pairs.
- Train with negative sampling; evaluate AUC/AP.
- **Success criterion:** GNN-based link predictor beats overlap heuristics from Exercise 5 on same split protocol.

### 15) Edge-feature and relation-aware prediction
- Add edge attributes and/or typed edges in the predictor.
- Compare plain vs edge-aware decoder.
- **Success criterion:** edge-aware model improves AP or calibration for relation prediction.

### 16) Heterogeneous / multi-relational GNN
- Build `HeteroData` with multiple node/edge types.
- Train RGCN-like or per-relation message passing model.
- **Success criterion:** typed model outperforms type-agnostic collapse baseline.

## Stage 5 — Theory, Scale, and Pretraining

### 17) WL expressiveness playground (theory-meets-code)
- Implement 1-WL color refinement.
- Compare WL, GIN-style model, and weaker GCN on challenging graph pairs.
- **Success criterion:** you produce at least one pair where GIN/WL distinguish and GCN fails (or vice versa with explanation).

### 18) Efficiency: full-batch vs neighbor sampling vs subgraph mini-batching
- Run same node task with three training regimes.
- Track memory, throughput, and performance.
- **Success criterion:** report a Pareto-style comparison showing compute/performance trade-offs.

### 19) Self-supervised pretraining + fine-tuning
- Pretext options: edge masking, feature masking, context prediction, or graph contrastive objective.
- Fine-tune on node or graph classification.
- **Success criterion:** pretrained initialization improves convergence speed and/or final metric over scratch.

### 20) Generative graph model (VGAE or autoregressive)
- Implement VGAE (recommended) or simple autoregressive edge addition.
- Evaluate reconstruction + sampled graph quality statistics.
- **Success criterion:** generated/reconstructed graphs match key structural statistics better than random baseline.

---

## Chemistry-focused fast track (recommended order)
1. Exercises 1 → 2 → 4  
2. Exercises 9 → 13 → 14  
3. Exercises 15 → 16  
4. Exercise 17  
5. Exercise 20 (molecular-graph flavored)

Use molecular descriptors and atom/bond features early so every stage stays relevant to polymer/material tasks.
