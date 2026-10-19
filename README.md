# RTE Client-Count Sensitivity v4 - Public Code Candidate

This candidate contains the server code bundle for the RTE client-count sensitivity v4 experiment associated with *Adaptive Aggregation for Federated LoRA via Client Update Representations*. It does not cover all experiments in the paper.

## Files and method entries

All eight Python files are byte-for-byte copies from server_source_bundle/rte_clientcount_v4/. No algorithm, residual handling, metric definition, or import was edited.

| Method | Entry option | Aggregation function |
| --- | --- | --- |
| FedEx | --agg_type ours | aggregate_models_ours |
| FSR | --agg_type fedawa_raw_global | aggregate_models_fedawa_raw_global |
| EUR | --agg_type fedawa_effective | aggregate_models_fedawa_effective |

The training entry is fed_train_glue_clientcount_v4.py. Keep both data_utils_clientcount_v4.py and data_utils.py: the entry explicitly uses the v4 partition loader, while shared modules also import the ordinary data module. Keep the Python files together.

## Recorded experiment settings

These settings are taken from the original server launch script, not inferred from CLI defaults:

- Dataset / model: RTE / RoBERTa-base.
- Dirichlet alpha: 0.1; partition seed: 42.
- Number of clients: 3, 5, or 10; training seeds: 1, 42, and 2026.
- Communication rounds: 50; local epochs: 1; local batch size: 32.
- Learning rate: 0.001; warmup ratio: 0.06.
- LoRA rank / alpha / dropout: 4 / 8 / 0.1.
- FSR/EUR temperature / strength: 1.0 / 0.5.
- Adapter priors and classifier aggregation: sample-size based.

The three supplied JSON manifests preserve the fixed client indices. They are copied unchanged into partition_manifests_rte_clientcount_v4/, the directory expected by the code relative to the working directory. Any future execution should use this candidate directory as the working directory. The K=5 and K=10 manifests record a label-swap repair protocol; its generating script is not included. Do not replace the fixed manifests with regenerated partitions.

## Environment and external resources

requirements.server_reference.txt is an unchanged copy of the server bundle's original requirements file. It is an environment reference, not a verified minimal installation specification. It does not list wandb, although the training entry imports wandb. The complete environment and missing dependency versions still require verification. No versions have been guessed and no dependencies were installed for this candidate.

GLUE data, RoBERTa-base weights/tokenizer, and compatible download access or caches are required at runtime. These resources are not bundled. The CUDA-tagged PyTorch dependency also requires an appropriate wheel source and compatible GPU environment.

## Provenance and validation limits

SOURCE_NOTES.txt is preserved verbatim, including its original bundle-relative paths and references to materials not included here. It states that the training entry, v4 data module, and aggregation module came from the experiment code snapshot; the other Python dependencies came from the server root and were not independently snapshotted for this experiment.

This candidate has not undergone independent training validation and is not claimed to have successfully reproduced any paper result. Copy validation consists of SHA-256 comparisons. No training modules were imported, no training was run, and no experimental result values were rechecked.

Only source code, fixed split indices, source notes, the reference requirements, and candidate documentation are included. No launch script, training logs, result CSVs, checkpoints, model weights, raw data, caches, virtual environments, or W&B runs are included. Original auxiliary functions remain inside the unchanged server Python files.

## Release status

This is a local candidate pending author review. Upstream licensing, copyright, and redistribution authorization remain unresolved; see LICENSE_PENDING.md. No final LICENSE is selected, and nothing has been uploaded.
