# ProMiNER: Profile-Based Medical Named Entity Reranker for BioNNE-L 2025

ProMiNER (Profile-Based Medical Named Entity Reranker) is a BioNNE-L shared-task system for linking biomedical mentions to UMLS concepts. The project is built on the NEREL-BIO dataset and its entity-linking extension [[1](#ref-1), [2](#ref-2)] and focuses on the Russian track, where the final system combines a dense retriever, dictionary-based cross-encoder pretraining, and cross-encoder reranking over compact candidate-context profiles.

Resources:

- [NEREL-BIO dataset repository](https://github.com/nerel-ds/NEREL-BIO/tree/master)
- [BioNNE-L shared task](https://github.com/nerel-ds/NEREL-BIO/tree/master/BioNNE-L_Shared_Task)
- [CodaLab competition](https://codalab.lisn.upsaclay.fr/competitions/21568#participate)

## Experiment Entry Points

The root notebooks are the primary entry points:

1. `bionnel-dense-only.ipynb` - zero-shot dense retrieval with Sentence Transformers. This is the simplest baseline and evaluates whether the pretrained embedding model can link mentions directly to vocabulary names.
2. `bionnel-hybrid-bm25.ipynb` - late fusion of dense retrieval and BM25 lexical retrieval.
3. `bionnel-hybrid-char-tfidf.ipynb` - late fusion of dense retrieval and character TF-IDF retrieval.
4. `bionnel-dense-finetuning.ipynb` - dense retriever fine-tuning on mention-to-concept positive pairs, with optional hard-negative mining.
5. `bionnel-cross-encoder-dictionary-pretrain.ipynb` - dictionary-based cross-encoder pretraining. The notebook turns vocabulary synonyms into pseudo-query/candidate pairs, prepares dense-retriever candidate pools, and trains the cross-encoder to compare mentions with compact candidate-context profiles before task-specific reranking.
6. `bionnel-cross-encoder-reranking.ipynb` - final cross-encoder reranking over candidates from the fine-tuned dense retriever. The notebook builds train/dev/test retriever caches, creates cross-encoder training pairs, evaluates model checkpoints on dev, and writes final prediction files.

The two cross-encoder notebooks also build compact candidate-context profiles. Instead of scoring a mention against a single dictionary string, each CUI is represented by a short heuristic profile assembled from the representative name and useful aliases. For example, instead of a single dictionary row:

```text
вестибулокохлеарный нерв
```

the candidate text can become:

```text
слуховой нерв; slukhovoi nerv; вестибулокохлеарный нерв; vestibuliarno-kokhlearnyi nerv; cherepnoi nerv viii; nervus vestibulocochlearis [viii]
```

Dictionary pretraining is used specifically to teach the cross-encoder how to interpret these candidate-context profiles before it sees supervised task examples. The same profile format is then reused during final reranking, so the model compares each mention with a richer, but still compact, representation of each candidate concept.

## Best Model Training Path

To reproduce the best Russian-track system, run the notebooks in this order:

1. `bionnel-dense-finetuning.ipynb` - preprocess mention and vocabulary text, then fine-tune the dense BERGAMOT retriever on task-specific mention-to-concept pairs.
2. `bionnel-cross-encoder-dictionary-pretrain.ipynb` - build candidate-context profiles and pretrain the cross-encoder on dictionary-derived pseudo-query/candidate pairs.
3. `bionnel-cross-encoder-reranking.ipynb` - fine-tune the final reranker on candidates produced by the fine-tuned retriever, initializing the cross-encoder from the dictionary-pretrained checkpoint.

This separates the two main learning problems: the retriever learns to recall strong candidate pools, while dictionary pretraining helps the cross-encoder understand the enriched candidate representation that is later used for supervised reranking.

The notebooks currently contain the hyperparameter values used to obtain the results reported below.

## Repository Contents

Supporting code lives under `lib/`:

- `lib/data/` - text normalization and vocabulary enrichment.
- `lib/retrieval/core/` - dense retrieval, shared batching helpers, late fusion primitives, and ranking metrics.
- `lib/retrieval/sparse/` - BM25 and character TF-IDF sparse indices.
- `lib/retrieval/pipelines/` - notebook-facing prediction pipelines.
- `lib/retrieval/tuning/` - dev-set evaluation and sparse-fusion grid search.
- `lib/retrieval/retriever_training/` - dense retriever training and training-pair construction.
- `lib/retrieval/reranking/` - candidate caches, cross-encoder training data, inference, model I/O, context builders, dictionary pretraining, and reranker training.
- `lib/utils/logging_utils.py` - shared logging helpers.

## Data

The BioNNE-L data used by the notebooks is included in this repository. The data comes from the [BioNNE-L shared task](https://github.com/nerel-ds/NEREL-BIO/tree/master/BioNNE-L_Shared_Task), which is hosted in the [NEREL-BIO repository](https://github.com/nerel-ds/NEREL-BIO/tree/master) [[1](#ref-1), [2](#ref-2)].

The files are arranged as follows:

```text
data/
  parquet/
    ru/
    en/
    bilingual/
  tsv/
    ru/
    en/
    bilingual/
  texts/
    ru/
    en/
  vocabular/
    bionnel_vocab_bilingual.parquet
```

The test collection is shared with references to the official [CodaLab evaluation page](https://codalab.lisn.upsaclay.fr/competitions/21568#participate) and the BioNNE-L overview paper [[3]](#ref-3).

## Results

Russian-track results:

| Configuration | Acc@1 | Acc@5 | MRR |
|---|---:|---:|---:|
| BERGAMOT baseline | 0.5200 | 0.5900 | 0.5500 |
| BERGAMOT + preprocessing | 0.5856 | 0.7481 | 0.6517 |
| Best char TF-IDF hybrid | 0.5880 | 0.7469 | 0.6511 |
| Fine-tuned BERGAMOT retriever | 0.6980 | 0.8375 | 0.7573 |
| Reranker, BCE, no dictionary pretraining | 0.6939 | 0.8166 | 0.7441 |
| Reranker, BCE, dictionary pretraining | 0.7167 | 0.8215 | 0.7608 |
| Reranker, LambdaLoss, dictionary pretraining | 0.7326 | 0.8408 | 0.7784 |

Russian-track comparison with other submitted systems. The participant results are taken from the [CodaLab leaderboard](https://codalab.lisn.upsaclay.fr/competitions/21568#results) and the BioNNE-L overview paper [[3]](#ref-3).

| Participant | Acc@1 | Acc@5 | MRR | Base model |
|---|---:|---:|---:|---|
| **This system** | **0.733** | **0.841** | **0.778** | BERGAMOT |
| BlancaPlanca | 0.716 | 0.828 | 0.761 | BERGAMOT |
| droidlyx86 | 0.710 | 0.840 | 0.760 | BERGAMOT |
| dstepakov | 0.700 | 0.760 | 0.720 | RoBERTa |
| EeyoreLee | 0.650 | 0.740 | 0.690 | SapBERT |
| AntoineI | 0.620 | 0.720 | 0.670 | bert-base-russian-upos, BioSyn |
| BERGAMOT baseline | 0.520 | 0.590 | 0.550 | BERGAMOT |

## Environment

The notebooks use PyTorch, Sentence Transformers, Transformers, Datasets, pandas, NumPy, scikit-learn, tqdm, and MLflow. Use `mlflow<3.6.0`: the notebooks are written for MLflow's file-based local tracking backend, while newer MLflow versions migrate local tracking to SQLite.

A typical workflow is to create a Python environment with GPU-enabled PyTorch, install the remaining Python dependencies, prepare the BioNNE-L data under the expected layout, and then run the notebooks in the order listed above.

Large local outputs such as MLflow runs, model checkpoints, generated predictions, and candidate caches are intentionally excluded from the public repository.

## References

This repository uses the NEREL-BIO dataset, its entity-linking extension, and the BioNNE-L test collection described in the following works:

<a id="ref-1"></a>[1] Loukachevitch N. et al. NEREL-BIO: a dataset of biomedical abstracts annotated with nested named entities // Bioinformatics. 2023. Vol. 39. No. 4. P. btad161.

<a id="ref-2"></a>[2] Loukachevitch N., Sakhovskiy A., Tutubalina E. Biomedical concept normalization over nested entities with partial UMLS terminology in Russian // Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024). 2024. P. 2383-2389.

<a id="ref-3"></a>[3] Sakhovskiy A., Loukachevitch N., Tutubalina E. Overview of the BioASQ BioNNE-L task on biomedical nested entity linking in CLEF 2025 // CLEF. 2025.
