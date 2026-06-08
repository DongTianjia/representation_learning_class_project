# Epistemic Modal Profiles in Language Models

This final project of STAT 37784 - Representation Learning studies whether language models represent and express epistemic modal profiles in natural discourse. It uses the multilingual MODAL corpus.

Main paper:

- `doc/arxiv/v1/your-paper.tex`

Main data and outputs:

- `data/modal_corpus/raw/`: downloaded MODAL XML files
- `data/modal_corpus/processed/modal_relations.tsv`: flattened relation-level corpus
- `data/modal_corpus/processed/modal_experiment_dataset.tsv`: filtered experiment table
- `data/modal_corpus/experiments/`: probe and LM scoring results

Reproduce from this directory, using the parent virtual environment:

```bash
../.venv/bin/python src/parse_modal_corpus.py
../.venv/bin/python src/build_modal_dataset.py
MPLCONFIGDIR=/Users/lilydong/Desktop/epistemic_alignment/.mplcache \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
../.venv/bin/python src/run_modal_experiments.py --batch-size 8 --max-length 512
../.venv/bin/python src/run_probe_ablation.py
../.venv/bin/python src/run_modal_representation_views.py --skip-dense
../.venv/bin/python src/run_modal_representation_views.py --skip-tfidf
```

The paper distinguishes frozen representation probes from output-distribution scoring, so decoded samples are not treated as the main estimand.
