# Modal Corpus Local Notes

Raw files were downloaded from the Hugging Face mirror `datasets-CNRS/modal`, which mirrors the ORTOLANG MODAL corpus. The original project site describes the corpus as spoken-dialogue excerpts in English, French, and Italian annotated for epistemic modality.

## Local Files

- `raw/Modal-English-all.xml`
- `raw/Modal-French-all.xml`
- `raw/Modal-Italian-all.xml`
- `raw/README.md`

The Hugging Face README lists `cc-by-sa-4.0`; the original Modal/ORTOLANG pages should be checked before redistributing derived data because some project pages mention share-alike licensing details.

## XML Structure

Each language file is TEI XML with two important parts:

- `text/body`: the discourse transcript. Markers and scope portions are embedded as TEI anchors, so local discourse context can be recovered.
- `text/back`: Analec annotation layers. The important layers are:
  - `spanGrp type="AnalecUnit" n="marker"`: marker spans such as `maybe`, `I think`, interrogatives, discourse responses, etc.
  - `spanGrp type="AnalecUnit" n="scope_portion"`: text spans that make up the proposition/scope being modalized.
  - `joinGrp type="AnalecSchema" n="scope"`: groups one or more scope portions into a scope.
  - `joinGrp type="AnalecRelation" n="epistemic_relation"`: links marker units to scope schemas.
  - `fvLib n="AnalecElementProperties"`: feature structures for markers, relations, and scopes.

Useful relation-level fields include:

- `construction_function`: acceptation, qualification, check, confirmation, information, etc.
- `construction_epistemic_type`: no evidence, quotative, indirect_inferential, indirect_reportive, memory, direct_visual, etc.
- `construction_source`: speaker/source annotation such as `SS`, `OS`, `SS_OT`.
- `construction_polarity`: positive, neutral, negative.
- context metadata: language, environment, turn-taking, speaker age group, number of speakers, source file.

## Experimental Use

This is not a prompt-response benchmark. It does not give an input prompt plus an expected generated modal expression. Instead, it gives naturally produced dialogue, the observed modal marker, the scope/proposition it modifies, and linguistic annotations. Good uses for our project:

- masked-marker or marker-choice prediction from transcript context and scope;
- testing whether hidden states linearly encode polarity, evidence type, source, or function;
- comparing model modal choices against human-observed markers in discourse.

CommitmentBank is structurally different: it has `premise`, `hypothesis`, and NLI-style commitment labels. It is better for estimating whether discourse commits the speaker to a proposition, while MODAL is better for studying the surface modal expression and its annotated discourse function.

## Flattening Script

Run from `representation_learning_class_project`:

```bash
python src/parse_modal_corpus.py \
  --raw-dir data/modal_corpus/raw \
  --output data/modal_corpus/processed/modal_relations.tsv
```

The output is one row per epistemic relation, with marker text, scope text, pre-marker context, the transcript continuation from the marker through the scope, a transcript context window, marker features, scope features, and relation features.

The script preserves original corpus labels and also adds normalized relation columns for cross-language work. For example, English/French `no evidence` and Italian `no_evidence` both become `relation_construction_epistemic_type_norm = no_evidence`.

The default context window is 1000 characters on each side of the modal marker. Increase `--context-chars` for broader discourse context, or modify the parser to segment by `metadata_file` if an experiment needs entire source excerpts.
