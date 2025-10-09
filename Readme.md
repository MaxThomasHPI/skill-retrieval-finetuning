# skill-retrieval-finetuning

This repository contains data preparation, training, and evaluation code for skill-to-skill and query-to-skill retrieval experiments built around ESCO and GRETA datasets. The project focuses on building dense retrieval models (embedding-based) and experimenting with hard-negative mining, dataset balancing and Multiple Negatives Ranking losses to improve retrieval quality for skill annotation and recommendation.

## Contents

- `data/` — datasets and intermediate artifacts (JSON/JSONL). Contains combined ESCO files, augmented data, and `eval_split`/`train_dataset*.jsonl` used for training and evaluation.
- `scripts/` — helper scripts for data preparation, evaluation, and training launcher notebooks. Subfolders include `preperation/`, `training/`, and `evaluation/`.
- `models/` — output and checkpoints for finetuned encoders (e.g. `finetuned_e5_esco_model_*`).
- `requirements.txt` — Python dependencies used across scripts and notebooks.

## Quick overview

High-level flow:

1. Prepare and clean the datasets in `data/` (combine sources, validate, augment and balance).
2. Generate train / eval JSONL files with positives and mined negatives and optionally synthetic samples.
3. Train a sentence/embedding model using MultipleNegativesRankingLoss with cached in-batch negatives plus curated hard negatives.
4. Evaluate on held-out `eval_split` with ranking metrics (accuracy@K, precision@K, recall@K, NDCG, MRR, MAP) and measure query latency.

## Data preparation

Notes on the latest training data preparation (isy-finetune-v2):

- Start from about 650 human-validated ESCO samples
- Used `scripts/preperation/reduceOverfittingRisk.py` to remove some samples that were related to very frequent esco skills, keeping only 12 most diverse samples per skill. Resulting in 568 samples.
- Used `scripts/preperation/compress_long_queries.py` to shorten very long queries (over 320 chars) by using LLMs to either summarize the query or split the sample into multiple shorter samples. Resulting in 923 samples.
- Used `scripts/preperation/generateForRareLables.py` to generate additional synthetic samples for rare labels (labels with less than 5 samples) and skills not yet covered to improve balance across ESCO categories and overall coverage. Resulting in 6883 samples.
- Used `scripts/preperation/enrich_skills.py` to add skill descriptions to the label text for richer targets and better balance between query and lable length.
- Used `scripts/preperation/create_evaluation_dataset.py` to create an evaluation split of 529 samples covering all ESCO categories.
- Used `hard_negative_mining.py` to mine hard negatives from the full ESCO skill set using a base encoder (e5-base) adding up to 8 additinal positive aware hard negatives per sample. Hard negatives were filtered to remove possible false negatives (i.e. samples that are too similar to the positive labels).

## Training

Primary training entrypoint:

- `scripts/training/kaggle-esco-finetuning-mnr.ipynb` — notebook used for the most recent finetune runs. It demonstrates dataset loading, caching strategy for negatives, and training loops.

Training strategy summary:

- Base model: `e5-base` embeddings (or similar dense encoder) were used as the starting point.
- Loss: CachedMultipleNegativesRankingLoss — combines in-batch negatives with a curated cache of hard negatives to stabilize and strengthen the model against difficult confounders.
- Data handling: random shuffling every epoch, label entries concatenated with label + description for richer targets, and a mixture of human-validated and synthetic samples to improve generalization.
- Practical details: recent runs used Kaggle with 2x T4 GPUs and took about 1.2 hours for the larger balanced dataset. Hyperparameters (batch size, LR, epochs) are recorded in the notebook; check the training cell metadata for exact values.

Assumptions:

- The repository currently stores trained checkpoints in `models/` and expects training notebooks to reference paths under `data/` for datasets. Models and datasets are currently not tracked in this repo.

## Evaluation

Evaluation is performed on held-out `data/eval_split/*` JSONL files containing 529 samples evenly sampled to cover all ESCO categories. Evaluation uses same ranking metrics from SentenceTransformers InformationRetrievalEvaluator with extended support to use a ChromaDB with all ESCO skills instead of the eval_dataset corpus only.

Summary of recent evaluation (high level):

<table id="T_571d6">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_571d6_level0_col0" class="col_heading level0 col0" >Model</th>
      <th id="T_571d6_level0_col1" class="col_heading level0 col1" >accuracy@1</th>
      <th id="T_571d6_level0_col2" class="col_heading level0 col2" >accuracy@3</th>
      <th id="T_571d6_level0_col3" class="col_heading level0 col3" >accuracy@5</th>
      <th id="T_571d6_level0_col4" class="col_heading level0 col4" >accuracy@10</th>
      <th id="T_571d6_level0_col5" class="col_heading level0 col5" >precision@1</th>
      <th id="T_571d6_level0_col6" class="col_heading level0 col6" >precision@3</th>
      <th id="T_571d6_level0_col7" class="col_heading level0 col7" >precision@5</th>
      <th id="T_571d6_level0_col8" class="col_heading level0 col8" >precision@10</th>
      <th id="T_571d6_level0_col9" class="col_heading level0 col9" >recall@1</th>
      <th id="T_571d6_level0_col10" class="col_heading level0 col10" >recall@3</th>
      <th id="T_571d6_level0_col11" class="col_heading level0 col11" >recall@5</th>
      <th id="T_571d6_level0_col12" class="col_heading level0 col12" >recall@10</th>
      <th id="T_571d6_level0_col13" class="col_heading level0 col13" >ndcg@10</th>
      <th id="T_571d6_level0_col14" class="col_heading level0 col14" >mrr@10</th>
      <th id="T_571d6_level0_col15" class="col_heading level0 col15" >map@100</th>
      <th id="T_571d6_level0_col16" class="col_heading level0 col16" >avg_time_per_query</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_571d6_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_571d6_row0_col0" class="data row0 col0" >e5-base</td>
      <td id="T_571d6_row0_col1" class="data row0 col1" >0.452652</td>
      <td id="T_571d6_row0_col2" class="data row0 col2" >0.592803</td>
      <td id="T_571d6_row0_col3" class="data row0 col3" >0.628788</td>
      <td id="T_571d6_row0_col4" class="data row0 col4" >0.715909</td>
      <td id="T_571d6_row0_col5" class="data row0 col5" >0.452652</td>
      <td id="T_571d6_row0_col6" class="data row0 col6" >0.216540</td>
      <td id="T_571d6_row0_col7" class="data row0 col7" >0.145455</td>
      <td id="T_571d6_row0_col8" class="data row0 col8" >0.084280</td>
      <td id="T_571d6_row0_col9" class="data row0 col9" >0.375721</td>
      <td id="T_571d6_row0_col10" class="data row0 col10" >0.516204</td>
      <td id="T_571d6_row0_col11" class="data row0 col11" >0.559749</td>
      <td id="T_571d6_row0_col12" class="data row0 col12" >0.640492</td>
      <td id="T_571d6_row0_col13" class="data row0 col13" >0.529609</td>
      <td id="T_571d6_row0_col14" class="data row0 col14" >0.534529</td>
      <td id="T_571d6_row0_col15" class="data row0 col15" >0.481411</td>
      <td id="T_571d6_row0_col16" class="data row0 col16" >0.109027</td>
    </tr>
    <tr>
      <th id="T_571d6_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_571d6_row1_col0" class="data row1 col0" >isy-finetuned</td>
      <td id="T_571d6_row1_col1" class="data row1 col1" >0.571970</td>
      <td id="T_571d6_row1_col2" class="data row1 col2" >0.774621</td>
      <td id="T_571d6_row1_col3" class="data row1 col3" >0.827652</td>
      <td id="T_571d6_row1_col4" class="data row1 col4" >0.878788</td>
      <td id="T_571d6_row1_col5" class="data row1 col5" >0.571970</td>
      <td id="T_571d6_row1_col6" class="data row1 col6" >0.293561</td>
      <td id="T_571d6_row1_col7" class="data row1 col7" >0.197348</td>
      <td id="T_571d6_row1_col8" class="data row1 col8" >0.112121</td>
      <td id="T_571d6_row1_col9" class="data row1 col9" >0.481526</td>
      <td id="T_571d6_row1_col10" class="data row1 col10" >0.692492</td>
      <td id="T_571d6_row1_col11" class="data row1 col11" >0.751755</td>
      <td id="T_571d6_row1_col12" class="data row1 col12" >0.821258</td>
      <td id="T_571d6_row1_col13" class="data row1 col13" >0.690687</td>
      <td id="T_571d6_row1_col14" class="data row1 col14" >0.682371</td>
      <td id="T_571d6_row1_col15" class="data row1 col15" >0.632835</td>
      <td id="T_571d6_row1_col16" class="data row1 col16" >0.119118</td>
    </tr>
    <tr>
      <th id="T_571d6_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_571d6_row2_col0" class="data row2 col0" >isy-finetune-v2</td>
      <td id="T_571d6_row2_col1" class="data row2 col1" >0.742424</td>
      <td id="T_571d6_row2_col2" class="data row2 col2" >0.873106</td>
      <td id="T_571d6_row2_col3" class="data row2 col3" >0.910985</td>
      <td id="T_571d6_row2_col4" class="data row2 col4" >0.945076</td>
      <td id="T_571d6_row2_col5" class="data row2 col5" >0.742424</td>
      <td id="T_571d6_row2_col6" class="data row2 col6" >0.349116</td>
      <td id="T_571d6_row2_col7" class="data row2 col7" >0.232197</td>
      <td id="T_571d6_row2_col8" class="data row2 col8" >0.125758</td>
      <td id="T_571d6_row2_col9" class="data row2 col9" >0.618847</td>
      <td id="T_571d6_row2_col10" class="data row2 col10" >0.800122</td>
      <td id="T_571d6_row2_col11" class="data row2 col11" >0.856144</td>
      <td id="T_571d6_row2_col12" class="data row2 col12" >0.906343</td>
      <td id="T_571d6_row2_col13" class="data row2 col13" >0.813567</td>
      <td id="T_571d6_row2_col14" class="data row2 col14" >0.816299</td>
      <td id="T_571d6_row2_col15" class="data row2 col15" >0.769737</td>
      <td id="T_571d6_row2_col16" class="data row2 col16" >0.104379</td>
    </tr>
  </tbody>
</table>

Key takeaways:

- The v2 finetune (balanced + hard negatives + CachedMNR) substantially improves top-1 and top-5 accuracy and MRR relative to both the base encoder and an earlier finetune.

- Latency per query remains low (~0.10s) and is acceptable for many production use cases. Exact latency will depend on hardware and retrieval index configuration.


