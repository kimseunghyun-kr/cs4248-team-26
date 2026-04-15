# CBDC NLP Adaptation: Code And Approach Brief

## 1. Project Goal

This project adapts the CBDC idea from the CVPR 2026 paper into a text-only sentiment setting. The current task is not financial sentiment analysis; it is general tweet-style sentiment classification with three labels: negative, neutral, and positive.

The main goal is not just to train a classifier, but to test whether CBDC-style latent representation shaping can make sentiment embeddings less dependent on nuisance cues such as punctuation, emoticons, laughter tokens, short length, and shallow topic/style shortcuts.

## 2. High-Level Pipeline

The full pipeline is orchestrated by run_all.py and is divided into four phases.

- `data/embed.py`: loads the dataset, freezes the selected Hugging Face backbone, and caches train/val/test sentence embeddings together with token IDs and metadata.
- `cbdc/refine.py`: materializes the derived conditions D1, D2, optional D2.5, and D3 into separate cache directories.
- `pipeline/prototype_classify.py`: evaluates each condition without training a classifier head, using cosine similarity to class prompt prototypes.
- `pipeline/evaluate.py`: writes the final summary report, including accuracy/F1 and direction-interpretability diagnostics.
- `pipeline/plot_pca.py`: optional PCA and cosine t-SNE diagnostics used for visualizing whether the embedding geometry moved.

## 3. Encoder Design

The central modeling code is in encoder.py. The backbone is frozen, and Phase 2 only trains a small target tail on top of an intermediate latent representation.

- Encoder-family models such as BERT, RoBERTa, XLM-R, and DistilBERT use first-token pooling as the sentence embedding.
- Decoder-family models such as Qwen2 and Llama use the last non-pad token as the pooled sentence embedding.
- The local implementation perturbs an intermediate latent representation and then runs the final encoder/decoder block(s), which makes it a faithful adaptation of the CBDC target-tail idea rather than a full multimodal reproduction.

## 4. Method Conditions

The project compares five conditions. Each condition produces its own cached embeddings and prompt prototypes.

- `B1 (raw)`: raw cached embeddings with raw class prompt prototypes.
- `D1 (debias_vl)`: a closed-form DebiasVL-style projection using dataset-grounded topic/style confounds. This is not an epoch-trained condition.
- `D2 (CBDC)`: pure prompt-driven CBDC training with fixed handcrafted style-bias pole pairs, using validation centroid macro-F1 for checkpoint selection.
- `D2.5 (CBDC no-label-select)`: same CBDC training objective as D2, but checkpoint selection uses prompt loss only, making it the cleanest label-free selector variant.
- `D3 (debias_vl->CBDC)`: DebiasVL-style confound discovery first, then CBDC training on top of the discovered anchors.

## 5. What Is Being Debiased

In the current codebase, the bias notion is not demographic fairness or finance-specific bias. It is mostly shortcut-style sentiment confounding.

The prompt bank in cbdc/prompts.py defines nuisance factors such as emoticons, question marks, repeated punctuation, very short messages, internet laughter, URLs, register/formality, and balanced everyday topics such as work, school, food, friends, music, and weekend plans.

The intended effect is to reduce reliance on these shallow correlations while preserving general content through keep_text prompts. In short: the system is trying to debias sentiment-from-shortcuts, not remove protected-attribute information.

## 6. How CBDC Training Works In This Code

For D2, D2.5, and D3, cbdc/refine.py runs bipolar latent PGD on prompt representations, not on labeled training examples. The attacked prompts come from target_text, and the preservation term is defined by keep_text.

Each epoch produces positive and negative perturbation branches. Their difference defines the current nuisance direction set, and the trainable tail is updated with a match loss plus a cross-knowledge penalty so that these nuisance directions become less aligned with the sentiment prototypes.

After checkpoint selection, the entire split is re-encoded with the selected condition-specific encoder state. The final plots and prototype evaluations therefore reflect the learned condition, not just a one-off PGD perturbation.

## 7. Evaluation Approach

The default evaluation path in the current project is prototype-based rather than linear-probe-based. That means no additional classifier head is trained for the main report.

- Embeddings are compared to class prompt prototypes using cosine similarity.
- The report includes validation/test accuracy, macro-F1, and per-class precision/recall/F1.
- Direction-interpretability diagnostics are computed from projection scores onto saved condition-specific directions.
- PCA and cosine t-SNE are used only as geometry diagnostics. They are useful for showing movement and local bundling, but they are not substitutes for macro-F1.

## 8. Current Empirical Read

The main conclusion so far is that the method is active in latent space, but strongly backbone-dependent. The losses move in Phase 2, the embeddings shift, and some backbones improve over raw baselines.

RoBERTa-base currently provides the strongest CBDC-style results. On the BERT family, D1 is often the strongest condition. Larger backbones and decoder-family models often expose prompt/prototype calibration failures such as neutral-class collapse, even when the latent-space training itself is clearly doing work.

This means the present bottleneck is probably not a dead optimizer. The stronger suspect is prompt-bank and prototype geometry calibration, especially in cbdc/prompts.py.

## 9. Why This Is A Reasonable Direction

At an assignment level, this is already a meaningful direction-setting project: it cleanly ports a nontrivial vision-language debiasing idea into NLP, preserves the main target-tail and latent-PGD logic, and exposes where the transfer works and where it breaks.

The current evidence suggests a real research question rather than a trivial implementation bug: some backbones respond well to the method, while others reveal that sentiment, topic, and stylistic cues are still entangled under the current prompt design.

## 10. Practical Next Steps

- Refine cbdc/prompts.py, especially target_text, keep_text, and the fixed CBDC style-bias pairs.
- Run cleaner ablations comparing D2 vs D2.5 vs D3 under the same backbone and fixed test split.
- Test whether a deeper target tail or different pooling choice helps the larger backbones.
- Add external sentiment transfer or domain-shift evaluation to show whether the learned representation shaping generalizes.
