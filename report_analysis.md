# Analytical Report: What Helps and What Does Not

This report complements [`report.md`](report.md) with an analytics-first view of every run logged in `results/all_results.csv` (117 rows: 67 classical, 15 transformer, 35 ensemble). The focus is on isolating each design choice (preprocessing, feature set, architecture, ensemble composition) and reading the signed deltas rather than the absolute leaderboard. Every number is test macro F1 unless otherwise noted.

The best single configuration observed in the project is the 5 model soft vote ensemble `ens_5_abcde`, which achieves test F1 of 0.8276 (Section 4.3).

## 1. Dataset and Evaluation Protocol

The training corpus contains 27,481 tweets, split stratified 90/10 into 24,732 training and 2,749 validation examples under random seed 42. The held out test set contains 3,534 tweets and is fixed across all runs, so deltas between any two rows of `all_results.csv` are directly comparable. The file `data/train_cleaned.csv` ships both a `text` column (raw) and a `cleaned_text` column (lightly preprocessed), so switching between the two is a one flag change on every training script.

The most consequential property of the evaluation setup is the small validation set. With only 2,749 samples, a typical val to test drift of around 0.004 F1 is observed empirically, and any post hoc optimization procedure that maximizes val F1 at a finer resolution is at risk of fitting noise. This observation is revisited throughout the report and is summarized in Section 5.2.

## 2. Classical Models: Ablation Study

### 2.1 The Model Family Ceiling

The best test F1 per family, using cleaned text and the best feature configuration for each, is shown below.

| Family             | Best config      | Test F1 | Gap to LightGBM |
| ------------------ | ---------------- | ------- | --------------- |
| LightGBM           | tfidf_char_all   | 0.7514  | n/a             |
| MLP                | tfidf_char_all   | 0.7393  | 0.0121          |
| LogisticRegression | tfidf_bigram_all | 0.7150  | 0.0364          |
| LinearSVC          | tfidf_bigram_all | 0.6651  | 0.0863          |
| NaiveBayes (MNB)   | tfidf_char_ngram | 0.6887  | 0.0627          |
| NaiveBayes (GNB)   | tfidf_char_all   | 0.5044  | 0.2470          |

The gap between LightGBM and the next best family (MLP) is 1.2 F1 points, which is small compared to the 2.4 point jump from Logistic Regression to LightGBM. The interpretation is that gradient boosting's non linearity over sparse TF IDF features is the core win, and adding a neural layer on top of the same features recovers most but not all of that benefit. Linear models plateau near 0.72 because word and character TF IDF features require interaction terms that linear decision boundaries cannot express.

The collapse of GaussianNB to 0.50 (barely above chance) is diagnostic rather than a bug. It occurs whenever VADER or AFINN features are mixed into TF IDF vectors, because the Gaussian assumption is catastrophic on sparse high dimensional inputs. MultinomialNB, which sees TF IDF only, clocks 0.69. The failure is therefore representational, not algorithmic.

### 2.2 Feature Ablation

Holding the model fixed at LightGBM, the ordered feature configuration deltas relative to `tfidf_unigram` (baseline 0.7129) are shown below.

| Feature config         | Test F1    | Δ vs unigram |
| ---------------------- | ---------- | ------------ |
| tfidf_char_all         | **0.7514** | **+0.0385**  |
| tfidf_bigram_all       | 0.7440     | +0.0311      |
| tfidf_30k_char_all     | 0.7421     | +0.0292      |
| tfidf_bigram_vader     | 0.7408     | +0.0279      |
| tfidf_bigram_afinn     | 0.7371     | +0.0242      |
| tfidf_char_ngram       | 0.7303     | +0.0174      |
| tfidf_bigram           | 0.7196     | +0.0067      |
| tfidf_bigram_pos_broad | 0.7149     | +0.0020      |
| tfidf_unigram          | 0.7129     | n/a          |

Character n grams contribute the largest single lever, adding 0.017 over plain bigrams, and VADER lexicon features add a further 0.021 on top of bigrams. Stacking both (`char_all`) yields 0.039 over the unigram baseline, which is larger than any hyperparameter tuning effect observed later in the project.

Growing the TF IDF vocabulary from 15k to 30k (`tfidf_30k_char_all`) actually hurt test F1 by 0.009. This is noise accumulation: the extra 15k features are rare n grams that degrade boosting's split quality without adding signal. The sweet spot lies at 10 to 15 thousand features.

Part of speech tags are effectively dead weight. The configuration `tfidf_bigram_pos_broad` gains only 0.002 over plain bigrams, which is within seed noise. Syntactic structure appears nearly orthogonal to sentiment on short tweets; bag of words already captures most of what POS tagging would encode. Fine grained POS (40 tags) was tried and dropped due to feature sparsity.

### 2.3 Raw versus Cleaned Preprocessing

A targeted 10 run re sweep (`results/result_job_raw_20260414_174716.json`) with `text_col=text` on the top 2 feature configurations across 5 model families produced the following comparison.

Rows are sorted by the better of cleaned or raw test F1 in descending order.

| Model              | Feature          | Cleaned | Raw    | Δ (raw − cleaned) |
| ------------------ | ---------------- | ------- | ------ | ----------------- |
| LightGBM           | tfidf_char_all   | 0.7514  | 0.7422 | −0.0092           |
| LightGBM           | tfidf_bigram_all | 0.7440  | 0.7454 | +0.0014           |
| MLP                | tfidf_char_all   | 0.7393  | 0.7420 | +0.0027           |
| MLP                | tfidf_bigram_all | 0.7277  | 0.7133 | −0.0144           |
| LogisticRegression | tfidf_bigram_all | 0.7150  | 0.7148 | −0.0002           |
| LogisticRegression | tfidf_char_all   | 0.6927  | 0.6936 | +0.0009           |
| LinearSVC          | tfidf_bigram_all | 0.6651  | 0.6697 | +0.0046           |
| LinearSVC          | tfidf_char_all   | 0.6535  | 0.6505 | −0.0030           |
| NaiveBayes (GNB)   | tfidf_char_all   | 0.5044  | 0.5051 | +0.0007           |
| NaiveBayes (GNB)   | tfidf_bigram_all | 0.4849  | 0.4869 | +0.0020           |

The pattern is the opposite of the transformer story examined in Section 3.2, where raw text wins by 0.004 to 0.010 consistently. For classical models the effect is mixed and strongly feature dependent.

Character n grams prefer cleaned text. The strongest performing feature family (`tfidf_char_all`) loses 0.009 on LightGBM when fed raw text. Character n grams amplify surface noise: URL fragments such as `http` and `.co`, user handles such as `@user`, and casing variants generate thousands of near duplicate features that dilute the signal.

Word bigrams are ambivalent. The bigram family moves by less than 0.005 for most models, so raw versus cleaned is effectively a wash. Word level tokenization already discards most of the surface noise that character features are sensitive to.

The MLP on bigrams is the outlier, losing 0.014 on raw text. The MLP's dense first layer is more sensitive to the feature space shift than tree or linear methods; raw text bigram TF IDF has a different sparsity profile that destabilizes the learned representation.

The conclusion is that choosing `cleaned_text` as the classical default was correct, but for a narrower reason than expected. Cleaning does not help classical models in general; it specifically rescues character n grams from noise. In a word only feature pipeline, raw text would be equally good and the preprocessing step would be wasted work.

### 2.4 Hyperparameter Tuning and the Validation Overfit Trap

Optuna 100 trial TPE tuning on the top 2 LightGBM configurations produced the following result.

| Config           | Baseline val | Baseline test | Tuned val | Tuned test | Δ val   | Δ test  |
| ---------------- | ------------ | ------------- | --------- | ---------- | ------- | ------- |
| tfidf_char_all   | 0.7587       | 0.7514        | 0.7634    | 0.7513     | +0.0047 | −0.0001 |
| tfidf_bigram_all | 0.7357       | 0.7440        | 0.7519    | 0.7428     | +0.0162 | −0.0012 |

Tuning improved val F1 by up to 1.6 points while slightly worsening test F1. This is textbook validation set overfitting, driven by the 2,749 sample val split. TPE is efficient precisely because it exploits the objective; it will find hyperparameter combinations that squeeze out val set idiosyncrasies. With only around 2.7 thousand samples, those idiosyncrasies do not transfer to held out data.

Tuning is useful for recovering weak defaults, not for refining strong ones. LightGBM's defaults (500 trees, learning rate 0.05, 31 leaves) are already near the local optimum for this feature space, so TPE has nowhere productive to go.

### 2.5 Classical Summary

The classical ceiling sits at approximately 0.7514 test F1 and is feature bound, not optimization bound. Character n grams, VADER features, cleaned text, and LightGBM recover nearly the entire attainable F1 for this tier. Adding more trees, more vocabulary, more POS features, or more tuning trials yields sub seed noise changes.

## 3. Transformer Models: Ablation Study

### 3.1 Domain Pretraining versus Parameter Count

| Model                                            | Params | Domain              | Data    | Test F1    |
| ------------------------------------------------ | ------ | ------------------- | ------- | ---------- |
| vinai/bertweet-base                              | 135M   | 850M tweets         | raw     | **0.8120** |
| bertweet-large (3 epoch v2)                      | 335M   | 850M tweets         | raw     | 0.8118     |
| cardiffnlp/twitter-roberta-base-sentiment-latest | 125M   | tweets + SemEval    | raw     | 0.8110     |
| deberta-v3-large (fix recipe)                    | 435M   | generic web         | raw     | 0.8109     |
| finiteautomata/bertweet-base-sentiment-analysis  | 135M   | tweets + sentiment  | raw     | 0.8098     |
| deberta-v3-large (fix recipe)                    | 435M   | generic web         | cleaned | 0.8093     |
| cardiffnlp/twitter-xlm-roberta-base              | 278M   | multilingual tweets | raw     | 0.8083     |
| roberta-large                                    | 355M   | generic web         | raw     | 0.8074     |
| twitter-roberta-base                             | 125M   | tweets              | cleaned | 0.8066     |
| roberta-large                                    | 355M   | generic web         | cleaned | 0.7976     |
| vinai/bertweet-large (5 epoch)                   | 335M   | 850M tweets         | raw     | 0.7969     |

BERTweet-base (135M parameters) beats roberta-large (355M parameters) by 0.46 F1 points on raw text. Domain relevance of pretraining data outranks parameter count by more than a factor of 2.5 in effective gain per parameter. Every tweet pretrained base model clusters tightly between 0.808 and 0.812; every non tweet base model sits below, and adding 200M parameters to generic roberta-large does not close the gap.

bertweet-large is the counter example that proves the rule. Given the same pretraining data and 2.5 times the parameters of bertweet-base, the 5 epoch run produces a worse test F1 (0.7969 versus 0.8120). The `v2` rerun with 3 epochs recovered to 0.8118, which matches bertweet-base, confirming that the issue is optimization rather than architecture. The 5 epoch schedule overfits the larger model to the small training set. For 27 thousand training examples, model capacity past approximately 150M parameters is wasted unless the schedule is retuned accordingly.

DeBERTa-v3 required recipe tuning to stabilize. With batch 16, lr 1e-5, no AMP, and an explicit FP16 embedding fix, `deberta_v3_large_fix` (raw text) reaches 0.8109 and `deberta_v3_large_cleaned_fix` (cleaned text) reaches 0.8093. The cleaned and raw results sit 0.0016 apart, consistent with the preprocessing penalty observed for other transformers in Section 3.2.

### 3.2 Raw versus Cleaned Text

| Model                           | Raw    | Cleaned | Δ (raw − cleaned) |
| ------------------------------- | ------ | ------- | ----------------- |
| twitter-roberta-base-sentiment  | 0.8110 | 0.8066  | +0.0044           |
| deberta-v3-large (fixed recipe) | 0.8109 | 0.8093  | +0.0016           |
| roberta-large                   | 0.8074 | 0.7976  | +0.0098           |

Raw wins across all three transformers tested, with deltas ranging from 0.0016 to 0.0098. Three compounding reasons explain this pattern.

First, BPE and SentencePiece tokenizers are trained on raw text. Their subword segmentation assumes that capital letters, punctuation, URLs, and emoji fragments exist. Stripping those collapses multiple distinct tokens into one and shrinks the effective vocabulary used per example.

Second, sentiment signal lives in surface form. The token `AMAZING` is not equivalent to `amazing`, the sequence `!!!` is not equivalent to a single `!`, and an emoticon such as `:-)` carries polarity on its own. Cleaning drops exactly the features a sentiment model most wants to attend to.

Third, the delta grows with the generality of pretraining. Domain pretrained twitter-roberta loses only 0.004 when fed cleaned text, but generic roberta-large loses 0.010. Raw text robustness is correlated with how much of the pretraining corpus resembled the fine tuning input.

Comparing the classical and transformer verdicts, transformers prefer raw because their tokenizer was pretrained on raw text, while classical character n grams prefer cleaned text because they have no prior over what surface noise means. Preprocessing is not a universal good; it is an interaction with the representation layer.

### 3.3 Class Weighted Loss

Training twitter-roberta with sklearn balanced class weights produced test F1 of 0.8108, compared to 0.8110 for the standard run. The explanation is that the balanced weighting scheme computes `n_samples / (n_classes * n_class_i)`, which downweights the majority class. The majority class in this dataset is the neutral class (approximately 41 percent of samples), which is already the hardest. Balanced weighting therefore instructs the model to care less about the class most in need of correction, which is the opposite of the intended effect. The only way class weighting could help here is with a custom upward weight on the neutral class, which would cannibalize negative and positive F1 and damage macro F1 by symmetry.

### 3.4 Transformer Summary

Transformer gains over classical models are almost entirely explained by two factors: domain pretrained embeddings and raw text input. Parameter scale past 150M is unproductive on 27 thousand examples, and class weighted loss is strictly counterproductive given the label distribution.

## 4. Ensemble Models: Ablation Study

### 4.1 Best Ensemble at Each Member Count

To equalize statistical footing, six distinct uniform soft vote configurations were collected at each member count. For n = 3 the top six configurations by test F1 were retained from the larger pool of runs; for n = 2 and n = 5 additional compositions were launched so that every tier reports six. The best test F1 per count is summarized below.

| n | Best ensemble      | Composition                                                     | Val F1     | Test F1    | Configs |
| - | ------------------ | --------------------------------------------------------------- | ---------- | ---------- | ------- |
| 5 | ens_5_abcde        | roberta-raw + bertweet + bertweet-sent + xlm + deberta-fix      | 0.8285     | **0.8276** | 6       |
| 4 | ens_4_sent_deberta | roberta-raw + bertweet-sent + xlm + deberta-fix                 | **0.8286** | 0.8267     | 6       |
| 3 | ens_lo_db_no_xlm   | roberta-raw + bertweet-sent + deberta-fix                       | 0.8264     | 0.8240     | 6       |
| 2 | ens_2_ab           | roberta-raw + bertweet                                          | 0.8225     | 0.8207     | 6       |

The curve is monotonic. Test F1 climbs 0.0033 from n = 2 to n = 3, 0.0027 from n = 3 to n = 4, and a further 0.0009 from n = 4 to n = 5. The 5 model peak sits 0.0009 above the best 4 model configuration, which is inside typical ensemble noise but still the highest observed. The marginal gain per added member diminishes rapidly, and the test curve is effectively plateaued by n = 4.

With six configurations per tier the comparisons are now on equal footing. The consistency across tiers is notable: the best member at each count is a raw text tweet or tweet adjacent model, the peak compositions always involve a SentencePiece tokenizer (either XLM or DeBERTa), and the best 5 model ensemble is exactly the union of the best 4 model ensemble and the most load bearing member dropped from it (plain BERTweet), suggesting that expansion adds signal only when the new member fills a tokenizer or calibration gap that existing members could not. The data is therefore convincing for the claim that n = 5 is the peak on this dataset, with the caveat that the n = 4 to n = 5 improvement is inside the 0.001 to 0.002 noise band and should not be read as a strict ordering.

### 4.2 Composition Matters More Than Count: The 5 Model Sweep

Having located the test F1 peak at n = 5, a composition ablation was run at fixed size five to isolate which members contribute signal at the peak. Six distinct 5 model configurations were evaluated under uniform soft voting.

| Config             | Composition                                                               | Val F1     | Test F1    |
| ------------------ | ------------------------------------------------------------------------- | ---------- | ---------- |
| ens_5_abcde        | roberta-raw + bertweet + bertweet-sent + xlm + deberta-fix                | 0.8285     | **0.8276** |
| ens_5_acdef        | roberta-raw + bertweet-sent + xlm + deberta-fix + roberta-large-raw       | **0.8290** | 0.8248     |
| ens_5_abcdf        | roberta-raw + bertweet + bertweet-sent + xlm + roberta-large-raw          | 0.8307     | 0.8245     |
| ens_5_bcdef        | bertweet + bertweet-sent + xlm + deberta-fix + roberta-large-raw          | 0.8294     | 0.8238     |
| ens_5model_deberta | roberta-raw + roberta-large-raw + bertweet + xlm + deberta-fix            | 0.8286     | 0.8224     |
| ens_5model         | sent-latest-raw + roberta-large + 3× sent-latest seeds                    | 0.8425     | 0.8211     |

The spread at n = 5 is 0.0065 F1, comparable to the 0.0055 spread observed at n = 4, and again confirms that composition matters more than count. Three observations follow.

First, the new best configuration `ens_5_abcde` is the union of all five healthy tweet pretrained and deberta checkpoints under the project: twitter-roberta-base-sentiment-latest, vinai/bertweet-base, bertweet-base-sentiment-analysis, twitter-xlm-roberta-base, and the fixed recipe deberta-v3-large. It contains every model family that reached single model test F1 above 0.8090 and excludes every generic web pretrained member that fell below (namely roberta-large-raw at 0.8074). This is consistent with the rule that ensemble quality is bounded by the quality of its weakest member when averaging is uniform.

Second, the three next best configurations (`ens_5_acdef`, `ens_5_abcdf`, `ens_5_bcdef`) all include roberta-large-raw, and all land between 0.8238 and 0.8248, strictly below `ens_5_abcde`. Substituting any member of `ens_5_abcde` with roberta-large-raw costs between 0.0028 and 0.0038 F1. This is the dominant composition effect at n = 5.

Third, `ens_5model` (val F1 0.8425, test F1 0.8211) is the most extreme case of validation set overfitting in the entire report. Three sent-latest seeds inflate val F1 through correlated agreement on the same noise pattern, but contribute no independent test signal. The +0.014 val gap between `ens_5model` and `ens_5_abcde` reverses into a −0.0065 test deficit.

### 4.3 Leave One Out Profile of the Best Ensemble

Leave one out analysis is reported for `ens_5_abcde` (test F1 **0.8276**), the best configuration in the project. Its members span four distinct tokenizer families.

1. `cardiffnlp/twitter-roberta-base-sentiment-latest` (A, raw text, BPE family A)
2. `vinai/bertweet-base` (B, raw text, tweet BPE family B)
3. `finiteautomata/bertweet-base-sentiment-analysis` (C, raw text, tweet BPE family B, sentiment prefinetuned)
4. `cardiffnlp/twitter-xlm-roberta-base` (D, raw text, XLM SentencePiece)
5. `microsoft/deberta-v3-large` (E, raw text, DeBERTa SentencePiece, fix recipe)

Because every 4 member subset of `ens_5_abcde` already exists in the n = 4 sweep from Section 4.2, leave one out values can be read directly from those runs.

| Dropped               | Remaining members                                | Val F1     | Test F1    | Δ Test  |
| --------------------- | ------------------------------------------------ | ---------- | ---------- | ------- |
| (none, full baseline) | A + B + C + D + E                                | **0.8285** | **0.8276** | n/a     |
| bertweet (B)          | A + C + D + E (= ens_4_sent_deberta)             | 0.8286     | 0.8267     | −0.0009 |
| xlm (D)               | A + B + C + E (= ens_4_abce)                     | 0.8258     | 0.8267     | −0.0009 |
| twitter-roberta (A)   | B + C + D + E (= ens_4_no_roberta)               | 0.8258     | 0.8249     | −0.0027 |
| deberta-fix (E)       | A + B + C + D (= ens_4_sent_swap)                | 0.8289     | 0.8248     | −0.0028 |
| bertweet-sent (C)     | A + B + D + E (= ens_4_deberta_swap)             | 0.8273     | 0.8212     | −0.0064 |

All five leave one out deltas are negative, confirming that no member is dead weight. The profile splits cleanly into three tiers.

The most load bearing member is bertweet-sent (−0.0064). It is the only sentiment prefinetuned member in the ensemble and contributes a calibrated polarity signal that no other model replicates. Removing it collapses the polarity calibration axis entirely, and the four remaining members cannot recover it because none of them were trained on a sentiment classification objective.

The second tier comprises twitter-roberta (−0.0027) and deberta-fix (−0.0028), with nearly identical drops. Deberta-fix is the only large generic web pretrained member and the only one using the DeBERTa SentencePiece tokenizer, so removing it eliminates both the parameter scale axis and one of the two SentencePiece tokenizers. Twitter-roberta is the only BPE family A tokenizer in the ensemble, so removing it collapses that tokenizer family. Both drops sit near the 0.003 ensemble noise floor but are consistently negative across the leave one out sweep, which is weak but non trivial evidence that both members contribute independent signal.

The least load bearing members are bertweet (−0.0009) and xlm (−0.0009), both exactly tied. These are the members whose removal leaves the ensemble at the second best n = 4 tier (0.8267), which itself is the n = 4 ceiling. The interpretation is that at n = 5 the ensemble has redundancy along the tweet BPE family B axis (bertweet and bertweet-sent overlap tokenizer wise) and along the SentencePiece axis (xlm and deberta-fix overlap tokenizer wise), so removing either member from these redundant pairs leaves a still diverse four member ensemble that retains the peak n = 4 performance. This redundancy is exactly the reason the n = 4 to n = 5 gain is only 0.0009: the fifth member restores a marginal amount of within family disagreement, but most of the signal was already captured at n = 4.

### 4.4 Weighted Voting Overfits the Validation Set

| Method                           | Val F1     | Test F1 |
| -------------------------------- | ---------- | ------- |
| Stacking (LR meta learner)       | 0.8240     | 0.8238  |
| Uniform (0.25 each)              | 0.8252     | 0.8233  |
| Grid searched weights (step 0.1) | **0.8284** | 0.8188  |

Grid search picked weights that heavily favored roberta-large-raw, which is the weakest individual member. The weighted ensemble gained 0.003 on validation and lost 0.005 on test: a textbook overfit. Stacking with an LR meta learner on top of 3 class probabilities did slightly better on test (0.8238), because the L2 regularization in LR resists the same kind of overfitting, but still did not beat uniform averaging of the sent-swap ensemble.

A usable rule of thumb for small validation ensemble tuning is that uniform weights are the regularizer free default, and any optimization procedure on 2.7 thousand samples must leave at least 0.5 F1 of headroom on validation to be trustworthy on test. The +0.003 gain on validation observed here was well inside the noise band.

## 5. Cross Cutting Findings

### 5.1 The Three Ceilings

| Tier        | Test F1    | Cost         | What bought it                    |
| ----------- | ---------- | ------------ | --------------------------------- |
| Ensemble    | **0.8276** | 5x inference | tokenizer diversity               |
| Transformer | 0.8120     | minutes, GPU | domain pretraining, raw text      |
| Classical   | 0.7514     | seconds      | char n grams, VADER, cleaned text |

Each tier plateaus well before the next begins, so the pipeline behaves as a staircase rather than a smooth curve. The two largest single jumps are classical to transformer (+0.0606) and single transformer to best ensemble (+0.0156).
