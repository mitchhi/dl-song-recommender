# Deep Learning Song Recommender

**Authors:** [Nicholas Geis](https://github.com/nicholassgeis), [Mitch Hamidi-Ismert](https://github.com/mitchhi), [Juan Salinas](https://github.com/juansalinas2)

This project builds a content-based music recommender that learns song similarity directly from audio. Using full-mix and stem spectrograms, we trained a late-fusion `ResNet18` to generate song embeddings shaped by genre-tag supervision and contrastive learning. The final model powers a web app where users can explore songs and receive recommendations based on learned audio similarity.

**App Website:** https://dl-song-recommender.onrender.com


## Background (*WORK ON THIS - needs to tell the situation and create the need for our model*)

Modern music platforms excel with user behavior, but behavior-only recommendation has limits. A content-agnostic approach struggles with cold-start profiles, sparse interaction data, and explaining similarity based on what songs actually sound like. 

Content-based recommendation is a necessary evolution for these systems. If a model can learn similarity directly from audio, it can improve discovery, support new or less popular tracks, and complement collaborative methods with a richer understanding of the music itself.

This project explores that direction by using deep learning to learn song embeddings from audio while using listener-provided tags as semantic supervision during training.
 

## Approach

The project follows a simple idea:

1. Build a semantic notion of similarity from listener tags.
2. Train audio models to reproduce that structure from spectrograms and stems.
3. Evaluate whether the learned embeddings produce useful song retrieval.

Genre tags provides the semantic signal during training, but the long-term aim is audio-based recommendation at inference time.

## Data and Preprocessing

Our dataset contains **11,239 songs** with associated metadata and derived audio representations. The project draws from two Kaggle sources:

1. [Augmented-Audio-10k](https://www.kaggle.com/datasets/reggiebain/augmented-audio-10k)
2. [Million Song Dataset + Spotify + Last.fm Music Tracks](https://www.kaggle.com/datasets/undefinenull/million-song-dataset-spotify-lastfm)

The metadata used throughout the project includes track identifiers, artist and title information, Spotify preview links, year, and [Last.fm](https://www.last.fm) user-generated listener tags, which were distributed within the [Million Song Dataset](http://millionsongdataset.com).

### Why Tags?

The recommender is ultimately intended to operate from audio alone at inference time, but the project needs a meaningful training signal for “musical similarity.” Listener tags provide that supervision. Tags such as `rock`, `indie`, and `chillout` give a weak but useful semantic description of how songs relate, and the later audio models are trained to reproduce that tag-informed structure from spectrogram inputs.

### Data split

The dataset is split into train, validation, and test sets (**73**/**12**/**15**) in the [preprocessing notebooks](https://github.com/mitchhi/dl-song-recommender/tree/main/notebooks/00_preprocessing). Before splitting, the project removes very rare tags (frequency less than 5) so that multilabel stratification on tags is more stable. Tag-derived structures used later in the pipeline are built **only from the training set**.  

### Audio preprocessing

```mermaid
flowchart LR
    A["full mix" audio clip] --> B[Stem separation]
 
    B --> B1[Bass]
    B --> B2[Drums]
    B --> B3[Other]
    B --> B4[Vocals]

    A --> C[Mel spectrogram]
    B1 --> D[Mel spectrogram]
    B2 --> E[Mel spectrogram]
    B3 --> F[Mel spectrogram]
    B4 --> G[Mel spectrogram]

    C --> J1[PNG]
    D --> J2[PNG]
    E --> J3[PNG]
    F --> J4[PNG]
    G --> J5[PNG]

    J1 --> K[Model]
    J2 --> K
    J3 --> K
    J4 --> K
    J5 --> K

```

Each song is represented by a **10-second** audio clip sampled at **22,050 Hz**. From that clip, the preprocessing pipeline creates four source-separated stem audio clips. Each audio clip is then represented as a mel spectrogram PNG. 

#### Step 1: Stem Separation

The full audio clip is split into four stems (bass/drums/other/vocals) using [Demucs](https://github.com/adefossez/demucs). This gives the model access not only to the full mix, but also to more musically targeted views of rhythm and harmony.

#### Step 2: Mel Spectrogram Generation

For the full mix and each stem, the project uses `librosa` to compute a [mel spectrogram](https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53), which is a visual representation of an audio signal on a frequency scale that mimics human hearing perception.

![Sample Mel Spectrogram](docs/diagrams/spectrogram_example.png)

Spectrogram magnitudes are converted to decibels normalized relative to the full mix’s maximum power and saved as an 8-bit color-mapped PNG with dimensions **862 × 256**.  

## Feature Engineering: Tag Embeddings

```mermaid
graph LR
    A[Raw tags] --> B[Co-occurrence]
    B --> C[PMI filtering]
    C --> D[Valid tag set]
    D --> E[Word2Vec embeddings]
    E --> F[Clustering]
    F --> G[Tag features]
```

We derive semantic supervision from listener-generated tags in the metadata for our training set. First, we compute tag co-occurrence statistics and filter noisy tag relationships using Positive PMI and minimum co-occurrence thresholds. The resulting clean tag vocabulary is used to train a skip-gram Word2Vec model, producing a 64-dimensional embedding for each valid tag. These tag embeddings are then grouped with hierarchical ward clustering into 20 semantic tag clusters, giving each tag both a dense vector representation and a cluster assignment. For each song, we map its cleaned tags into cluster IDs and compute song-level semantic features such as `tag_clusters` and `dominant_cluster`, which are used in model evaluation. 

For the full details, see [Tag Embeddings and Clustering](docs/tag_embeddings.md).

## Model Design

The core model is a late-fusion `ResNet18` trained on spectrograms. Each song is represented by a full-mix spectrogram together with stem spectrograms, and the same encoder is applied across these views to learn a compact audio representation.

```mermaid
flowchart LR
    A[Song Audio] --> B[Mix plus 4 Stem Spectrograms]
 
    B --> C[Shared ResNet18 Encoder]

 
    C --> D[Harmonic Branch]


    C --> E1[Mix Embedding]
    D --> E2[Harmonic Pooling]
    C --> E3[Drum Branch]

    E1 --> F[Late Fusion]
    E2 --> F
    E3 --> F

    F --> G[Song Embedding]

    G --> I1[Cosine Retrieval]
    G --> I2[Relational Semantic Loss]
    G --> I3[View Alignment Loss]

    H[Frozen Tag Teacher] --> I2

```

These views are combined into a single retrieval embedding,

$$
z = \mathrm{normalize}(m + \alpha_h h + \alpha_d d),
$$

where $m$ is the mix embedding, $h$ is a harmonic embedding formed from bass, other, and vocals, and $d$ is a drum embedding. The learned coefficients $\alpha_h$ and $\alpha_d$ control how strongly those musical components shape the final representation. Recommendation is then performed by nearest-neighbor search in this embedding space.

What changes across notebooks 4-7 is not the backbone so much as the learning objective. The first model uses genre-tag-based learning: genre tags define which songs should be close, and the `ResNet18` is trained to recover that tag-defined geometry from audio alone. Later models add contrastive learning to keep different views of the same song close in embedding space. The final experiment combines both ideas by training against a blended audio-tag teacher.

At a high level, the project studies one stable `ResNet18` architecture under three related training ideas: **tag-based learning**, **contrastive learning**, and **blended audio-tag supervision**. For the full architecture, loss function, and notebook-by-notebook progression, see [Model Architecture](docs/model_architecture.md).

 

## Evaluation

Saved experiment manifests show strong semantic retrieval performance across the later models.

| Model | Artist Hit@10 | Tag Overlap Hit@10 | Semantic Teacher Coverage |
| --- | ---: | ---: | ---: |
| Tag-aligned audio encoder | 0.118 | 0.944 | 0.994 |
| Contrastive semantic audio encoder | 0.141 | 0.949 | 0.993 |
| Audio-grounded contrastive encoder | 0.114 | 0.949 | 0.994 |
| Blended-teacher late-fusion encoder | 0.123 | 0.937 | 0.994 |

These results suggest...

## User Evaluation

We evaluate ResNet04 (our final contrastive model) against a tag-embedding baseline model by generating the top-k recommendations for various k and then computing several quality metrics. The baseline embeds each song as a weighted average of its raw tag vectors, resulting in a 5,120-dimensional representation. ResNet04 produces a compact 512-dimensional audio embedding.

For each track in the testing dataset, we create a list of the top-k recommendations from the training dataset and compute several metrics. The averages of those metrics across several values of k are presented below.

### Metrics

To define several of the metrics, a notion of track **relevance** is necessary. Two tracks are considered *relevant* if

1. both tracks have the same dominant cluster or
2. they have *significant* cluster overlap.

From here, we used the following metrics to evaluate the performance of ResNet04 and the baseline model. Each of these measures whether the retrieved songs share semantic structure with the query.

| Metric | What it measures |
|---|---|
| **Precision@k** | Fraction of the top-k recommendations that are relevant |
| **Recall@k** | Fraction of all relevant songs in the catalog that appear in top-k |
| **MAP@k** | Mean Average Precision — rewards relevant songs appearing earlier in the list |
| **NDCG@k** | Normalized Discounted Cumulative Gain — rank-weighted relevance score |
| **Tag Jaccard@k** | Overlap between query and recommended songs' tag sets |
| **Cluster Jaccard@k** | Overlap between query and recommended songs' tag cluster sets |
| **Dominant Cluster Accuracy@k** | Fraction of recommended songs sharing the query's dominant semantic cluster |

Since music recommendations vary between individuals, we wanted to include some metrics that were not tag based. The following metrics are used to measure **diversity and novelty** across the track recommendations.

| Metric | What it measures |
|---|---|
| **Artist Diversity@k** | Fraction of unique artists in top-k (higher = more variety) |
| **Intra-List Diversity@k** | 1 − mean pairwise cosine similarity of recommended embeddings (higher = more spread in embedding space) |
| **Novelty@k** | Mean self-information of recommended artists relative to catalog frequency (higher = rarer artists surfaced) |
| **Discounted Novelty@k** | Rank-weighted novelty, so rare artists appearing earlier contribute more |

See Notebook 09 for more precise definitions on each of the metrics.

### Results (test set)

#### k = 5

| Metric | Baseline | ResNet04 | Δ |
|---|---:|---:|---:|
| Precision@5 | 0.383 | **0.702** | +83% |
| Recall@5 | 0.0014 | **0.0025** | +79% |
| MAP@5 | 0.527 | **0.762** | +45% |
| NDCG@5 | 0.611 | **0.804** | +32% |
| Tag Jaccard@5 | 0.110 | **0.217** | +97% |
| Cluster Jaccard@5 | 0.197 | **0.356** | +81% |
| Dominant Cluster Acc@5 | 0.196 | **0.385** | +96% |
| Artist Diversity@5 | **0.976** | 0.908 | −7% |
| Intra-List Diversity@5 | 0.029 | **0.037** | +26% |
| Novelty@5 | 10.340 | **10.478** | +1% |
| Discounted Novelty@5 | 10.340 | **10.473** | +1% |

#### k = 10

| Metric | Baseline | ResNet04 | Δ |
|---|---:|---:|---:|
| Precision@10 | 0.368 | **0.701** | +90% |
| Recall@10 | 0.0026 | **0.0050** | +92% |
| MAP@10 | 0.500 | **0.752** | +50% |
| NDCG@10 | 0.638 | **0.817** | +28% |
| Tag Jaccard@10 | 0.105 | **0.217** | +107% |
| Cluster Jaccard@10 | 0.190 | **0.356** | +87% |
| Dominant Cluster Acc@10 | 0.186 | **0.384** | +106% |
| Artist Diversity@10 | **0.957** | 0.853 | −11% |
| Intra-List Diversity@10 | 0.031 | **0.041** | +32% |
| Novelty@10 | 10.342 | **10.480** | +1% |
| Discounted Novelty@10 | 10.341 | **10.477** | +1% |

#### k = 20

| Metric | Baseline | ResNet04 | Δ |
|---|---:|---:|---:|
| Precision@20 | 0.360 | **0.699** | +94% |
| Recall@20 | 0.005 | **0.010** | +100% |
| MAP@20 | 0.458 | **0.736** | +61% |
| NDCG@20 | 0.654 | **0.826** | +26% |
| Tag Jaccard@20 | 0.101 | **0.216** | +114% |
| Cluster Jaccard@20 | 0.185 | **0.357** | +93% |
| Dominant Cluster Acc@20 | 0.180 | **0.386** | +114% |
| Artist Diversity@20 | **0.930** | 0.785 | −18% |
| Intra-List Diversity@20 | 0.032 | **0.047** | +47% |
| Novelty@20 | 10.356 | **10.480** | +1% |
| Discounted Novelty@20 | 10.351 | **10.478** | +1% |

### Subjective Metrics

Due to the subjective nature of musical taste, we were worried that our metrics would be lacking in some human element. To complement the quantitative metrics, we ran a blind listening survey to compare ResNet04, the baseline and two other models. The survey presented participants with 12 query songs spanning a range of genres. For each query, four playlists of 5 recommendations (one per model) were displayed side by side in randomized order. There was no indication of which model generated each playlist. Participants then listened to preview clips and rated each recommended track as **Good**, **Neutral**, or **Bad**.

Across all 12 query songs (60 rated tracks per model per respondent), the rating distributions were:

| Model | Good | Neutral | Bad |
|---|---:|---:|---:|
| Baseline | 15.8% | 30.8% | 53.4% |
| ResNet04 | **48.4%** | 30.8% | **20.8%** |
| ResNet06 | 45.4% | 29.1% | 25.5% |
| ResNet07 | 40% | 31.7% | 28.3% |

Although this survey was only administered to 4 participants, we observed that ResNet04 received both the highest share of Good ratings and the lowest share of Bad ratings while the baseline received the lowest and highest of those respective ratings. Additionally, the ratings of the other models seemed consistent with our findings using the validation dataset. As a result, these subjective results combined with the promising quantitative metrics on the validation dataset solidified our decision to select ResNet04 as our final model.

## Conclusion 

Across all list sizes, ResNet04 nearly doubles semantic relevance under Precision and Dominant Cluster Accuracy when compared to the baseline. The gains in MAP and NDCG confirm that relevant songs are not just present but also frequently rank near the top of a recommendation list. Artist diversity dips slightly as k grows (0.976 → 0.930 for the baseline, 0.908 → 0.785 for ResNet04), which aligns with expectations as longer lists are more likely to repeat artists. However, intra-list diversity is consistently higher for ResNet04 at every k. This suggests that the recommended embeddings are more spread out in audio space despite the tendency to recommend repeated artists.

## Future Directions

## Repository Structure
```
.
├── configs/                 # YAML configuration files
├── data/                    # metadata, processed splits, and run data
├── docs/                    # project documentation
├── notebooks/               # preprocessing, EDA, modeling, and evaluation notebooks
├── src/song_recommender/    # package code
├── README.md
├── pyproject.toml
├── requirements.txt
└── environment.yml
```

## Installation

From the repository root:

``` console
conda env create -f environment.yml
conda activate dl-song-recommender
pip install --upgrade pip
pip install -e .
```
 
