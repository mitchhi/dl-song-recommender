*(Keep clean and short, with separate pages for deep mathematical descriptions)*
To install repo src: inside the repo root on terminal,
1. Activate conda environment
2. Run `pip install --upgrade pip`
3. Run `pip install -e .`

# Deep Learning Song Recommender
This project builds a content-based music recommender that learns song similarity directly from audio. Using full-mix and stem spectrograms, we trained a late-fusion `ResNet18` to generate song embeddings shaped by genre-tag supervision and contrastive learning. The final model powers a web app where users can explore songs and receive recommendations based on learned audio similarity.

**App Website:** https://dl-song-recommender.onrender.com


## Background (Too wordy. Needed?)

Modern music platforms are extremely good at learning from user behavior, but behavior-only recommendation has limits. It struggles with cold-start profiles, sparse interaction data, and explaining similarity based on what songs actually sound like. (I'm lying here, I don't know what modern music platforms do.)

Content-based recommendation is a necessary evolution for these systems. If a model can learn similarity directly from audio, it can improve discovery, support new or less popular tracks, and complement collaborative methods with a richer understanding of the music itself.

This project explores that direction by using deep learning to learn song embeddings from audio while using listener-provided tags as semantic supervision during training.
 

## Approach

The project follows a simple idea:

1. Build a semantic notion of similarity from listener tags.
2. Train audio models to reproduce that structure from spectrograms and stems.
3. Evaluate whether the learned embeddings produce useful song retrieval.

Genre tags provides the semantic signal during training, but the long-term aim is audio-based recommendation at inference time.

## Feature Engineering

We derive semantic supervision from listener-generated tags in the metadata for our training set. First, we compute tag co-occurrence statistics and filter noisy tag relationships using Positive PMI and minimum co-occurrence thresholds. The resulting clean tag vocabulary is used to train a skip-gram Word2Vec model, producing a 64-dimensional embedding for each valid tag. These tag embeddings are then grouped with hierarchical ward clustering into 20 semantic tag clusters, giving each tag both a dense vector representation and a cluster assignment. For each song, we map its cleaned tags into cluster IDs and compute song-level semantic features such as `tag_clusters` and `dominant_cluster`, which are used in model evaluation. 

## Model Design

The core model is a late-fusion `ResNet18` trained on spectrograms. Each song is represented by a full-mix spectrogram together with stem spectrograms, and the same encoder is applied across these views to learn a compact audio representation.

![Model architecture](docs/diagrams/model_1.png)

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
*(Talk about this for future improvements, not for model selection.)*
<!-- Maybe some diagrams will help here. This (I believe) should be our main metric. -->


## Conclusions & Future Works
*(Add ideas as they come up. We'll synthesize everything and clean up towards the end.)*
 
