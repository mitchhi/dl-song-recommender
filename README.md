To install repo src: inside the repo root on terminal,
1. Activate conda environment
2. Run `pip install --upgrade pip`
3. Run `pip install -e .`

# Deep Learning Song Recommender

A content-based music recommendation system that learns song similarity from audio, using listener tags as semantic supervision and spectrogram-based deep learning models to retrieve related songs.

**App Website:** https://dl-song-recommender.onrender.com

## Why This Matters (Too wordy. Needed?)

Modern music platforms are extremely good at learning from user behavior, but behavior-only recommendation has limits. It struggles with cold-start profiles, sparse interaction data, and explaining similarity based on what songs actually sound like. (I'm lying here, I don't know what modern music platforms do.)

Content-based recommendation is a necessary evolution for these systems. If a model can learn similarity directly from audio, it can improve discovery, support new or less popular tracks, and complement collaborative methods with a richer understanding of the music itself.

This project explores that direction by using deep learning to learn song embeddings from audio while using listener-provided tags as semantic supervision during training.
 

## Approach

The project follows a simple idea:

1. Build a semantic notion of similarity from listener tags.
2. Train audio models to reproduce that structure from spectrograms and stems.
3. Evaluate whether the learned embeddings produce useful song retrieval.

Genre tags provides the semantic signal during training, but the long-term aim is audio-based recommendation at inference time.

## Model Design 

The system starts with a tag-based embedding built from listener-provided metadata. This acts as a semantic teacher: a reference signal for which songs should be considered similar.

The later models then learn audio embeddings from spectrograms and stems that try to match that semantic structure directly from audio. The earliest audio model is mainly focused on reproducing tag-defined similarity. Later models add InfoNCE contrastive learning to preserve both semantic similarity and audio consistency. The final model uses a late-fusion architecture trained against a blended teacher that combines tag-based similarity with a frozen audio-similarity baseline, producing a more balanced notion of recommendation quality.


![Model architecture](docs/diagrams/model_1.png)
<!-- Chat about model 2. -->

At a high level, the architecture is:

- Inputs: spectrograms and stem-based audio views
- Encoder: ResNet-style audio embedding model
- Training signal: tag-based semantic teacher, later extended with contrastive and blended objectives
- Output: fixed-length song embeddings used for nearest-neighbor retrieval



## Experimental Progression 

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

<!-- Maybe some diagrams will help here. This (I believe) should be our main metric. -->
 
