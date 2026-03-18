To install repo src: inside the repo root on terminal,
1. Activate conda environment
2. Run `pip install --upgrade pip`
3. Run `pip install -e .`

## Architecture

<span style="color:red">(We should change model names.)</span>

Unlike many recommender systems that are content-agnostic and rely mainly on user behavior, this project learns similarity from the music itself. It first builds a tag embedding from listener-provided metadata, using those tags as a semantic teacher that defines a prior notion of song similarity. From there, later models learn embeddings from spectrograms and stems that try to reproduce that semantic structure directly from audio. Model 4 is the most tag-focused version, Models 5 and 6 add InfoNCE contrastive learning to preserve both semantic and audio relationships, and Model 7 extends this progression with a late-fusion architecture trained against a blended teacher that combines tag-based similarity with a frozen audio-similarity baseline.
