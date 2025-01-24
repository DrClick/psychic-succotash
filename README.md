# ClassicalComposer

This is a programming challenge for SFL. A client has requested us to build a composer classifier based on live captured audio that can be set to stream chunks of data to our model at 15, 30, 60-second intervals, as desired. The input to the model will be midi files. The specific request is to identify if a streamed midi file is composed by any of (Bach, Beethoven, Schubert, and Brahms) or by anyone else. 

The final report is a jupyter notebook available in notebooks/overview.ipynb (see the notebooks section for a list of all available notebooks)

# Client Spec
The original client spec is available in docs/client_spec.txt

# Building
This container can take some time to build, it is assumed you are using VSCode and devcontainers. 
Can be built manually with 
```
docker compose --project-name classicalcomposer -f <PATH_TO_CODE>\docker-compose.yml -f <PATH_TO_CODE>\.devcontainer\docker-compose.yml -f <PATH_TO_TO_LOG> build
```

# Setup

This project uses hatch to mange the python env and dependencies [hatch basic usage](https://hatch.pypa.io/latest/tutorials/environment/basic-usage/)

```hatch run python main.py```

You will need to confirgure your git credentials on the host system if using dev-containers
set python interpreter to this path
```
hatch env create
hatch env find
```

Be sure to rebuild the dev containers after updating .env variables or dependencies or strange errors can occur.
# Scripts

| Script Name                               | Description                               |
|-------------------------------------------|-------------------------------------------|
| `hatch run python scripts/generate_dataset.py`    | creates the dataset files |
| `hatch run python scripts/train.py`    | trains the models |
| `hatch run python scripts/eval.py`    | evaluate a model |



# Notebooks
To launch jupyter, run ```hatch run jupyter lab```
| Notebook Name                               | Description                               |
|-------------------------------------------|-------------------------------------------|
| `python notebooks/overview.ipynb`    | Default |


# Usage

This project ships with trained models, you can either use those or use the code to train a new model. See the scripts section for usage on retraining.
To use the pipeline 

`docker commands here `

or as a web front end

`docker commands here `
