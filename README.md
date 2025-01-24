# ClassicalComposer

This is a programming challenge for SFL. A client has requested the development of a composer classifier based on live-captured audio that can stream chunks of data to the model at 15, 30, or 60-second intervals, as desired. The input to the model will be MIDI files. The specific task is to binary classifier that identifies whether a streamed MIDI file was composed by any of the following composers: **Bach, Beethoven, Schubert, Brahms** or **not**.

The **final report** is available as a Jupyter notebook in `/notebooks/overview.ipynb` or as a PDF in [/notebooks/overview.pdf](/notebooks/overview.pdf)

# Client Spec
The original client spec is available in `docs/client_spec.txt`


# Development Setup

### Building
This container may take some time to build. It is assumed you are using **VS Code** and **Dev Containers**. You can also build it manually with the following command:
```bash
docker compose --project-name classicalcomposer -f <PATH_TO_CODE>\docker-compose.yml -f <PATH_TO_CODE>\.devcontainer\docker-compose.yml -f <PATH_TO_TO_LOG> build
```

### Python Environment and Dependency Management 
This project uses **Hatch** to manage the Python environment and dependencies. For more information, see the [hatch basic usage](https://hatch.pypa.io/latest/tutorials/environment/basic-usage/).

Run the following command to execute the main program:

```bash
hatch run python main.py```
```

In order to set the python interpreter path in VSCode use the following:
```bash
hatch env create
hatch env find
```

### Configuration Management
This project uses **dyanconf** for configuration management.Please see this link for a  [quick start](https://www.dynaconf.com/)

### Scripts

| Script Name                               | Description                               |
|-------------------------------------------|-------------------------------------------|
| `hatch run python scripts/generate_dataset.py`    | Generates the dataset files |
| `hatch run python scripts/train.py`    | trains the models |
| `hatch run python scripts/eval.py`    | evaluate a midi file |



# Notebooks
To launch Jupyter Lab, run 
```bash
hatch run jupyter:lab
```
| Notebook Name                               | Description                               |
|-------------------------------------------|-------------------------------------------|
| `/notebooks/overview.ipynb`    | Overview of the project adn recommendations for next steps |
| `/notebooks/eda.ipynb`    | Original EDA/modeling for the project and initial research |
| `/notebooks/cnn.ipynb`    | Experimental CNN modeling for the project |


# Usage

This project ships with trained models. You can either use these pre-trained models or train new ones using the provided code (see the Scripts section for details on retraining).

### Running the Pipeline
To use the pipeline, run the following Docker commands:

`<Insert relevant Docker commands>`

### Web Front-End
To use the web front-end, run:

`<Insert relevant Docker commands>`
