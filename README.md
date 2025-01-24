# ClassicalComposer

This is a programming challenge for SFL. A client has requested us to build a composer classifier based on live captured audio that can be set to stream chunks of data to our model at 15, 30, 60-second intervals, as desired. The input to the model will be midi files. The specific request is to identify if a new midi file is composed by any of (Bach, Beethoven, Schubert, and Brahms) or by anyone else. 


 
Goal:  
The goal of this project is to develop a classifier/pipeline that is able to determine which midi files in the provided PS2 folder are not written by the four (4) composers above (it is a small number).  

# Building
This container can take some time to build, it is assumed you are using VSCode and devcontainers. 
Can be built manually with 
```
docker compose --project-name classicalcomposer -f <PATH_TO_CODE>\docker-compose.yml -f <PATH_TO_CODE>\.devcontainer\docker-compose.yml -f <PATH_TO_TO_LOG> build
```

# Setup

This project uses hatch to mange the python env and dependencies [hatch basic usage](https://hatch.pypa.io/latest/tutorials/environment/basic-usage/)

```hatch run python main.py```

# Development setup
You will need to confirgure your git credentials on the host system if using dev-containers
set python interpreter to this path
```
hatch env create
hatch env find
```

Be sure to rebuild the dev containers after updating .env variables or dependencies or strange errors can occur.
# Scripts

Run the following to setup the needed resources
| Script Name                               | Description                               |
|-------------------------------------------|-------------------------------------------|
| `hatch run python scripts/generate_dataet.py`    | creates the dataset files |



# Notebooks
To launch jupyter, run ```hatch run jupyter lab```
| Notebook Name                               | Description                               |
|-------------------------------------------|-------------------------------------------|
| `python notebooks/overview.ipynb`    | Default |

# Front End
To start a front end client for chatting, open [src/chat/index.html?handle=ASH0001&channel=CH_ASH0001](file:///C:/code/healiom/src/healiom_agent/chat/index.html?handle=ASH0001&channel=CH_ASH0001). NOTE The query string parameters are needed and hard coded at this point.
