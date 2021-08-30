![](doc/mavis/images/logo.png) 


## Docker

    docker-compose up
    
Or with directly

    docker run -p 8501:8501 thetoby/mavis


For the full command to add
- GPU support for tensorflow, 
- a persistent data location for mavis, 
- a path on the host system that should be available in the container 
    
    
    docker run \
        --gpus all \
        - ~/Projects/mavis-git/mavis/src/pipelines:/usr/mavis/src/pipelines  # Addons Location
        -p 8501:8501 \
        thetoby/mavis

---

## From Source 


### Environment

	conda create -n mavis python=3.7
	conda activate mavis
	conda install pip
	
For Polygon Rendering:

    sudo apt-get install cairo
     
### Tensorflow conda install (CUDA included)
(01. JAN. 21.)

-  Linux (CUDA GPU):
    
    
    # To get cudnn + cuda 11.0 from conda
	conda install tensorflow==2.1 
	# To to use the required tensorflow veriosn
	pip install tensorflow==2.3.1 
	
- Windows (CUDA GPU):

    
    conda install tensorflow==2.3

- Any CPU


	conda install tensorflow==2.3

### Final Steps

	pip install -r requirements.txt


## Run from Source

	cd src/
	conda activate mavis
	streamlit run app.py


