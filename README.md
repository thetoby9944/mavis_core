![](doc/mavis/images/logo.png) 



### Environment

	conda create -n mavis python=3.7
	conda activate mavis
	pip install mavis_core
	     
### Separate Tensorflow conda install (CUDA included)
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


## Run 

	cd my_mavis_modules/
	mavis


