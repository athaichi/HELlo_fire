# HELlo_fire
DAG semester project explroring real-time path planning for fire break creation


# Running instructions
Make sure that you have the following installed: 
- numpy 
- matplotlib
- pybullet

Also make sure that conda is up and running. Installation instructions can be found at this link (https://www.anaconda.com/docs/getting-started/miniconda/install#macos-2).

Then run the project by running the following commands: 
``` 
conda activate wildfire
python3 sim.py 

```

Each pixel represents a 6ft by 6ft square of field. 
The tractor is 12 ft wide (tiller) and a combined total of 24 ft long (tiller + tractor).
We estimate it will move 10-12 mph and still till with enough too create a feasible fire break. 