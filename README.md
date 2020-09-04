# GAN
1. This package assumes that you have installed tesnorflow 2.1.0 already and have required CUDa and other things already setup on your device.
2. To install other required libraries download the package and change directory to this folder and then  :
```bash 
conda env update -f myenv.yml
 ``` 
3. To run GAN.py, open command prompt and run :
```bash 
python GAN.py --path <enter the directory path where you want to save image generated>
 ``` 
  
 # If you don't have tensorflow installed.
 
 
 #### Conda (Recommended)

```bash

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov3-tf2-gpu
```


#### Pip

```bash
pip install -r requirements.txt
```

#### Installation
 ```bash 
python GAN.py --path <enter the directory path where you want to save image generated>
 ``` 
