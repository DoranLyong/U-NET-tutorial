## Data preparation
1. Download a dataset from here -> [link](https://www.kaggle.com/c/carvana-image-masking-challenge/data)
2. Extract the download file into ```data``` directory. 
3. Then, run the ```data_split.py``` in order to split the dataset into 'train_set' and 'val_set'; (8:2 ratio). 
    ```bash 
    ~$ python data_split.py
    ```
4. For checking the dataset with visualization (ongoing)
    ```bash
    ~$ python data_vis.py
    ```

<br/>

## Set up your hyperparameters 
* check ```cfg.yaml``` if you need to change your directory path. 
* you can edit the hyperparameter setup for training.

<br/>

## Run training 
```bash
~$ python train.py 
