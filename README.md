# Build a Counting Benchmark in 3 minutes

### 1. Install dependencies

```
pip install -r requirements.txt
pip install git+https://github.com/ElementAI/LCFCN
```
This command installs pydicom and the [Haven library](https://github.com/haven-ai/haven-ai) which helps in managing the experiments.


### 2. Download Datasets

- Trancos Dataset 
  ```
  wget http://agamenon.tsc.uah.es/Personales/rlopez/data/trancos/TRANCOS_v3.tar.gz
  ```

### 3. Define Hyperparameters

```
EXP_GROUPS['trancos'] =  {"dataset": {'name':'trancos', 
                          'transform':'rgb_normalize'},
         "model": {'name':'lcfcn','base':"fcn8_vgg16"},
         "batch_size": [1,5,10],
         "max_epoch": [100],
         'dataset_size': [
                          {'train':'all', 'val':'all'},
                          ],
         'optimizer':['adam'],
         'lr':[1e-5]
         }
```

### 4. Train and Validate
```
python trainval.py -e {EXP_GROUP} -d {DATADIR} -sb {SAVEDIR_BASE} -r 1
```

- `{DATADIR}` is where the dataset is located.
- `{SAVEDIR_BASE}` is where the experiment weights and results will be saved.
- `{EXP_GROUP}` specifies the exp_group such as `trancos` training hyper-parameters defined in [`exp_configs.py`](exp_configs.py).

###  5. View Results

```
> jupyter nbextension enable --py widgetsnbextension --sys-prefix
> jupyter notebook
```

