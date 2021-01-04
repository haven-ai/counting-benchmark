# Build a Counting Benchmark in 3 minutes

## Usage

```
pip install git+https://github.com/ElementAI/LCFCN
```


## Experiments

### 1. Install dependencies

```
pip install -r requirements.txt
```
This command installs pydicom and the [Haven library](https://github.com/haven-ai/haven-ai) which helps in managing the experiments.


### 2. Download Datasets

- Trancos Dataset 
  ```
  wget http://agamenon.tsc.uah.es/Personales/rlopez/data/trancos/TRANCOS_v3.tar.gz
  ```

### 3. Train and Validate
```
python trainval.py -e {EXP_GROUP} -d {DATADIR} -sb {SAVEDIR_BASE} -r 1
```

- `{DATADIR}` is where the dataset is located.
- `{SAVEDIR_BASE}` is where the experiment weights and results will be saved.
- `{EXP_GROUP}` specifies the exp_group such as `trancos` training hyper-parameters defined in [`exp_configs.py`](exp_configs.py).

###  4. View Results

```
> jupyter nbextension enable --py widgetsnbextension --sys-prefix
> jupyter notebook
```

