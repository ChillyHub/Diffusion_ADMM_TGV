# Diffusion_ADMM_TGV


### Requirement
Make a conda environment and install dependencies
```bash
conda env create --file environment.yml
conda activate diff_tgv_env
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```


### Download the data
* **CT** experiments (in-distribution)
```bash
DATA_DIR=./data/CT/ind/256_sorted
mkdir -p "$DATA_DIR"
wget -O "$DATA_DIR"/256_sorted.zip https://www.dropbox.com/sh/ibjpgo5seksjera/AADlhYqCWq5C4K0uWSrCL_JUa?dl=1
unzip -d "$DATA_DIR"/ "$DATA_DIR"/256_sorted.zip
```
* **CT** experiments (out-of-distribution)
```bash
DATA_DIR=./data/CT/ood/256_sorted
mkdir -p "$DATA_DIR"
wget -O "$DATA_DIR"/slice.zip https://www.dropbox.com/s/h3drrlx0pvutyoi/slice.zip?dl=0
unzip -d "$DATA_DIR"/ "$DATA_DIR"/slice.zip
```

### Download pre-trained model weights
* **CT** experiments
```bash
mkdir -p exp/ve/AAPM_256_ncsnpp_continuous
wget -O exp/ve/AAPM_256_ncsnpp_continuous/checkpoint_185.pth https://www.dropbox.com/s/7zevc3eu8xkqx0x/checkpoint_185.pth?dl=1
```


### Reconstruction

```bash
conda activate diff_tgv_env
python experience.py
```

or set parameter self

```bash
conda activate diffusion-mbir
python generator_AAPM_3d_sparseview.py
```

### Analyzing result

Look file analyzing_tool.py
