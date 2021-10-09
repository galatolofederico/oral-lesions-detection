# oral-lesions-detection

Colab link: https://colab.research.google.com/drive/1jffx1pkT37xJEGaTfi8_9ScYdqjfBG_5?usp=sharing

Reports on the test set link: http://131.114.50.176/owncloud/s/zYMKHGHAe2NRTcA/download

## Installation

### Clone and install the dependencies

```
git clone https://github.com/galatolofederico/oral-lesions-detection.git
conda create -n oral-lesions-detection python=3.7
conda activate oral-lesions-detection
conda install pytorch torchvision -c pytorch-lts -c nvidia
pip install sklearn shapely opencv-python
```

### install detectron2

Check your CUDA version

```
nvcc --version
```

Install the corresponding detectron2 package

for CUDA 11.1
```
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html
```

for CUDA 10.2
```
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.8/index.html
```

### Download models and dataset

Download model and fitted PCA
```
mkdir ./models
wget http://131.114.50.176/owncloud/s/Tk6nl8Mjah2XeLW/download -O ./models/model.pth
wget http://131.114.50.176/owncloud/s/F2gPYU8HO7a4MoD/download -O ./models/pca.pkl
```

Download dataset
```
mkdir ./datasets
wget http://131.114.50.176/owncloud/s/zs7J18r37omcMmy/download -O ./datasets/lesions.zip
unzip ./datasets/lesions.zip -d ./datasets/
```

Download features
```
mkdir ./assets
wget http://131.114.50.176/owncloud/s/zqXO0hrOq2FVLPO/download -O ./assets/features.zip
unzip ./assets/features.zip -d ./assets
```

## Run Inference

### Run explain on a file

```
./explain.sh --model ./models/model.pth --pca-model ./models/pca.pkl --features-database ./assets/features/train-features.json --dataset-folder ./datasets/lesions --file datasets/lesions/images/a0b5f450f8603b13990805df60af640e.jpg
```

### Run predict on file

```
python predict-file.py --model ./models/model.pth --file datasets/lesions/images/a0b5f450f8603b13990805df60af640e.jpg
```

### Run predict on the test dataset

```
python predict-dataset.py --dataset-folder ./datasets/lesions --model ./models/model.pth
```

## Train the model

Simplify the dataset dataset into neoplastic, aphthous and traumatic

```
python -m scripts.simplify-dataset --folder ./datasets/lesions
```

Split dataset into train and test

```
python -m scripts.dataset-split --folder ./datasets/lesions
```

Run train

```
python train.py --dataset-folder ./datasets/lesions
```

To see the list of the hyperparameters run

```
python train.py --help
```

## Scripts

### simplify-dataset
Simplify the dataset into 3 classes (neoplastic, aphthous, traumatic)

```
python -m scripts.simplify-dataset --folder ./datasets/lesions
```

### datasets-split
Split the dataset into train and test

```
python -m scripts.dataset-split --folder ./datasets/lesions
```

### view-dataset
View a dataset

```
python -m scripts.view-dataset --dataset-folder ./datasets/lesions
```

### view-augmentation
View an augmented dataset

```
python -m scripts.view-augmentation --dataset-folder ./datasets/lesions
```

### extract-tsne-features
Extract features for t-SNE plot 

```
python -m scripts.extract-tsne-features --dataset-folder ./datasets/lesions/ --dataset ./datasets/lesions/dataset.json --model ./models/model.pth  --output ./assets/tsne-features.pkl
```

### tsne
Plot the t-SNE map

```
python -m scripts.tsne --features ./assets/tsne-features.pkl
```

### build-features-database
Extract features from all the dataset images (used in explain)

```
python -m scripts.build-features-database --dataset-folder ./datasets/lesions --model ./models/model.pth --output ./assets/extracted-features
```

### fit-pca
Fit the PCA used in explain

```
python -m scripts.fit-pca --features-database assets/features/features.json  --output models/fitted_pca.pkl
```

## Run Hyperparameter Optimization

Set the environment variable `OPTUNA_STORAGE` to your preferred optuna storage (available storages [here](https://docs.sqlalchemy.org/en/14/core/engines.html#sqlalchemy.create_engine))

```
export OPTUNA_STORAGE=sqlite:///optuna.sqlite
```

Install optuna

```
pip install optuna
```

Create study

```
python create-study.py --study-name <your_study_name>
```

Run trial

```
python run-trial.py --study-name <your_study_name>
```