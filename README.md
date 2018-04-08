# Movie-Recommendation
A Full Scale Movie Recommendation Engine In Python And Numpy

(Note: This project makes use of `Git LFS` for managing large datasets. So viewing the files in the browser wouldn't be possible.)

## Installation
For a one stop solution, I always use `conda`. First install `conda` and create you own environment as:
```
conda create --name myenv
```
where `myenv` is the environment name (which can be changed at will). Then activate the environment by:
```
source activate myenv
```
and then install the dependencies using the following command:
```
conda install -c conda-forge matplotlib numpy pandas theano
```

## Running Tests
You need to clean the given dataset. For `MovieLens 10M` dataset, you can just run the following command in the `src` folder:
```
python3 preprocess.py
```
which would create a dense dataset file named `ratings_cleaned.csv`, and the training and test sets as `train.csv` and `test.csv` in the `MovieLens 10M Dataset` folder.

You can then execute the project using the following command:
```
python3 main.py
```
All the plots would be saved under `plots` folder. Additionally, you can change the parameter and the test conditions for various tests in `main.py` file to suit your needs. Be cautioned, many of the test conditions implemented would take days to finish on an average machine. Most of the plots shown were computed using `Google Cloud Platform`.

## Notes
The official `Netflix Prize` dataset hasn't been uploaded for the size related reasons, but can be downloaded from [here](https://www.kaggle.com/netflix-inc/netflix-prize-data). It should be noted that the dataset should work out of the box, provided the above mentioned instructions are followed. Lastly, for testing on any new dataset, please ensure that the dataset is structured in a similar manner as the `ratings.csv` file provided.