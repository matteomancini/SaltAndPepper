# Salt and Pepper (AKA: series and pictures) - A general structure for deep learning on signals/spectra with PyTorch

This project defines the fundamental structure for work-in-progress deep learning projects aimed at classifying signals by means of the associated time series or the related spectral maps. All the packages required are listed in the `requirements.txt` file.

## Running the examples

To run the basic train procedure, one can use either the script `train.py` or the notebook `train.ipynb`. So far the only model implemented is a fully connected network with ReLU activations - the number of hidden layers and units can be changed. The datasets included so far are `cinc2017` and `ecg_sample` and they are automatically downloaded when executing the script/notebook.

## Adding a new dataset

To add a new dataset that can be easily loaded, one needs to create a new folder under `datasets` with the same name as the dataset, add a `__init__.py` file and implement the following methods:

- `download_data()` - needed only if the data need to be downloaded;
- `load_data()` - this method is required and details how to load the actual data, it is supposed to return the series/maps and the associated labels;
- `get_label_names()` - this method should return the actual class names associated with the labels.

## To-do list

- comments;
- data class for spectral maps;
- more models;
- more plots;
- ...