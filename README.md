# Deep learning for element identification using Raman spectrum

This repository proposes attention-based residual networks for element identification of minerals using Raman spectrum. Deep learning (DL) for Raman spectrum has been adopted for mineral classification (Liu et al., 2017; Zhang et al., 2020). On this basis, this repository aims to identify specific elements in minerals or other inorganic materials by DL-based Raman spectroscopy. 

## Requirements

The code has been tested running under Python 3.9.12, with the following packages and their dependencies installed:

```
numpy==1.16.5
pandas==1.4.2
pytorch==1.7.1
sklearn==0.21.3
```

## Usage

```bash
python main.py
```

Users can define the elements to be identified by modifying list `elem`.

## Datasets

Following previous research (Zhang et al., 2020) for mineral classification, this repo validates the models on the excellent oriented Raman spectra data of RRUFF dataset. To fix the dimension of different spectra, each spectrum is interpolated in 100-1400cm-1 with 1024 points. The spectra data of the minerals with only one sample per mineral class are removed.

The data can be downloaded via this [link](https://rruff.info/zipped_data_files/raman/excellent_unoriented.zip). The Raman spectra data are acquired by spectrometers with different excitation wavelength (532nm, 780nm, ...). Besides, The dataset includes the raw data and the data processed by baseline removal and cosmicray removal. This repo provides options for users to adopt processed data or raw data with different excitation wavelength.

```
parser.add_argument('--wavelength', type=str, default='780',
                    help='Excitation wavelength of data')
parser.add_argument('--raw', type=str, default='Processed',
                    help='Processed/RAW')  
```

Function `preprocess(args.raw,args.wavelength)` copies the selected data to folder `./RRUFF_data`. Then, the data can be accessed by function `read_data('./RRUFF_data')`. For convience, the data selected by default setting is saved in `780-process.csv`, and can be accessed by reading the CSV instead of reading all files in data folder.

## Results

The identification AUC ranges from 0.75 to 0.95 on different elements. Most elements can achieve the AUC more than 0.8.

## Options

We adopt an argument parser by package  `argparse` in Python, and the options for running code are defined as follow:

```python
parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=True,
                    help='CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate.')
parser.add_argument('--wd', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Dimension of representations')
parser.add_argument('--c', type=int, default=2,
                    help='Num of classes')
parser.add_argument('--d', type=int, default=1024,
                    help='Num of spectra dimension')
parser.add_argument('--model', type=str, default='NN',
                    help='Model')
parser.add_argument('--wavelength', type=str, default='780',
                    help='Excitation wavelength of data')
parser.add_argument('--raw', type=str, default='Processed',
                    help='Processed/RAW')                     

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()
```

## References

Liu et al. Deep convolutional neural networks for raman spectrum recognition: a unified solution. Analyst. 2017

Zhang et al. Tranfer-learning-based Raman spectra identification. J Raman Spectrosc. 2020