# Gravitational-Wave Signal Classification with a CNN

This project uses a convolutional neural network (CNN) to classify gravitational-wave time series as either signals or noise. The model works with spectrograms generated from simulated data and is trained using PyTorch.

The main notebook is:

`Gravitational_waves_detection.ipynb`

## Project structure

The notebook contains the following main steps:

1. Installation and imports
2. Definition of global parameters
3. Gravitational-wave signal generation
4. Gaussian detector noise and instrumental glitch generation
5. Spectrogram generation and visualization
6. Training, validation, and test dataset creation
7. CNN architecture definition
8. Model training with early stopping
9. Learning-curve analysis
10. ROC curve and confusion-matrix evaluation
11. Accuracy as a function of signal-to-noise ratio (SNR)
12. Model saving and loading
13. Generalization tests with new physical parameters
14. Generalization tests using different signal morphologies
15. Comparison with matched filtering under ideal and realistic conditions

## Method

Gravitational-wave signals are generated using **PyCBC** when it is available. The notebook uses physical waveform approximants including:

- `IMRPhenomD` for binary-black-hole and neutron-star–black-hole systems
- `TaylorT4` for binary-neutron-star systems

The simulated signals are combined with Gaussian detector noise and, depending on the configuration, instrumental glitches.

Each time series is converted into a logarithmic spectrogram using a short-time Fourier transform (STFT). These spectrograms are used as the input to the CNN.

The network is implemented in **PyTorch** and consists of convolutional layers with batch normalization and max pooling, followed by global average pooling and fully connected layers with dropout. The final layer uses a sigmoid activation to produce the probability that an input contains a gravitational-wave signal.

## Dataset

The dataset is generated directly in the notebook. Signal and noise examples are sampled from the parameter ranges defined in `PARAMS`.

The configurable parameters include:

- Signal-to-noise ratio
- Source distance
- Lower frequency cutoff
- Component masses
- Aligned spin components
- Noise amplitude
- Glitch probability and parameters
- Signal duration
- Spectrogram parameters

The generated dataset is divided into training, validation, and test sets using stratified splitting.

## Evaluation

The notebook evaluates the trained CNN using:

- Training and validation loss
- Training and validation accuracy
- ROC curve and AUC
- Confusion matrix
- Accuracy as a function of SNR
- Tests with parameter values outside the original training configuration
- Tests using linear and quadratic chirp morphologies

The notebook also includes a comparison with matched filtering in two regimes:

- An idealized case with a fixed waveform and Gaussian white noise
- A more realistic case including parameter variation and instrumental glitches

The matched-filtering comparison is intended as a complementary analysis rather than a strictly equivalent benchmark.

## Reproducibility

A random seed of `42` is used for NumPy, Python, and PyTorch random-number generation.

The notebook can use a GPU when one is available through CUDA or Apple's MPS backend; otherwise it falls back to the CPU.

## Requirements

The main dependencies are listed in `requirements.txt`.

To install them with pip:

```bash
pip install -r requirements.txt
```

The notebook also contains an installation cell suitable for use in Google Colab.

## Running the notebook

The notebook was designed to be run interactively, including in Google Colab.

After installing the dependencies, open:

`Gravitational_waves_detection.ipynb`

and run the cells in order.

Generating the complete dataset can take considerably longer than running the demonstration cells, depending on the selected number of samples and whether a GPU is available.

## Project context

This work was developed as **Project 19 — Artificial Intelligence in Physics**, focusing on the application of neural networks to gravitational-wave signal detection in noisy data.
