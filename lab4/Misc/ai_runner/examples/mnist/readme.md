---
title:  MNIST example
---

# Overview

This is a typical example to use the ST AI runner Python module with the user data against
the generated c-model to perform a advanced validation flow.

`train.py` allows to generate and train a Keras model with the MNIST data set. The model
is also quantized using TFLite converter with a part of the data-set.

`test.py` demonstrates how to test the generate c-model (X86 or STM32 implementation)
with the MNIST test data set. Classification report is based
on sklearn.metrics function.


# How-to

- Create and train the Keras model

```bash
python train.py
```

Generated model files: `mnist_fp32.h5`, `mnist_q_with_fp32_io.tflite`


- Generate and validate the c-model with STM AI CLI

Standard approach with random data based on the numerical comparaison between the
execution of the original model and the generated c-model. 

```bash
stm32ai validate -m mnist_q_with_fp32_io.tflite --split-weights
```

Note that the `--split-weights` is not mandatory to validate the generated model. This
is currentlty a work-around to generate a X86 DLL (in default `stm32ai_ws` directory)
including the weights.

To perform also a validation on the board, a STM32 board should be also flashed with
the aiValidation test application including the model.


- Report the accuracy of the generated c-model and the MNIST test set

With the X86 library

```bash
python test.py -d stm32ai_ws
```


With the STM32 board (first valid serial COM port is used)

```bash
python test.py -d serial
```

To indicate a specific serial COM port, following command can be used:

```bash
python test.py -d serial:COMx             # Windows world
python test.py -d serial:/dev/ttyACMx     # Linux/MacOS world
```



