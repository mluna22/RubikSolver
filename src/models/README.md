# Models

This directory contains models trained and used for evaluation in our experiments.

| Problem        | `filename`       | Steps   |
| :------------- | :--------------- | :------ |
| Rubik's Cube   | `cube3.pth`      | 2000000 |

These models are provided in [TorchScript](https://pytorch.org/docs/stable/jit.html) format, which can be loaded using `torch.jit.load(filename)`.
Ensure you have `torch>=1.12` installed to use these models.
