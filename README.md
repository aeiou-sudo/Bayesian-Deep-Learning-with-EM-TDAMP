# EM-TDAMP for Bayesian Deep Learning

This repository contains an implementation of the Expectation Maximization with Turbo Deep Approximate Message Passing (EM-TDAMP) algorithm for Bayesian deep learning, as described in the paper "Bayesian Deep Learning via Expectation Maximization and Turbo Deep Approximate Message Passing."

## Overview

EM-TDAMP combines expectation maximization with approximate message passing techniques to perform Bayesian inference in deep neural networks. The key advantages of this approach include:

1. Principled uncertainty estimation
2. Automatic network pruning through structured sparsity
3. Faster convergence compared to conventional training methods

This implementation demonstrates the algorithm on the MNIST handwritten digit classification task.

## Requirements

See `requirements.txt` for the necessary Python dependencies.

## Usage

```bash
python train_mnist_tdamp.py --batch_size 100 --em_iters 30 --sparsity 0.5
```

### Command Line Arguments

- `--batch_size`: Mini-batch size for training (default: 100)
- `--hidden_dim`: Number of hidden units (default: 128)
- `--em_iters`: Number of EM iterations (default: 30)
- `--sparsity`: Target sparsity level (default: 0.5)
- `--seed`: Random seed for reproducibility (default: 42)
- `--output_dir`: Directory to save results (default: "./results")
- `--save_model`: Save the final model (flag)
- `--train_standard`: Also train a standard MLP for comparison (flag)

## Algorithm Details

The EM-TDAMP algorithm consists of two main modules:

1. **Module A (DAMP)**: Deep Approximate Message Passing for computing posterior distributions of weights.
2. **Module B (SPMP)**: Sum-Product Message Passing for enforcing structured sparsity using spike-and-slab priors.

These modules work together in an iterative expectation-maximization framework:

- **E-step**: Update posterior distributions of weights using approximate message passing.
- **M-step**: Update hyperparameters (sparsity parameters and noise variance).

## Implementation Notes

- The implementation uses TensorFlow for automatic differentiation and GPU acceleration.
- The forward pass returns both the pre-activation logits and intermediate activations needed for message updates.
- The probit-product likelihood is implemented for robust classification.
- Numerical stability is ensured through careful implementation of the Q-function using `erfc`.
- The PasP (Posterior as Prior) update correctly accumulates information across mini-batches.
- Vectorized operations are used wherever possible for efficient computation.

## Results

After training, the script will generate several plots in the output directory:

1. Accuracy vs. EM iteration
2. Sparsity vs. EM iteration
3. Noise variance vs. EM iteration
4. Confusion matrix on test set
5. Sparsity vs. Test accuracy
6. Per-class uncertainty visualization
7. Neuron activation probabilities

## References

The implementation is based on the paper "Bayesian Deep Learning via Expectation Maximization and Turbo Deep Approximate Message Passing."