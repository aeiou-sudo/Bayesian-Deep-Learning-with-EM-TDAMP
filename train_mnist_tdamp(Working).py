import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import argparse
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
with tf.device('/GPU:0'):
    a = tf.constant([1.0])
    b = tf.constant([2.0])
    c = a + b
    logger.info("GPU test result: %s", c.numpy())
# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='EM-TDAMP for MNIST Classification')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size for training')
parser.add_argument('--em_iters', type=int, default=30, help='Number of EM iterations')
parser.add_argument('--sparsity', type=float, default=0.5, help='Target sparsity level')
args = parser.parse_args()

# 1. Project Setup & Data
def load_and_preprocess_data():
    """Load and preprocess a subset of the MNIST dataset."""
    logger.info("Loading and preprocessing MNIST dataset")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Subsample training data
    x_train = x_train[:10000] / 255.0  # Take first 10,000 samples
    y_train = y_train[:10000]
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784) / 255.0
    
    # One-hot encode labels
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    
    logger.info("MNIST dataset loaded: %d training samples, %d test samples", x_train.shape[0], x_test.shape[0])
    return train_dataset, test_dataset, x_train, y_train, x_test, y_test

# 2. Model Definition
def build_model():
    """Define MLP model with access to weights and activations."""
    logger.info("Building MLP model: 784 -> 128 (ReLU) -> 10")
    inputs = keras.Input(shape=(784,))
    hidden = keras.layers.Dense(128, activation='relu', name='hidden')(inputs)
    outputs = keras.layers.Dense(10, name='outputs')(hidden)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# 3. Probit-Product Likelihood
def probit_product_likelihood(logits, y_true, v):
    """Compute log probit-product likelihood for classification using vectorized operations."""
    y_idx = tf.argmax(y_true, axis=1)  # Shape: [batch_size]
    y_idx = tf.cast(y_idx, tf.int32)  # Ensure int32 dtype to match tf.range
    zi = logits  # Shape: [batch_size, num_classes]
    yi = tf.gather(zi, y_idx, batch_dims=1)  # Shape: [batch_size]
    diff = (zi - tf.expand_dims(yi, axis=1)) / tf.sqrt(tf.cast(v, tf.float32))  # Shape: [batch_size, num_classes]
    q_values = 0.5 * tf.math.erfc(diff / tf.sqrt(2.0))  # Shape: [batch_size, num_classes]
    indices = tf.stack([tf.range(tf.shape(logits)[0]), y_idx], axis=1)  # Shape: [batch_size, 2]
    q_values = tf.tensor_scatter_nd_update(q_values, indices, tf.ones([tf.shape(logits)[0]], dtype=tf.float32))
    log_q = tf.math.log(q_values + 1e-10)  # Shape: [batch_size, num_classes]
    log_likelihood = tf.reduce_sum(log_q, axis=1)  # Shape: [batch_size]
    return log_likelihood

# 4. Turbo Deep Approximate Message-Passing (TDAMP)
class TDAMP:
    def __init__(self, model, num_groups, v_init=1.0):
        """Initialize TDAMP with model weights and Bayesian parameters."""
        logger.info("Initializing TDAMP with %d groups, initial v=%.2f", num_groups, v_init)
        self.model = model
        self.num_groups = num_groups  # Number of neurons in hidden layer
        self.v = tf.Variable(v_init, dtype=tf.float32, trainable=False)  # Noise variance
        self.rho = tf.Variable(tf.ones([num_groups], dtype=tf.float32) * 0.5, trainable=False)  # Sparsity parameters
        # Initialize weight means and variances as Variables
        self.weight_means = [tf.Variable(w, trainable=False) for w in model.get_weights()]
        self.weight_variances = [tf.Variable(tf.ones_like(w) * 0.1, trainable=False) for w in model.get_weights()]
    
    def module_a_damp(self, x, y, priors):
        """Module A: Deep Approximate Message Passing (simplified)."""
        # Unpack priors (means and variances)
        prior_means, prior_variances = priors
        
        # Simplified DAMP: Sample weights from Gaussian priors and compute posteriors
        sampled_weights = []
        for mean, var in zip(prior_means, prior_variances):
            # Sample weights from N(mean, var)
            noise = tf.random.normal(tf.shape(mean), mean=0.0, stddev=tf.sqrt(var))
            sampled_weight = mean + noise
            sampled_weights.append(sampled_weight)
        
        # Set model weights to sampled weights for forward pass
        original_weights = self.model.get_weights()
        self.model.set_weights(sampled_weights)
        
        # Compute logits and likelihood
        with tf.GradientTape() as tape:
            logits = self.model(x, training=False)
            log_likelihood = probit_product_likelihood(logits, y, self.v)
            loss = -tf.reduce_mean(log_likelihood)
        
        # Compute gradients w.r.t. sampled weights
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Update posterior means and variances (simplified: gradient-based update)
        posterior_means = []
        posterior_variances = []
        for i, (mean, var, grad) in enumerate(zip(prior_means, prior_variances, gradients)):
            if grad is None:
                # Handle case where gradient is None (e.g., for biases if not used)
                posterior_means.append(mean)
                posterior_variances.append(var)
                continue
            # Update mean: Move in direction of negative gradient
            learning_rate = 0.01
            new_mean = mean - learning_rate * grad
            # Update variance: Reduce variance to simulate posterior tightening
            new_var = var * 0.95
            posterior_means.append(tf.Variable(new_mean, trainable=False))
            posterior_variances.append(tf.Variable(new_var, trainable=False))
        
        # Restore original weights
        self.model.set_weights(original_weights)
        
        # Update internal weight means and variances
        for i in range(len(self.weight_means)):
            self.weight_means[i].assign(posterior_means[i])
            self.weight_variances[i].assign(posterior_variances[i])
        
        return posterior_means, posterior_variances, loss
    
    def module_b_spmp(self, posteriors):
        """Module B: Sum-Product Message Passing for group sparse priors (simplified)."""
        posterior_means, posterior_variances = posteriors
        
        # Simplified SPMP: Update sparsity (rho) based on posterior means
        # For hidden layer weights (first weight matrix), compute group norms
        hidden_weights_mean = posterior_means[0]  # Shape: [784, 128]
        group_norms = tf.reduce_mean(tf.abs(hidden_weights_mean), axis=0)  # Shape: [128]
        
        # Update rho: Higher group norm -> higher probability of being active
        new_rho = tf.sigmoid(group_norms * 10.0)  # Scale for sharper probabilities
        self.rho.assign(tf.clip_by_value(new_rho, 0.0, 1.0))
        
        # Compute extrinsic messages (simplified: scale posteriors)
        extrinsic_means = [m * 0.99 for m in posterior_means]
        extrinsic_variances = [v * 0.99 for v in posterior_variances]
        
        return extrinsic_means, extrinsic_variances, self.rho
    
    def step(self, x, y, batch_idx):
        """Perform one TDAMP iteration."""
        # Module B -> Module A: Compute priors
        priors = (self.weight_means, self.weight_variances)
        
        # Module A: DAMP to compute posteriors
        posterior_means, posterior_variances, loss = self.module_a_damp(x, y, priors)
        posteriors = (posterior_means, posterior_variances)
        
        # Module A -> Module B: Compute extrinsics
        extrinsics, new_rho = self.module_b_spmp(posteriors)[:2]
        
        # Update parameters (PasP: Posterior as Prior)
        self.weight_means = [tf.Variable(m, trainable=False) for m in posteriors[0]]
        self.weight_variances = posteriors[1]
        
        # Log mini-batch progress (every 100 batches to avoid clutter)
        if batch_idx % 100 == 0:
            avg_rho = tf.reduce_mean(self.rho).numpy()
            logger.info("Batch %d: Loss = %.4f, Average rho = %.4f", batch_idx, loss, avg_rho)
        
        return posteriors

# 5. EM Loop
def em_loop(model, train_dataset, test_dataset, em_iters, sparsity_target):
    """Run EM-TDAMP training loop."""
    logger.info("Starting EM-TDAMP training for %d iterations with target sparsity %.2f", em_iters, sparsity_target)
    tdamp = TDAMP(model, num_groups=128, v_init=1.0)
    history = {'val_accuracy': [], 'sparsity': []}

    for em_iter in range(em_iters):
        logger.info("EM Iteration %d/%d", em_iter + 1, em_iters)
        batch_idx = 0
        total_loss = 0.0
        num_batches = 0
        
        # E-step
        for x_batch, y_batch in train_dataset:
            posteriors = tdamp.step(x_batch, y_batch, batch_idx)
            batch_idx += 1
            num_batches += 1
            total_loss += -tf.reduce_mean(probit_product_likelihood(model(x_batch, training=False), y_batch, tdamp.v)).numpy()
        
        # M-step
        # Update noise variance v (simplified: reduce slightly)
        tdamp.v.assign(tdamp.v * 0.99)
        
        # Update sparsity parameters with thresholding
        threshold = np.percentile(tdamp.rho.numpy(), 100 * (1 - sparsity_target))  # Use NumPy percentile
        threshold = tf.cast(threshold, tf.float32)  # Cast to TensorFlow float32
        tdamp.rho.assign(tf.where(tdamp.rho > threshold, tdamp.rho, 0.0))
        
        # Evaluate on test set
        accuracy = evaluate_model(model, test_dataset, tdamp.weight_means)
        sparsity = compute_sparsity(tdamp.rho)
        avg_loss = total_loss / num_batches
        active_neurons = tf.reduce_sum(tf.cast(tdamp.rho > 0, tf.float32)).numpy()
        
        history['val_accuracy'].append(float(accuracy))
        history['sparsity'].append(float(sparsity))
        
        # Log iteration summary
        logger.info(
            "Val Accuracy = %.4f, Sparsity = %.4f, Avg Loss = %.4f, Noise Variance = %.4f, Active Neurons = %d/%d",
            accuracy, sparsity, avg_loss, tdamp.v.numpy(), int(active_neurons), tdamp.num_groups
        )
    
    logger.info("Training completed. Final Val Accuracy = %.4f, Final Sparsity = %.4f", 
                history['val_accuracy'][-1], history['sparsity'][-1])
    return history, tdamp

# 6. Evaluation and Visualization
def evaluate_model(model, dataset, weight_means):
    """Evaluate model accuracy using posterior means."""
    original_weights = model.get_weights()
    model.set_weights(weight_means)
    correct = 0
    total = 0
    for x_batch, y_batch in dataset:
        logits = model(x_batch, training=False)
        predictions = tf.argmax(logits, axis=1)
        labels = tf.argmax(y_batch, axis=1)
        correct += tf.reduce_sum(tf.cast(predictions == labels, tf.float32))
        total += x_batch.shape[0]
    model.set_weights(original_weights)
    return correct / total

def compute_sparsity(rho):
    """Compute sparsity as fraction of active neurons."""
    return 1.0 - tf.reduce_mean(tf.cast(rho > 0, tf.float32))

def plot_results(history, x_test, y_test, weight_means, model):
    """Generate required plots."""
    logger.info("Generating visualization plots")
    # Accuracy vs. EM Iteration
    plt.figure(figsize=(8, 6))
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('EM Iteration')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. EM Iteration')
    plt.legend()
    plt.savefig('accuracy_vs_iteration.png')
    plt.close()
    
    # Sparsity vs. Test Accuracy
    plt.figure(figsize=(8, 6))
    plt.scatter(history['sparsity'], history['val_accuracy'], c='blue')
    plt.xlabel('Sparsity (%)')
    plt.ylabel('Test Accuracy')
    plt.title('Sparsity vs. Test Accuracy')
    plt.savefig('sparsity_vs_accuracy.png')
    plt.close()
    
    # Per-class uncertainty
    model.set_weights(weight_means)
    logits = model(x_test, training=False)
    probs = tf.nn.softmax(logits)
    variances = tf.math.reduce_variance(probs, axis=0)
    plt.figure(figsize=(8, 6))
    plt.bar(range(10), variances)
    plt.xlabel('Class')
    plt.ylabel('Predictive Variance')
    plt.title('Per-Class Uncertainty')
    plt.savefig('per_class_uncertainty.png')
    plt.close()
    logger.info("Plots saved: accuracy_vs_iteration.png, sparsity_vs_accuracy.png, per_class_uncertainty.png")

# Main execution
if __name__ == "__main__":
    logger.info("Starting EM-TDAMP MNIST classification")
    train_dataset, test_dataset, x_train, y_train, x_test, y_test = load_and_preprocess_data()
    model = build_model()
    history, tdamp = em_loop(model, train_dataset, test_dataset, args.em_iters, args.sparsity)
    plot_results(history, x_test, y_test, tdamp.weight_means, model)
    
    # Save final model
    logger.info("Saving final model to mnist_tdamp_model.h5")
    model.set_weights(tdamp.weight_means)
    model.save('mnist_tdamp_model.h5')
    
    # Generate requirements.txt
    logger.info("Generating requirements.txt")
    requirements = """
tensorflow-macos==2.13.0
tensorflow-metal
tensorflow-probability==0.20.0
numpy==1.23.5
scipy==1.9.3
matplotlib==3.6.2
"""
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    logger.info("EM-TDAMP training finished")