import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import argparse
import os
import logging
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns

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
parser.add_argument('--em_iters', type=int, default=50, help='Number of EM iterations')
parser.add_argument('--sparsity', type=float, default=0.5, help='Target sparsity level')
args = parser.parse_args()

# 1. Project Setup & Data
def load_and_preprocess_data():
    """Load and preprocess the full MNIST dataset."""
    logger.info("Loading and preprocessing MNIST dataset")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Use full training data
    x_train = x_train[:10000] / 255.0  # Take first 10,000 samples
    y_train = y_train[:10000]
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784) / 255.0
    
    # One-hot encode labels
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(60000).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
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
        # Initialize legacy Adam optimizer for M1/M2 compatibility
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)
    
    def module_a_damp(self, x, y, priors):
        """Module A: Deep Approximate Message Passing (simplified)."""
        # Unpack priors (means and variances)
        prior_means, prior_variances = priors
        
        # Sample weights from Gaussian priors
        sampled_weights = []
        for mean, var in zip(prior_means, prior_variances):
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
        
        # Compute gradients w.r.t. trainable variables
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Log gradient norms for debugging
        grad_norms = [tf.norm(g).numpy() if g is not None else 0.0 for g in gradients]
        logger.debug("Gradient norms: %s", grad_norms)
        
        # Update posterior means using Adam optimizer
        posterior_means = []
        posterior_variances = []
        for i, (mean, var, grad) in enumerate(zip(prior_means, prior_variances, gradients)):
            if grad is None:
                # Handle case where gradient is None (e.g., for biases if not used)
                posterior_means.append(mean)
                posterior_variances.append(var)
                continue
            # Create a variable for optimization
            temp_mean = tf.Variable(mean, trainable=True)
            self.optimizer.apply_gradients([(grad, temp_mean)])
            # Update variance: Reduce variance to simulate posterior tightening
            new_var = var * 0.95
            posterior_means.append(tf.Variable(temp_mean, trainable=False))
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
        priors = (self.weight_means, self.weight_variances)
        posterior_means, posterior_variances, loss = self.module_a_damp(x, y, priors)
        posteriors = (posterior_means, posterior_variances)
        extrinsics, new_rho = self.module_b_spmp(posteriors)[:2]
        self.weight_means = [tf.Variable(m, trainable=False) for m in posteriors[0]]
        self.weight_variances = posteriors[1]
        
        if batch_idx % 100 == 0:
            avg_rho = tf.reduce_mean(self.rho).numpy()
            logger.info("Batch %d: Loss = %.4f, Average rho = %.4f", batch_idx, loss, avg_rho)
        
        return posteriors

# 5. EM Loop
def em_loop(model, train_dataset, test_dataset, em_iters, sparsity_target):
    """Run EM-TDAMP training loop with early stopping."""
    logger.info("Starting EM-TDAMP training for %d iterations with target sparsity %.2f", em_iters, sparsity_target)
    tdamp = TDAMP(model, num_groups=128, v_init=1.0)
    history = {
        'val_accuracy': [], 'val_precision': [], 'val_recall': [], 'val_f1': [],
        'sparsity': [], 'active_neurons': [], 'avg_entropy': []
    }
    patience = 3
    min_delta = 0.001
    best_accuracy = -float('inf')
    patience_counter = 0

    for em_iter in range(em_iters):
        logger.info("EM Iteration %d/%d", em_iter + 1, em_iters)
        batch_idx = 0
        total_loss = 0.0
        num_batches = 0
        
        for x_batch, y_batch in train_dataset:
            posteriors = tdamp.step(x_batch, y_batch, batch_idx)
            batch_idx += 1
            num_batches += 1
            total_loss += -tf.reduce_mean(probit_product_likelihood(model(x_batch, training=False), y_batch, tdamp.v)).numpy()
        
        tdamp.v.assign(tf.maximum(tdamp.v * 0.99, 1e-6))
        threshold = np.percentile(tdamp.rho.numpy(), 100 * (1 - sparsity_target))
        threshold = tf.cast(threshold, tf.float32)
        tdamp.rho.assign(tf.where(tdamp.rho > threshold, tdamp.rho, 0.0))
        
        accuracy, precision, recall, f1, cm = evaluate_model(model, test_dataset, tdamp.weight_means)
        sparsity = compute_sparsity(tdamp.rho)
        avg_entropy = compute_average_entropy(model, test_dataset, tdamp.weight_means)
        avg_loss = total_loss / num_batches
        active_neurons = tf.reduce_sum(tf.cast(tdamp.rho > 0, tf.float32)).numpy()
        
        history['val_accuracy'].append(float(accuracy))
        history['val_precision'].append(float(precision))
        history['val_recall'].append(float(recall))
        history['val_f1'].append(float(f1))
        history['sparsity'].append(float(sparsity))
        history['active_neurons'].append(int(active_neurons))
        history['avg_entropy'].append(float(avg_entropy))
        
        if accuracy > best_accuracy + min_delta:
            best_accuracy = accuracy
            patience_counter = 0
            final_cm = cm  # Save the confusion matrix of the best model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered at iteration %d", em_iter + 1)
                break
        
        logger.info(
            "Val Accuracy = %.4f, Precision = %.4f, Recall = %.4f, F1 = %.4f, Sparsity = %.4f, Avg Entropy = %.4f, Avg Loss = %.4f, Active Neurons = %d/%d",
            accuracy, precision, recall, f1, sparsity, avg_entropy, avg_loss, int(active_neurons), tdamp.num_groups
        )
    
    logger.info("Training completed. Final Val Accuracy = %.4f, Final Sparsity = %.4f", 
                history['val_accuracy'][-1], history['sparsity'][-1])
    return history, tdamp, final_cm

# 6. Evaluation and Visualization
def evaluate_model(model, dataset, weight_means):
    """Evaluate model with multiple metrics using posterior means."""
    original_weights = model.get_weights()
    model.set_weights(weight_means)
    all_predictions = []
    all_labels = []
    for x_batch, y_batch in dataset:
        logits = model(x_batch, training=False)
        predictions = tf.argmax(logits, axis=1)
        labels = tf.argmax(y_batch, axis=1)
        all_predictions.extend(predictions.numpy())
        all_labels.extend(labels.numpy())
    model.set_weights(original_weights)
    
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro')
    cm = confusion_matrix(all_labels, all_predictions)
    
    return accuracy, precision, recall, f1, cm

def compute_average_entropy(model, dataset, weight_means):
    """Compute average predictive entropy."""
    model.set_weights(weight_means)
    entropies = []
    for x_batch, _ in dataset:
        logits = model(x_batch, training=False)
        probs = tf.nn.softmax(logits)
        entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-10), axis=1)
        entropies.extend(entropy.numpy())
    return np.mean(entropies)

def compute_sparsity(rho):
    """Compute sparsity as fraction of active neurons."""
    return 1.0 - tf.reduce_mean(tf.cast(rho > 0, tf.float32)).numpy()

def plot_results(history, x_test, y_test, weight_means, model, tdamp, final_cm):
    """Generate enhanced visualization plots."""
    logger.info("Generating enhanced visualization plots")
    
    # Accuracy vs. EM Iteration
    plt.figure(figsize=(8, 6))
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('EM Iteration')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. EM Iteration')
    plt.legend()
    plt.savefig('accuracy_vs_iteration.png')
    plt.close()
    
    # Precision, Recall, F1-Score vs. EM Iteration
    plt.figure(figsize=(8, 6))
    plt.plot(history['val_precision'], label='Precision')
    plt.plot(history['val_recall'], label='Recall')
    plt.plot(history['val_f1'], label='F1-Score')
    plt.xlabel('EM Iteration')
    plt.ylabel('Metric Value')
    plt.title('Precision, Recall, and F1-Score vs. EM Iteration')
    plt.legend()
    plt.savefig('precision_recall_f1_vs_iteration.png')
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
    
    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(final_cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Active Neurons vs. EM Iteration
    plt.figure(figsize=(8, 6))
    plt.plot(history['active_neurons'], label='Active Neurons')
    plt.xlabel('EM Iteration')
    plt.ylabel('Number of Active Neurons')
    plt.title('Active Neurons vs. EM Iteration')
    plt.legend()
    plt.savefig('active_neurons_vs_iteration.png')
    plt.close()
    
    # Rho Distribution
    plt.figure(figsize=(8, 6))
    plt.hist(tdamp.rho.numpy(), bins=20)
    plt.xlabel('Rho Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Rho Values')
    plt.savefig('rho_distribution.png')
    plt.close()
    
    logger.info("Plots saved: accuracy_vs_iteration.png, precision_recall_f1_vs_iteration.png, sparsity_vs_accuracy.png, per_class_uncertainty.png, confusion_matrix.png, active_neurons_vs_iteration.png, rho_distribution.png")

# Main execution
if __name__ == "__main__":
    logger.info("Starting EM-TDAMP MNIST classification")
    train_dataset, test_dataset, x_train, y_train, x_test, y_test = load_and_preprocess_data()
    model = build_model()
    history, tdamp, final_cm = em_loop(model, train_dataset, test_dataset, args.em_iters, args.sparsity)
    plot_results(history, x_test, y_test, tdamp.weight_means, model, tdamp, final_cm)
    
    # Save metrics
    df = pd.DataFrame(history)
    df.to_csv('training_history.csv', index=False)
    np.savetxt('final_confusion_matrix.csv', final_cm, delimiter=',', fmt='%d')
    logger.info("Metrics saved: training_history.csv, final_confusion_matrix.csv")
    
    # Save final model
    logger.info("Saving final model to mnist_tdamp_model.keras")
    model.set_weights(tdamp.weight_means)
    model.save('mnist_tdamp_model.keras')
    
    # Generate requirements.txt
    logger.info("Generating requirements.txt")
    requirements = """
tensorflow==2.13.0
tensorflow-probability==0.20.0
numpy==1.23.5
scipy==1.9.3
matplotlib==3.6.2
pandas==1.5.2
scikit-learn==1.2.0
seaborn==0.12.2
"""
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    logger.info("EM-TDAMP training finished")