import tensorflow       as tf
from   tensorflow.keras import backend as K

def coral_loss(num_classes):
    """
    Custom CORAL loss function for ordinal regression.

    Args:
        num_classes: The total number of ordered classes (bins).

    Returns:
        A Keras loss function.
    """
    def loss(y_true_levels, y_pred_logits):
        """Calculates the CORAL loss."""
        # Ensure y_true_levels is integer
        y_true_levels            = tf.cast(y_true_levels, tf.int32)
        # Remove singleton dimension if present
        y_true_levels            = tf.squeeze(y_true_levels) 

        # Create true cumulative labels (batch_size, num_classes - 1)
        val                      = tf.cast(tf.range(0, num_classes - 1), tf.int32)
        val                      = tf.expand_dims(val, axis=0) 

        y_true_levels_expanded   = tf.expand_dims(y_true_levels, axis=1) 
        # Create boolean mask: True where val < y_true_levels
        y_true_cum               = tf.cast(val < y_true_levels_expanded, tf.float32)
        # Apply sigmoid to logits to get cumulative probabilities
        y_pred_cumprobs          = tf.sigmoid(y_pred_logits)

        # Calculate CORAL loss (log loss on cumulative probabilities)
        loss                     = - (y_true_cum 
                                      * tf.math.log(y_pred_cumprobs + K.epsilon()) 
                                      + (1.0 - y_true_cum) 
                                      * tf.math.log(1.0 - y_pred_cumprobs + K.epsilon()))
        
        # Sum the loss across thresholds for each sample.
        loss                     = tf.reduce_sum(loss, axis=1)
        # Average the loss over all samples in the batch.
        loss                     = tf.reduce_mean(loss)
        return K.mean(loss)

    return loss

def levels_from_logits(logits):
    """Converts model logits (cumulative probabilities) to predicted levels (bins)."""
    cumprobs       = tf.sigmoid(logits)
    # Check where cumprobs > 0.5. Summing these gives the predicted level.
    predict_levels = tf.reduce_sum(tf.cast(cumprobs > 0.5, tf.int32), axis=1)
    return predict_levels

def levels_to_probs(levels, num_classes):
    """
    Converts predicted levels back to probability distribution over classes
    (for consistency if needed, although not directly used by CORAL prediction).
    """
    levels = tf.cast(levels, tf.int32)
    # Create one-hot encoding based on the predicted level
    probs = tf.one_hot(levels, depth=num_classes)
    return probs