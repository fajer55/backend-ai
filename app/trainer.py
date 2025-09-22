import tensorflow as tf
from tensorflow import keras
import numpy as np
import time

class ModelTrainer:
    def __init__(self):
        self.model = None
        self.history = None
    
    def train_model(self, model, X_train, y_train, X_val, y_val, config, progress_callback=None):
        """Train the model with given data and configuration"""
        
        # Get training parameters
        epochs = config.get('epochs', 50)
        batch_size = config.get('batchSize', 32)
        validation_split = config.get('validationSplit', 0.2) if X_val is None else None
        
        # Prepare callbacks
        callbacks = []
        
        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )
        callbacks.append(early_stopping)
        
        # Model checkpoint
        checkpoint = keras.callbacks.ModelCheckpoint(
            'temp_best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
        callbacks.append(checkpoint)
        
        # Learning rate reduction
        lr_reducer = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=0
        )
        callbacks.append(lr_reducer)
        
        # Custom progress callback
        if progress_callback:
            custom_callback = CustomProgressCallback(progress_callback)
            callbacks.append(custom_callback)
        
        # Prepare validation data
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        else:
            validation_data = None
            validation_split = config.get('validationSplit', 0.2)
        
        # Train the model
        try:
            history = model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=validation_data,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=0  # Silent training, progress handled by callback
            )
            
            self.model = model
            self.history = history
            
            return history
            
        except Exception as e:
            raise Exception(f"Training failed: {str(e)}")
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model"""
        if self.model is None:
            raise Exception("No trained model available")
        
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Return structured results
        metrics = {}
        if len(results) >= 2:
            metrics['loss'] = results[0]
            metrics['accuracy'] = results[1] if len(results) > 1 else None
        
        return metrics
    
    def predict(self, X):
        """Make predictions with the trained model"""
        if self.model is None:
            raise Exception("No trained model available")
        
        predictions = self.model.predict(X, verbose=0)
        return predictions
    
    def get_model_summary(self):
        """Get model architecture summary"""
        if self.model is None:
            raise Exception("No trained model available")
        
        # Capture model summary
        summary_lines = []
        self.model.summary(print_fn=lambda x: summary_lines.append(x))
        
        return '\n'.join(summary_lines)


class CustomProgressCallback(keras.callbacks.Callback):
    """Custom callback to track training progress"""
    
    def __init__(self, progress_callback):
        super().__init__()
        self.progress_callback = progress_callback
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch"""
        if self.progress_callback and hasattr(self.progress_callback, 'on_epoch_end'):
            self.progress_callback.on_epoch_end(epoch, logs)


class TrainingMonitor:
    """Monitor training metrics and provide insights"""
    
    def __init__(self):
        self.training_history = []
    
    def analyze_training(self, history):
        """Analyze training history and provide insights"""
        insights = {
            'converged': self._check_convergence(history),
            'overfitting': self._check_overfitting(history),
            'best_epoch': self._find_best_epoch(history),
            'training_stability': self._assess_stability(history),
            'recommendations': []
        }
        
        # Generate recommendations based on analysis
        insights['recommendations'] = self._generate_recommendations(insights, history)
        
        return insights
    
    def _check_convergence(self, history):
        """Check if training has converged"""
        if 'loss' not in history.history:
            return False
        
        losses = history.history['loss']
        if len(losses) < 5:
            return False
        
        # Check if loss has stabilized in the last 5 epochs
        recent_losses = losses[-5:]
        loss_std = np.std(recent_losses)
        loss_mean = np.mean(recent_losses)
        
        # Convergence if standard deviation is less than 1% of mean
        return loss_std < 0.01 * loss_mean
    
    def _check_overfitting(self, history):
        """Check for signs of overfitting"""
        if 'loss' not in history.history or 'val_loss' not in history.history:
            return False
        
        train_losses = history.history['loss']
        val_losses = history.history['val_loss']
        
        if len(train_losses) < 5:
            return False
        
        # Check if validation loss is consistently higher and increasing
        # while training loss is decreasing
        recent_train = np.mean(train_losses[-5:])
        recent_val = np.mean(val_losses[-5:])
        
        train_trend = np.polyfit(range(5), train_losses[-5:], 1)[0]
        val_trend = np.polyfit(range(5), val_losses[-5:], 1)[0]
        
        # Overfitting if val_loss > train_loss and val_loss trending up
        return recent_val > recent_train * 1.1 and val_trend > 0
    
    def _find_best_epoch(self, history):
        """Find the epoch with best validation performance"""
        if 'val_loss' not in history.history:
            return len(history.history['loss']) - 1
        
        val_losses = history.history['val_loss']
        return np.argmin(val_losses)
    
    def _assess_stability(self, history):
        """Assess training stability"""
        if 'loss' not in history.history:
            return 'unknown'
        
        losses = history.history['loss']
        if len(losses) < 10:
            return 'insufficient_data'
        
        # Calculate coefficient of variation for recent epochs
        recent_losses = losses[-10:]
        cv = np.std(recent_losses) / np.mean(recent_losses)
        
        if cv < 0.05:
            return 'very_stable'
        elif cv < 0.1:
            return 'stable'
        elif cv < 0.2:
            return 'moderately_stable'
        else:
            return 'unstable'
    
    def _generate_recommendations(self, insights, history):
        """Generate training recommendations"""
        recommendations = []
        
        if insights['overfitting']:
            recommendations.append({
                'type': 'overfitting',
                'message': 'Model is overfitting. Consider adding regularization, dropout, or reducing model complexity.',
                'actions': ['Add dropout layers', 'Reduce learning rate', 'Add early stopping', 'Get more training data']
            })
        
        if not insights['converged']:
            recommendations.append({
                'type': 'convergence',
                'message': 'Model has not converged. Consider training for more epochs or adjusting learning rate.',
                'actions': ['Increase epochs', 'Adjust learning rate', 'Check data preprocessing']
            })
        
        if insights['training_stability'] == 'unstable':
            recommendations.append({
                'type': 'stability',
                'message': 'Training is unstable. Consider reducing learning rate or batch size.',
                'actions': ['Reduce learning rate', 'Use learning rate scheduler', 'Increase batch size']
            })
        
        # Performance-based recommendations
        if 'val_accuracy' in history.history:
            final_acc = history.history['val_accuracy'][-1]
            if final_acc < 0.7:
                recommendations.append({
                    'type': 'performance',
                    'message': 'Model accuracy is below 70%. Consider improving the model architecture or data quality.',
                    'actions': ['Try different architecture', 'Increase model capacity', 'Improve data preprocessing', 'Feature engineering']
                })
        
        return recommendations


class DataAugmentation:
    """Data augmentation utilities for image data"""
    
    @staticmethod
    def get_image_augmentation(config):
        """Get image data augmentation based on config"""
        aug_config = config.get('augmentation', {})
        
        if not aug_config.get('enabled', False):
            return None
        
        augmentation = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=aug_config.get('rotation', 20),
            width_shift_range=aug_config.get('width_shift', 0.2),
            height_shift_range=aug_config.get('height_shift', 0.2),
            horizontal_flip=aug_config.get('horizontal_flip', True),
            zoom_range=aug_config.get('zoom', 0.2),
            fill_mode='nearest'
        )
        
        return augmentation
    
    @staticmethod
    def apply_augmentation(X_train, y_train, augmentation, batch_size=32):
        """Apply data augmentation to training data"""
        if augmentation is None:
            return X_train, y_train
        
        # Create augmented data generator
        train_generator = augmentation.flow(
            X_train, y_train,
            batch_size=batch_size
        )
        
        return train_generator


class ModelOptimizer:
    """Utilities for model optimization and hyperparameter tuning"""
    
    def __init__(self):
        self.best_params = None
        self.best_score = None
    
    def optimize_hyperparameters(self, model_builder, X_train, y_train, X_val, y_val, param_space):
        """Simple grid search for hyperparameter optimization"""
        # This is a simplified version - in production, use more sophisticated methods
        best_score = -np.inf
        best_params = None
        
        for params in self._generate_param_combinations(param_space):
            try:
                # Build model with current parameters
                model = model_builder.build_model(params, {'input_shape': X_train.shape[1:]})
                model = model_builder.compile_model(model, params, {})
                
                # Train with early stopping
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=20,  # Limited epochs for optimization
                    verbose=0,
                    callbacks=[keras.callbacks.EarlyStopping(patience=5)]
                )
                
                # Evaluate
                score = max(history.history.get('val_accuracy', [0]))
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
            except Exception as e:
                print(f"Failed to evaluate params {params}: {e}")
                continue
        
        self.best_params = best_params
        self.best_score = best_score
        
        return best_params, best_score
    
    def _generate_param_combinations(self, param_space):
        """Generate parameter combinations for grid search"""
        # Simplified implementation
        combinations = []
        
        # Example: generate a few combinations
        learning_rates = param_space.get('learning_rate', [0.001])
        batch_sizes = param_space.get('batch_size', [32])
        
        for lr in learning_rates:
            for bs in batch_sizes:
                combinations.append({
                    'learningRate': lr,
                    'batchSize': bs
                })
        
        return combinations[:10]  # Limit to 10 combinations for demo
