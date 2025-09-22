import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class ModelBuilder:
    def __init__(self):
        self.framework = 'tensorflow'  # Default framework
    
    def build_model(self, config, data_info):
        """Build model based on configuration"""
        if self.framework == 'tensorflow':
            return self._build_tensorflow_model(config, data_info)
        elif self.framework == 'pytorch':
            return self._build_pytorch_model(config, data_info)
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")
    
    def _build_tensorflow_model(self, config, data_info):
        """Build TensorFlow/Keras model"""
        model_type = config.get('modelType', 'perceptron')
        
        if model_type == 'perceptron':
            return self._build_perceptron_model_tf(config, data_info)
        elif model_type == 'mlp':
            return self._build_mlp_model_tf(config, data_info)
        elif model_type == 'cnn':
            return self._build_cnn_model_tf(config, data_info)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _build_perceptron_model_tf(self, config, data_info):
        """Build single layer perceptron with TensorFlow"""
        model = keras.Sequential()
        
        input_shape = data_info['input_shape']
        if isinstance(input_shape, int):
            model.add(layers.Input(shape=(input_shape,)))
        else:
            model.add(layers.Input(shape=input_shape))
            model.add(layers.Flatten())
        
        num_classes = data_info.get('num_classes', 1)
        task_type = data_info.get('task_type', 'classification')
        
        # ✅ الحل البسيط: استخدم دائماً مخرجات متعددة للتصنيف
        if task_type == 'classification':
            if num_classes == 2:
                # ✅ للتصنيف الثنائي: مخرجان مع softmax
                model.add(layers.Dense(2, activation='softmax', name='perceptron_layer'))
            else:
                model.add(layers.Dense(num_classes, activation='softmax', name='perceptron_layer'))
        else:
            model.add(layers.Dense(1, activation='linear', name='perceptron_layer'))
        
            return model
    def _build_mlp_model_tf(self, config, data_info):
        """Build multi-layer perceptron with TensorFlow"""
        model = keras.Sequential()
        
        input_shape = data_info['input_shape']
        if isinstance(input_shape, int):
            model.add(layers.Input(shape=(input_shape,)))
        else:
            model.add(layers.Input(shape=input_shape))
            model.add(layers.Flatten())
        
        # Hidden layers
        layer_configs = config.get('layers', [])
        for i, layer_config in enumerate(layer_configs[:-1]):
            if layer_config['type'] == 'dense':
                model.add(layers.Dense(
                    units=layer_config.get('units', 64),
                    activation=layer_config.get('activation', 'relu'),
                    name=f'hidden_layer_{i+1}'
                ))
            elif layer_config['type'] == 'dropout':
                model.add(layers.Dropout(
                    rate=layer_config.get('rate', 0.5),
                    name=f'dropout_layer_{i+1}'
                ))
        
        # ✅ طبقة الإخراج المحسنة
        num_classes = data_info.get('num_classes', 1)
        task_type = data_info.get('task_type', 'classification')
        
        if task_type == 'classification':
            if num_classes == 2:
                # ✅ مخرجان حتى للتصنيف الثنائي
                model.add(layers.Dense(2, activation='softmax', name='output_layer'))
            else:
                model.add(layers.Dense(num_classes, activation='softmax', name='output_layer'))
        else:
            model.add(layers.Dense(1, activation='linear', name='output_layer'))
        
        return model
    def _build_cnn_model_tf(self, config, data_info):
        """Build CNN model with TensorFlow"""
        model = keras.Sequential()
        
        # Input layer
        input_shape = data_info['input_shape']
        model.add(layers.Input(shape=input_shape))
        
        # Build layers according to config
        layer_configs = config.get('layers', [])
        
        for i, layer_config in enumerate(layer_configs):
            layer_type = layer_config['type']
            
            if layer_type == 'conv2d':
                model.add(layers.Conv2D(
                    filters=layer_config.get('filters', 32),
                    kernel_size=layer_config.get('kernelSize', 3),
                    activation=layer_config.get('activation', 'relu'),
                    padding='same',
                    name=f'conv2d_layer_{i+1}'
                ))
            
            elif layer_type == 'maxpool2d':
                model.add(layers.MaxPooling2D(
                    pool_size=layer_config.get('poolSize', 2),
                    name=f'maxpool_layer_{i+1}'
                ))
            
            elif layer_type == 'flatten':
                model.add(layers.Flatten(name='flatten_layer'))
            
            elif layer_type == 'dense':
                model.add(layers.Dense(
                    units=layer_config.get('units', 64),
                    activation=layer_config.get('activation', 'relu'),
                    name=f'dense_layer_{i+1}'
                ))
            
            elif layer_type == 'dropout':
                model.add(layers.Dropout(
                    rate=layer_config.get('rate', 0.5),
                    name=f'dropout_layer_{i+1}'
                ))
        
        # Ensure we have output layer with correct number of classes
        output_layers = [layer for layer in layer_configs if layer['type'] == 'dense' and 
                        layer.get('activation') in ['softmax', 'sigmoid', 'linear']]
        
        if not output_layers:
            # Add output layer automatically
            num_classes = data_info.get('num_classes', 1)
            task_type = data_info.get('task_type', 'classification')
            
            if task_type == 'classification':
                if num_classes > 2:
                    model.add(layers.Dense(num_classes, activation='softmax', name='output_layer'))
                else:
                    # Binary classification: use 2 units with softmax OR 1 unit with sigmoid
                    model.add(layers.Dense(2, activation='softmax', name='output_layer'))
            else:
                model.add(layers.Dense(1, activation='linear', name='output_layer'))
        else:
            # Check if output layer has correct number of units
            output_layer = output_layers[-1]  # Get the last output layer
            num_classes = data_info.get('num_classes', 1)
            task_type = data_info.get('task_type', 'classification')
            
            if task_type == 'classification' and num_classes == 2:
                # For binary classification, ensure correct configuration
                if output_layer.get('units', 1) not in [1, 2]:
                    print(f"Warning: Binary classification detected but output layer has {output_layer.get('units')} units. Expected 1 or 2.")
        
        return model
    
    def _build_pytorch_model(self, config, data_info):
        """Build PyTorch model (for future implementation)"""
        # This would be implemented for PyTorch support
        raise NotImplementedError("PyTorch model building not implemented yet")
    
    def compile_model(self, model, config, data_info):
        """Compile the model with optimizer, loss, and metrics"""
        optimizer_name = config.get('optimizer', 'adam')
        learning_rate = config.get('learningRate', 0.001)
        task_type = data_info.get('task_type', 'classification')
        num_classes = data_info.get('num_classes', 1)
        
        # Configure optimizer
        if optimizer_name.lower() == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name.lower() == 'sgd':
            optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # ✅ الحل البسيط: استخدم دائماً sparse_categorical_crossentropy للتصنيف
        if task_type == 'classification':
            loss = 'sparse_categorical_crossentropy'  # يعمل مع أي عدد مخرجات
            metrics = ['accuracy']
            print(f"🎯 Using classification setup ({num_classes} classes)")
        else:  # regression
            loss = 'mse'
            metrics = ['mae']
            print("📈 Using regression setup")
        
        print(f"🔧 Optimizer: {optimizer_name}, Loss: {loss}, Learning Rate: {learning_rate}")
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        return model
class PerceptronLayer(layers.Layer):
    """Custom Perceptron layer implementation for educational purposes"""
    
    def __init__(self, units, activation='sigmoid', **kwargs):
        super(PerceptronLayer, self).__init__(**kwargs)
        self.units = units
        self.activation_name = activation
        self.activation_fn = keras.activations.get(activation)
    
    def build(self, input_shape):
        # Create weights and bias
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True,
            name='perceptron_weights'
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='perceptron_bias'
        )
        super(PerceptronLayer, self).build(input_shape)
    
    def call(self, inputs):
        # Linear combination: w*x + b
        linear_output = tf.matmul(inputs, self.w) + self.b
        # Apply activation function
        return self.activation_fn(linear_output)
    
    def get_config(self):
        config = super(PerceptronLayer, self).get_config()
        config.update({
            'units': self.units,
            'activation': self.activation_name
        })
        return config


class CustomLayers:
    """Custom layer implementations for advanced architectures"""
    
    @staticmethod
    def residual_block(x, filters, kernel_size=3):
        """Residual block for ResNet-like architectures"""
        shortcut = x
        
        x = layers.Conv2D(filters, kernel_size, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv2D(filters, kernel_size, padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Adjust shortcut dimensions if needed
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)
        
        return x
    
    @staticmethod
    def attention_block(x, filters):
        """Simple attention mechanism"""
        attention = layers.Conv2D(filters, 1, activation='sigmoid')(x)
        return layers.Multiply()([x, attention])


class ModelAnalyzer:
    """Analyze model architecture and provide insights"""
    
    @staticmethod
    def analyze_perceptron_suitability(data_info):
        """Analyze if data is suitable for single layer perceptron"""
        insights = {
            'suitable': True,
            'warnings': [],
            'recommendations': []
        }
        
        # Check data type
        if data_info.get('data_type') == 'images':
            insights['suitable'] = False
            insights['warnings'].append("Perceptrons are not suitable for image data")
            insights['recommendations'].append("Use CNN for image classification")
        
        # Check number of classes
        num_classes = data_info.get('num_classes', 1)
        if num_classes > 2:
            insights['warnings'].append("Perceptrons work best with binary classification")
            insights['recommendations'].append("Consider MLP for multi-class problems")
        
        # Check task type
        task_type = data_info.get('task_type', 'classification')
        if task_type == 'regression':
            insights['warnings'].append("Perceptrons are primarily designed for classification")
            insights['recommendations'].append("Consider MLP for regression tasks")
        
        # Check feature dimensionality
        input_shape = data_info.get('input_shape', 1)
        if isinstance(input_shape, (list, tuple)):
            feature_count = input_shape[0] if len(input_shape) > 0 else 1
        else:
            feature_count = input_shape
            
        if feature_count > 100:
            insights['warnings'].append("High-dimensional data may require more complex models")
            insights['recommendations'].append("Consider MLP for better feature representation")
        
        return insights
    
    @staticmethod
    def estimate_model_complexity(config, data_info):
        """Estimate model complexity and training requirements"""
        model_type = config.get('modelType', 'mlp')
        
        if model_type == 'perceptron':
            return {
                'complexity': 'Very Low',
                'parameters': 'Minimal',
                'training_time': 'Very Fast',
                'memory_usage': 'Very Low',
                'description': 'Single layer with linear decision boundary'
            }
        elif model_type == 'mlp':
            layers = config.get('layers', [])
            total_params = 0
            
            # Rough parameter estimation for dense layers
            for i, layer in enumerate(layers):
                if layer.get('type') == 'dense':
                    units = layer.get('units', 64)
                    if i == 0:
                        # First layer
                        input_size = data_info.get('input_shape', 100)
                        if isinstance(input_size, (list, tuple)):
                            input_size = input_size[0]
                        total_params += input_size * units + units
                    else:
                        # Subsequent layers
                        prev_units = layers[i-1].get('units', 64) if i > 0 else 64
                        total_params += prev_units * units + units
            
            if total_params < 10000:
                complexity = 'Low'
                training_time = 'Fast'
            elif total_params < 100000:
                complexity = 'Medium'
                training_time = 'Medium'
            else:
                complexity = 'High'
                training_time = 'Slow'
                
            return {
                'complexity': complexity,
                'parameters': f'~{total_params:,}',
                'training_time': training_time,
                'memory_usage': 'Low to Medium',
                'description': 'Multi-layer network for complex patterns'
            }
        else:  # CNN
            return {
                'complexity': 'High',
                'parameters': 'High',
                'training_time': 'Slow',
                'memory_usage': 'High',
                'description': 'Convolutional layers for spatial feature extraction'
            }