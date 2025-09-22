import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import logging
from datetime import datetime

try:
    import cv2
except ImportError:
    cv2 = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTester:
    def __init__(self, models_folder='trained_models', uploads_folder='uploads'):
        self.models_folder = models_folder
        self.uploads_folder = uploads_folder
        self.loaded_model = None
        self.model_metadata = None
        self.scaler = None
        self.label_encoder = None
        self.session_id = None
        self.input_shape = None
        
    def get_available_models(self):
        """Get list of available trained models with improved error handling"""
        if not os.path.exists(self.models_folder):
            logger.warning(f"Models folder not found: {self.models_folder}")
            return []
        
        models = []
        try:
            for item in os.listdir(self.models_folder):
                model_path = os.path.join(self.models_folder, item)
                if os.path.isdir(model_path):
                    try:
                        if self._is_valid_model_directory(model_path):
                            model_info = self._get_model_info(model_path)
                            if model_info:
                                models.append(model_info)
                    except Exception as e:
                        logger.error(f"Error processing model {item}: {str(e)}")
                        continue
        except Exception as e:
            logger.error(f"Error reading models folder: {str(e)}")
            return []
        
        # Sort by creation time (newest first)
        models.sort(key=lambda x: x.get('created_at', 0), reverse=True)
        return models
    
    def _is_valid_model_directory(self, model_path):
        """Check if directory contains a valid model with multiple format support"""
        try:
            # Check for TensorFlow SavedModel format
            if os.path.exists(os.path.join(model_path, 'saved_model.pb')):
                return True
            
            # Check for Keras format
            if os.path.exists(os.path.join(model_path, 'keras_metadata.pb')):
                return True
                
            # Check for .h5 files
            for file in os.listdir(model_path):
                if file.endswith(('.h5', '.hdf5')):
                    return True
                    
            # Check for variables folder (SavedModel format)
            if os.path.exists(os.path.join(model_path, 'variables')):
                return True
                
            return False
        except Exception as e:
            logger.error(f"Error validating model directory {model_path}: {str(e)}")
            return False
    
    def _get_model_info(self, model_path):
        """Extract model information with enhanced error handling"""
        try:
            model_name = os.path.basename(model_path)
            created_at = os.path.getctime(model_path)
            
            # Try to load training metadata
            metadata = {}
            metadata_file = os.path.join(model_path, 'training_metadata.json')
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load metadata for {model_name}: {str(e)}")
            
            # Extract session ID
            session_id = None
            if model_name.startswith('model_'):
                session_id = model_name.replace('model_', '')
            
            # Create display name
            display_name = self._create_display_name(model_name, metadata, created_at)
            
            # Get class information
            class_info = self._get_class_info(model_path, metadata)
            
            # Validate model can be loaded
            model_status = self._check_model_status(model_path)
            
            return {
                'id': model_name,
                'name': model_name,
                'display_name': display_name,
                'path': model_path,
                'session_id': session_id,
                'created_at': created_at,
                'data_type': metadata.get('data_type', 'unknown'),
                'task_type': metadata.get('task_type', 'unknown'),
                'model_type': metadata.get('model_type', 'unknown'),
                'model_architecture': metadata.get('model_architecture', 'Unknown'),
                'input_shape': metadata.get('input_shape'),
                'num_classes': metadata.get('num_classes'),
                'class_names': class_info.get('class_names', []),
                'feature_names': class_info.get('feature_names', []),
                'final_accuracy': metadata.get('final_accuracy'),
                'metadata': metadata,
                'status': model_status
            }
        except Exception as e:
            logger.error(f"Error getting model info for {model_path}: {str(e)}")
            return None
    
    def _check_model_status(self, model_path):
        """Check if model can be loaded successfully"""
        try:
            test_model = tf.keras.models.load_model(model_path)
            del test_model  # Free memory
            return "healthy"
        except Exception as e:
            logger.error(f"Model health check failed for {model_path}: {str(e)}")
            return "corrupted"
    
    def _create_display_name(self, model_name, metadata, created_at):
        """Create user-friendly display name"""
        try:
            from datetime import datetime
            
            data_type = metadata.get('data_type', 'unknown')
            task_type = metadata.get('task_type', 'unknown')
            model_type = metadata.get('model_type', 'neural_network')
            
            # Format creation time
            try:
                dt = datetime.fromtimestamp(created_at)
                time_str = dt.strftime('%m/%d %H:%M')
            except:
                time_str = 'unknown'
            
            # Create descriptive name based on model type
            if model_type == 'perceptron':
                return f"Perceptron Model - {time_str}"
            elif model_type == 'mlp':
                if data_type == 'tabular':
                    if task_type == 'classification':
                        num_classes = metadata.get('num_classes', '?')
                        return f"MLP Classifier ({num_classes} classes) - {time_str}"
                    else:
                        return f"MLP Regressor - {time_str}"
                else:
                    return f"Multi-Layer Perceptron - {time_str}"
            elif model_type == 'cnn':
                if task_type == 'classification':
                    num_classes = metadata.get('num_classes', '?')
                    return f"CNN Image Classifier ({num_classes} classes) - {time_str}"
                else:
                    return f"CNN Model - {time_str}"
            else:
                return f"Neural Network - {time_str}"
        except Exception as e:
            logger.error(f"Error creating display name: {str(e)}")
            return f"Model - {model_name}"
    
    def _get_class_info(self, model_path, metadata):
        """Get class names and feature information with error handling"""
        info = {'class_names': [], 'feature_names': []}
        
        try:
            # Get from metadata first
            if 'class_names' in metadata:
                info['class_names'] = metadata['class_names']
            
            # Try to get from session data
            session_id = metadata.get('session_id')
            if session_id:
                session_file = os.path.join(self.uploads_folder, f"{session_id}_session.json")
                if os.path.exists(session_file):
                    try:
                        with open(session_file, 'r', encoding='utf-8') as f:
                            session_data = json.load(f)
                        
                        preview_data = session_data.get('preview', {})
                        
                        if preview_data.get('type') == 'images':
                            class_names = preview_data.get('class_names', [])
                            if class_names:
                                info['class_names'] = class_names
                        elif preview_data.get('type') == 'csv':
                            column_names = preview_data.get('column_names', [])
                            if column_names:
                                info['feature_names'] = column_names[:-1]
                                if not info['class_names']:
                                    info['target_name'] = column_names[-1]
                    
                    except Exception as e:
                        logger.warning(f"Could not load session data: {str(e)}")
        except Exception as e:
            logger.error(f"Error getting class info: {str(e)}")
        
        return info
    
    def load_model(self, model_id):
        """Load a specific model with comprehensive error handling"""
        try:
            model_path = os.path.join(self.models_folder, model_id)
            if not os.path.exists(model_path):
                return {'success': False, 'error': f'Model directory not found: {model_id}'}
            
            # Clear previous model
            self._cleanup()
            
            # Load the TensorFlow model
            try:
                self.loaded_model = tf.keras.models.load_model(model_path)
                logger.info(f"Successfully loaded model: {model_id}")
            except Exception as e:
                error_msg = f"Failed to load TensorFlow model: {str(e)}"
                logger.error(error_msg)
                return {'success': False, 'error': error_msg}
            
            # Load model metadata
            metadata_file = os.path.join(model_path, 'training_metadata.json')
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        self.model_metadata = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load metadata: {str(e)}")
                    self.model_metadata = {}
            else:
                self.model_metadata = {}
            
            # Store session info
            self.session_id = model_id
            
            # Get input shape from model
            if self.loaded_model:
                try:
                    self.input_shape = self.loaded_model.input_shape
                except:
                    self.input_shape = None
            
            # Load preprocessing objects
            self._load_preprocessing_objects(model_path)
            
            # Get enhanced model info
            model_info = self._get_model_info(model_path)
            
            return {
                'success': True,
                'model_info': model_info,
                'input_shape': self.input_shape,
                'preprocessing_available': {
                    'scaler': self.scaler is not None,
                    'label_encoder': self.label_encoder is not None
                }
            }
            
        except Exception as e:
            error_msg = f"Unexpected error loading model {model_id}: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def _load_preprocessing_objects(self, model_path):
        """Load preprocessing objects with error handling"""
        # Load scaler
        scaler_path = os.path.join(model_path, 'scaler.pkl')
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("Scaler loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load scaler: {str(e)}")
                self.scaler = None
        
        # Load label encoder
        encoder_path = os.path.join(model_path, 'label_encoder.pkl')
        if os.path.exists(encoder_path):
            try:
                with open(encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                logger.info("Label encoder loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load label encoder: {str(e)}")
                self.label_encoder = None
    
    def test_csv_input(self, csv_data):
        """Test model with CSV input - improved validation and error handling"""
        if not self.loaded_model:
            return {'success': False, 'error': 'No model loaded. Please load a model first.'}
        
        try:
            # Validate and parse CSV data
            parsed_data = self._parse_csv_input(csv_data)
            if not parsed_data['success']:
                return parsed_data
            
            input_data = parsed_data['data']
            
            # Validate input shape
            expected_features = self._get_expected_features()
            if expected_features and len(input_data[0]) != expected_features:
                return {
                    'success': False, 
                    'error': f'Expected {expected_features} features, got {len(input_data[0])}. Please check the input format.'
                }
            
            # Apply preprocessing
            if self.scaler:
                try:
                    input_data = self.scaler.transform(input_data)
                    logger.info("Applied scaler preprocessing")
                except Exception as e:
                    logger.warning(f"Failed to apply scaler: {str(e)}")
            
            # Make prediction
            try:
                prediction = self.loaded_model.predict(input_data, verbose=0)
                logger.info(f"Prediction completed, shape: {prediction.shape}")
            except Exception as e:
                return {'success': False, 'error': f'Prediction failed: {str(e)}'}
            
            # Process results
            task_type = self.model_metadata.get('task_type', 'classification')
            model_type = self.model_metadata.get('model_type', 'unknown')
            
            if task_type == 'classification':
                return self._process_classification_result(prediction, model_type)
            else:
                return self._process_regression_result(prediction, model_type)
                
        except Exception as e:
            error_msg = f"CSV testing failed: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def _parse_csv_input(self, csv_data):
        """Parse and validate CSV input data"""
        try:
            if not csv_data or not str(csv_data).strip():
                return {'success': False, 'error': 'Empty input data provided'}
            
            # Convert to string and clean
            csv_str = str(csv_data).strip()
            
            # Split by commas and clean values
            raw_values = [x.strip() for x in csv_str.split(',')]
            
            # Remove empty values
            raw_values = [x for x in raw_values if x]
            
            if not raw_values:
                return {'success': False, 'error': 'No valid values found in input'}
            
            # Convert to numeric values
            values = []
            for i, val in enumerate(raw_values):
                try:
                    # Try to convert to float
                    num_val = float(val)
                    values.append(num_val)
                except ValueError:
                    return {
                        'success': False, 
                        'error': f'Invalid numeric value at position {i+1}: "{val}". All values must be numeric.'
                    }
            
            # Convert to numpy array with proper shape
            input_data = np.array(values).reshape(1, -1)
            
            return {'success': True, 'data': input_data}
            
        except Exception as e:
            return {'success': False, 'error': f'Failed to parse CSV data: {str(e)}'}
    
    def _get_expected_features(self):
        """Get expected number of features from model metadata"""
        try:
            if self.input_shape:
                if len(self.input_shape) == 2:  # (batch, features)
                    return self.input_shape[1]
                elif len(self.input_shape) == 1:  # (features,)
                    return self.input_shape[0]
            
            # Try to get from metadata
            input_shape = self.model_metadata.get('input_shape')
            if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 1:
                if isinstance(input_shape[0], int):
                    return input_shape[0]
            elif isinstance(input_shape, int):
                return input_shape
            
            return None
        except Exception as e:
            logger.warning(f"Could not determine expected features: {str(e)}")
            return None
    
    def test_image_input(self, image_path):
        """Test model with image input - improved preprocessing"""
        if not self.loaded_model:
            return {'success': False, 'error': 'No model loaded. Please load a model first.'}
        
        try:
            # Validate image file
            if not os.path.exists(image_path):
                return {'success': False, 'error': 'Image file not found'}
            
            # Get expected input shape
            expected_shape = self._get_expected_image_shape()
            
            # Load and preprocess image
            processed_image = self._preprocess_image(image_path, expected_shape)
            if not processed_image['success']:
                return processed_image
            
            image_data = processed_image['data']
            
            # Make prediction
            try:
                prediction = self.loaded_model.predict(image_data, verbose=0)
                logger.info(f"Image prediction completed, shape: {prediction.shape}")
            except Exception as e:
                return {'success': False, 'error': f'Prediction failed: {str(e)}'}
            
            # Process results (images are typically classification)
            model_type = self.model_metadata.get('model_type', 'cnn')
            return self._process_classification_result(prediction, model_type)
            
        except Exception as e:
            error_msg = f"Image testing failed: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def _get_expected_image_shape(self):
        """Get expected image input shape"""
        try:
            if self.input_shape and len(self.input_shape) >= 3:
                # Extract height, width from input shape (batch, height, width, channels)
                return (self.input_shape[1], self.input_shape[2])
            
            # Default to common size
            return (224, 224)
        except Exception as e:
            logger.warning(f"Could not determine expected image shape: {str(e)}")
            return (224, 224)
    
    def _preprocess_image(self, image_path, target_size):
        """Preprocess image with comprehensive error handling"""
        try:
            # Try OpenCV first
            if cv2 is not None:
                try:
                    image = cv2.imread(image_path)
                    if image is None:
                        raise ValueError("OpenCV could not read the image")
                    
                    # Convert BGR to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, target_size)
                    image = image.astype(np.float32) / 255.0
                    
                except Exception as e:
                    logger.warning(f"OpenCV failed: {str(e)}, trying PIL")
                    image = None
            else:
                image = None
            
            # Fallback to PIL if OpenCV failed or not available
            if image is None:
                try:
                    with Image.open(image_path) as pil_image:
                        # Convert to RGB if needed
                        if pil_image.mode != 'RGB':
                            pil_image = pil_image.convert('RGB')
                        
                        # Resize
                        pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
                        
                        # Convert to numpy array
                        image = np.array(pil_image, dtype=np.float32) / 255.0
                        
                except Exception as e:
                    return {'success': False, 'error': f'Failed to process image with PIL: {str(e)}'}
            
            # Ensure proper shape
            if len(image.shape) == 2:  # Grayscale
                image = np.stack([image] * 3, axis=-1)
            elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
                image = image[:, :, :3]  # Remove alpha channel
            
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
            
            return {'success': True, 'data': image}
            
        except Exception as e:
            return {'success': False, 'error': f'Image preprocessing failed: {str(e)}'}
    
    def _process_classification_result(self, prediction, model_type='unknown'):
        """Process classification results with improved binary classification handling"""
        try:
            probabilities = prediction[0]
            num_classes = self.model_metadata.get('num_classes', len(probabilities))
            
            # Handle different output formats
            if len(probabilities) == 1:
                # Single output (sigmoid for binary classification)
                confidence = float(probabilities[0])
                
                # For sigmoid output, confidence is probability of positive class
                if confidence > 0.5:
                    predicted_class_idx = 1
                    final_confidence = confidence
                else:
                    predicted_class_idx = 0
                    final_confidence = 1 - confidence
                
                # Create probabilities for both classes
                all_probs = [1 - confidence, confidence]
                
            else:
                # Multi-class or binary with softmax (2 outputs)
                predicted_class_idx = np.argmax(probabilities)
                final_confidence = float(probabilities[predicted_class_idx])
                all_probs = probabilities.tolist()
            
            # Get class names
            class_names = self.model_metadata.get('class_names', [])
            if class_names and len(class_names) > predicted_class_idx:
                predicted_class = class_names[predicted_class_idx]
            else:
                if len(all_probs) == 2 and not class_names:
                    predicted_class = "Positive" if predicted_class_idx == 1 else "Negative"
                else:
                    predicted_class = f"Class {predicted_class_idx}"
            
            # Prepare all class probabilities
            all_probabilities = []
            for i, prob in enumerate(all_probs):
                if class_names and len(class_names) > i:
                    class_name = class_names[i]
                elif len(all_probs) == 2:
                    class_name = "Positive" if i == 1 else "Negative"
                else:
                    class_name = f"Class {i}"
                    
                all_probabilities.append({
                    'class': class_name,
                    'probability': float(prob),
                    'percentage': float(prob * 100)
                })
            
            # Sort by probability
            all_probabilities.sort(key=lambda x: x['probability'], reverse=True)
            
            # Model insights
            model_insights = self._get_model_insights(model_type, 'classification')
            
            return {
                'success': True,
                'task_type': 'classification',
                'model_type': model_type,
                'predicted_class': predicted_class,
                'confidence': final_confidence,
                'confidence_percentage': final_confidence * 100,
                'all_probabilities': all_probabilities,
                'model_insights': model_insights,
                'raw_prediction': probabilities.tolist()
            }
            
        except Exception as e:
            error_msg = f'Error processing classification result: {str(e)}'
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def _process_regression_result(self, prediction, model_type='unknown'):
        """Process regression results"""
        try:
            predicted_value = float(prediction[0][0])
            
            # Model insights
            model_insights = self._get_model_insights(model_type, 'regression')
            
            return {
                'success': True,
                'task_type': 'regression',
                'model_type': model_type,
                'predicted_value': predicted_value,
                'raw_prediction': prediction.tolist(),
                'model_insights': model_insights
            }
            
        except Exception as e:
            error_msg = f'Error processing regression result: {str(e)}'
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def _get_model_insights(self, model_type, task_type):
        """Get model-specific insights"""
        insights = {
            'perceptron': {
                'classification': "Perceptron prediction using linear decision boundary. Best for linearly separable data.",
                'regression': "Perceptron regression (note: primarily designed for classification)."
            },
            'mlp': {
                'classification': "MLP prediction using non-linear decision boundaries with multiple hidden layers.",
                'regression': "MLP regression using non-linear function approximation."
            },
            'cnn': {
                'classification': "CNN prediction using convolutional feature extraction for spatial data.",
                'regression': "CNN regression using spatial feature extraction."
            }
        }
        
        return insights.get(model_type, {}).get(task_type, f"{model_type} prediction completed.")
    
    def get_training_data_info(self, model_id):
        """Get training data information with enhanced error handling"""
        try:
            model_path = os.path.join(self.models_folder, model_id)
            metadata_file = os.path.join(model_path, 'training_metadata.json')
            
            if not os.path.exists(metadata_file):
                return {'success': False, 'error': 'No training metadata available for this model'}
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            session_id = metadata.get('session_id')
            if not session_id:
                return {'success': False, 'error': 'No session information available'}
            
            session_file = os.path.join(self.uploads_folder, f"{session_id}_session.json")
            if not os.path.exists(session_file):
                return {'success': False, 'error': 'Training session data not found'}
            
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            preview_data = session_data.get('preview', {})
            
            if preview_data.get('type') != 'csv':
                return {'success': False, 'error': 'Model was not trained on CSV data'}
            
            # Get sample data for guidance
            sample_data = preview_data.get('sample', [])
            column_names = preview_data.get('column_names', [])
            
            # Create example inputs (without target column)
            examples = []
            if sample_data and column_names:
                feature_columns = column_names[:-1]  # Exclude target column
                
                for i, row in enumerate(sample_data[:3]):  # First 3 examples
                    example_values = []
                    for col in feature_columns:
                        if col in row:
                            value = row[col]
                            if isinstance(value, (int, float)):
                                example_values.append(str(value))
                            else:
                                example_values.append(str(value).strip())
                    
                    if example_values:
                        examples.append(', '.join(example_values))
            
            # Model-specific guidance
            model_type = metadata.get('model_type', 'unknown')
            guidance_notes = {
                'perceptron': "This Perceptron model expects linearly separable binary classification data.",
                'mlp': "This MLP model can handle complex non-linear patterns in the data.",
                'cnn': "This CNN model was trained on tabular data (unusual - typically used for images)."
            }
            
            return {
                'success': True,
                'feature_names': column_names[:-1] if column_names else [],
                'target_name': column_names[-1] if column_names else 'target',
                'num_features': len(column_names) - 1 if column_names else 0,
                'examples': examples[:3],
                'input_format': f"Enter {len(column_names) - 1 if column_names else '?'} comma-separated numeric values",
                'model_type': model_type,
                'guidance_note': guidance_notes.get(model_type, "Enter values matching the training data format.")
            }
            
        except Exception as e:
            error_msg = f"Error getting training data info: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def get_model_summary(self):
        """Get comprehensive model summary"""
        if not self.loaded_model:
            return {'success': False, 'error': 'No model loaded'}
        
        try:
            # Get model summary as string
            summary_lines = []
            self.loaded_model.summary(print_fn=lambda x: summary_lines.append(x))
            summary_text = '\n'.join(summary_lines)
            
            # Enhanced summary with model type information
            model_type = self.model_metadata.get('model_type', 'unknown')
            enhanced_summary = f"Model Type: {model_type.upper()}\n"
            enhanced_summary += f"Architecture: {self.model_metadata.get('model_architecture', 'Unknown')}\n"
            enhanced_summary += f"Task: {self.model_metadata.get('task_type', 'Unknown')}\n"
            enhanced_summary += f"Data Type: {self.model_metadata.get('data_type', 'Unknown')}\n"
            enhanced_summary += f"Status: {'Loaded Successfully' if self.loaded_model else 'Not Loaded'}\n\n"
            enhanced_summary += "Model Architecture:\n"
            enhanced_summary += summary_text
            
            # Add model-specific characteristics
            characteristics = {
                'perceptron': [
                    "- Single layer with linear decision boundary",
                    "- Best for linearly separable binary classification",
                    "- Cannot solve XOR-like problems",
                    "- Fast training and prediction",
                    "- Minimal memory requirements"
                ],
                'mlp': [
                    "- Multiple hidden layers for complex patterns",
                    "- Universal function approximator",
                    "- Can handle non-linear decision boundaries",
                    "- Suitable for both classification and regression",
                    "- Requires careful hyperparameter tuning"
                ],
                'cnn': [
                    "- Convolutional layers for spatial feature extraction",
                    "- Translation invariant features",
                    "- Hierarchical feature learning",
                    "- Optimized for image and spatial data",
                    "- Parameter sharing reduces overfitting"
                ]
            }
            
            if model_type in characteristics:
                enhanced_summary += f"\n\n{model_type.upper()} Characteristics:\n"
                enhanced_summary += '\n'.join(characteristics[model_type])
            
            return {
                'success': True,
                'summary': enhanced_summary,
                'basic_summary': summary_text,
                'input_shape': self.loaded_model.input_shape,
                'output_shape': self.loaded_model.output_shape,
                'total_params': self.loaded_model.count_params(),
                'metadata': self.model_metadata,
                'model_type': model_type,
                'preprocessing_info': {
                    'scaler_available': self.scaler is not None,
                    'label_encoder_available': self.label_encoder is not None
                }
            }
            
        except Exception as e:
            error_msg = f"Error getting model summary: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def _cleanup(self):
        """Clean up loaded model and preprocessing objects"""
        try:
            if self.loaded_model:
                del self.loaded_model
                self.loaded_model = None
            
            self.model_metadata = None
            self.scaler = None
            self.label_encoder = None
            self.session_id = None
            self.input_shape = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.info("Model cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self._cleanup()


class ModelSessionManager:
    """Manage multiple model testing sessions"""
    
    def __init__(self, models_folder='trained_models', uploads_folder='uploads'):
        self.models_folder = models_folder
        self.uploads_folder = uploads_folder
        self.active_sessions = {}
        self.max_sessions = 5  # Limit concurrent sessions
    
    def create_session(self, session_id=None):
        """Create a new model testing session"""
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Clean old sessions if at limit
        if len(self.active_sessions) >= self.max_sessions:
            self._cleanup_oldest_session()
        
        # Create new session
        tester = ModelTester(self.models_folder, self.uploads_folder)
        self.active_sessions[session_id] = {
            'tester': tester,
            'created_at': datetime.now(),
            'last_used': datetime.now()
        }
        
        return session_id
    
    def get_session(self, session_id):
        """Get existing session"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]['last_used'] = datetime.now()
            return self.active_sessions[session_id]['tester']
        return None
    
    def close_session(self, session_id):
        """Close and cleanup session"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]['tester']._cleanup()
            del self.active_sessions[session_id]
    
    def _cleanup_oldest_session(self):
        """Remove oldest session"""
        if self.active_sessions:
            oldest_session = min(self.active_sessions.items(), 
                               key=lambda x: x[1]['last_used'])
            self.close_session(oldest_session[0])
    
    def cleanup_all(self):
        """Cleanup all sessions"""
        for session_id in list(self.active_sessions.keys()):
            self.close_session(session_id)


class ModelCompatibilityChecker:
    """Enhanced compatibility checker"""
    
    @staticmethod
    def check_compatibility(data_info, model_metadata):
        """Check compatibility between data and model"""
        issues = []
        warnings = []
        recommendations = []
        
        data_type = data_info.get('data_type', 'unknown')
        model_data_type = model_metadata.get('data_type', 'unknown')
        
        # Check data type compatibility
        if data_type != model_data_type and model_data_type != 'unknown':
            issues.append(f"Data type mismatch: expecting {model_data_type}, got {data_type}")
        
        # Check input shape compatibility
        if 'input_shape' in data_info and 'input_shape' in model_metadata:
            expected_shape = model_metadata['input_shape']
            actual_shape = data_info['input_shape']
            
            if expected_shape != actual_shape:
                issues.append(f"Input shape mismatch: expecting {expected_shape}, got {actual_shape}")
        
        # Model-specific checks
        model_type = model_metadata.get('model_type', 'unknown')
        
        if model_type == 'perceptron':
            if data_info.get('data_type') == 'images':
                issues.append("Perceptron models are not suitable for image data")
                recommendations.append("Use CNN for image classification")
            
            if model_metadata.get('num_classes', 2) > 2:
                warnings.append("Perceptron works best with binary classification")
        
        elif model_type == 'cnn':
            if data_info.get('data_type') != 'images':
                warnings.append("CNN models are optimized for image data")
        
        return {
            'compatible': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'recommendations': recommendations
        }
    
    @staticmethod
    def get_model_recommendations(data_info):
        """Get model recommendations based on data characteristics"""
        data_type = data_info.get('data_type', 'unknown')
        task_type = data_info.get('task_type', 'classification')
        num_classes = data_info.get('num_classes', 1)
        
        recommendations = []
        
        if data_type == 'images':
            recommendations.append({
                'model': 'CNN',
                'reason': 'Convolutional layers are optimal for spatial data and image processing',
                'confidence': 'High',
                'priority': 1
            })
            recommendations.append({
                'model': 'MLP',
                'reason': 'Can work with flattened image data but less optimal',
                'confidence': 'Medium',
                'priority': 2
            })
        elif data_type == 'tabular':
            if task_type == 'classification' and num_classes == 2:
                recommendations.extend([
                    {
                        'model': 'MLP',
                        'reason': 'Handles non-linear patterns effectively',
                        'confidence': 'High',
                        'priority': 1,
                        'note': 'Best overall choice for complex data'
                    },
                    {
                        'model': 'Perceptron',
                        'reason': 'Simple and fast for binary classification',
                        'confidence': 'Medium',
                        'priority': 2,
                        'note': 'Only if data is linearly separable'
                    }
                ])
            else:
                recommendations.append({
                    'model': 'MLP',
                    'reason': 'Versatile for complex tabular data patterns',
                    'confidence': 'High',
                    'priority': 1
                })
        
        # Sort by priority
        recommendations.sort(key=lambda x: x.get('priority', 999))
        
        return recommendations