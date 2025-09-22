import tensorflow as tf
import onnx
import tf2onnx
import os
import zipfile
import shutil
from datetime import datetime

class ModelExporter:
    def __init__(self):
        self.supported_formats = ['tensorflow', 'onnx', 'keras', 'tflite']
    
    def export_model(self, model, format_type, output_dir):
        """Export trained model to specified format"""
        if format_type not in self.supported_formats:
            raise ValueError(f"Unsupported export format: {format_type}")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format_type == 'tensorflow':
            return self._export_tensorflow(model, output_dir, timestamp)
        elif format_type == 'onnx':
            return self._export_onnx(model, output_dir, timestamp)
        elif format_type == 'keras':
            return self._export_keras(model, output_dir, timestamp)
        elif format_type == 'tflite':
            return self._export_tflite(model, output_dir, timestamp)
        else:
            raise ValueError(f"Export format {format_type} not implemented")
    
    def _export_tensorflow(self, model, output_dir, timestamp):
        """Export as TensorFlow SavedModel format"""
        model_dir = os.path.join(output_dir, f'tensorflow_model_{timestamp}')
        os.makedirs(model_dir, exist_ok=True)
        
        # Save the model
        saved_model_path = os.path.join(model_dir, 'saved_model')
        model.save(saved_model_path, save_format='tf')
        
        # Create metadata file
        metadata = {
            'export_timestamp': timestamp,
            'model_type': 'tensorflow_saved_model',
            'framework_version': tf.__version__,
            'model_architecture': self._get_model_summary(model),
            'input_shape': str(model.input_shape) if hasattr(model, 'input_shape') else 'unknown',
            'output_shape': str(model.output_shape) if hasattr(model, 'output_shape') else 'unknown'
        }
        
        metadata_path = os.path.join(model_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            import json
            json.dump(metadata, f, indent=2)
        
        # Create usage instructions
        instructions = self._create_tensorflow_usage_instructions()
        instructions_path = os.path.join(model_dir, 'README.md')
        with open(instructions_path, 'w') as f:
            f.write(instructions)
        
        # Create ZIP archive
        zip_path = os.path.join(output_dir, f'tensorflow_model_{timestamp}.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(model_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, model_dir)
                    zipf.write(file_path, arcname)
        
        # Clean up temporary directory
        shutil.rmtree(model_dir)
        
        return zip_path
    
    def _export_onnx(self, model, output_dir, timestamp):
        """Export as ONNX format"""
        try:
            # Create output path
            onnx_path = os.path.join(output_dir, f'model_{timestamp}.onnx')
            
            # Convert TensorFlow model to ONNX
            spec = (tf.TensorSpec(model.input_shape, tf.float32, name="input"),)
            output_path = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=onnx_path)
            
            # Create metadata file
            metadata_path = os.path.join(output_dir, f'model_{timestamp}_metadata.json')
            metadata = {
                'export_timestamp': timestamp,
                'model_type': 'onnx',
                'framework_version': tf.__version__,
                'onnx_version': onnx.__version__,
                'model_architecture': self._get_model_summary(model),
                'input_shape': str(model.input_shape) if hasattr(model, 'input_shape') else 'unknown',
                'output_shape': str(model.output_shape) if hasattr(model, 'output_shape') else 'unknown'
            }
            
            with open(metadata_path, 'w') as f:
                import json
                json.dump(metadata, f, indent=2)
            
            # Create usage instructions
            instructions = self._create_onnx_usage_instructions()
            instructions_path = os.path.join(output_dir, f'model_{timestamp}_README.md')
            with open(instructions_path, 'w') as f:
                f.write(instructions)
            
            return onnx_path
            
        except Exception as e:
            raise Exception(f"ONNX export failed: {str(e)}")
    
    def _export_keras(self, model, output_dir, timestamp):
        """Export as Keras .h5 format"""
        keras_path = os.path.join(output_dir, f'keras_model_{timestamp}.h5')
        
        # Save the model
        model.save(keras_path)
        
        # Create metadata
        metadata_path = os.path.join(output_dir, f'keras_model_{timestamp}_metadata.json')
        metadata = {
            'export_timestamp': timestamp,
            'model_type': 'keras_h5',
            'framework_version': tf.__version__,
            'model_architecture': self._get_model_summary(model),
            'input_shape': str(model.input_shape) if hasattr(model, 'input_shape') else 'unknown',
            'output_shape': str(model.output_shape) if hasattr(model, 'output_shape') else 'unknown'
        }
        
        with open(metadata_path, 'w') as f:
            import json
            json.dump(metadata, f, indent=2)
        
        return keras_path
    
    def _export_tflite(self, model, output_dir, timestamp):
        """Export as TensorFlow Lite format"""
        try:
            # Convert to TensorFlow Lite
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            # Optional optimizations
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            tflite_model = converter.convert()
            
            # Save the model
            tflite_path = os.path.join(output_dir, f'model_{timestamp}.tflite')
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            # Create metadata
            metadata_path = os.path.join(output_dir, f'tflite_model_{timestamp}_metadata.json')
            metadata = {
                'export_timestamp': timestamp,
                'model_type': 'tensorflow_lite',
                'framework_version': tf.__version__,
                'model_size_bytes': len(tflite_model),
                'model_architecture': self._get_model_summary(model),
                'optimizations_applied': ['DEFAULT'],
                'input_shape': str(model.input_shape) if hasattr(model, 'input_shape') else 'unknown',
                'output_shape': str(model.output_shape) if hasattr(model, 'output_shape') else 'unknown'
            }
            
            with open(metadata_path, 'w') as f:
                import json
                json.dump(metadata, f, indent=2)
            
            return tflite_path
            
        except Exception as e:
            raise Exception(f"TensorFlow Lite export failed: {str(e)}")
    
    def _get_model_summary(self, model):
        """Get model architecture summary as string"""
        try:
            summary_lines = []
            model.summary(print_fn=lambda x: summary_lines.append(x))
            return '\n'.join(summary_lines)
        except:
            return "Model summary not available"
    
    def _create_tensorflow_usage_instructions(self):
        """Create usage instructions for TensorFlow SavedModel"""
        return """# TensorFlow SavedModel Usage Instructions

## Loading the Model

```python
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('saved_model')

# Or using tf.saved_model for more control
model = tf.saved_model.load('saved_model')
```

## Making Predictions

```python
import numpy as np

# Prepare your input data (adjust shape according to your model)
input_data = np.array([[...]])  # Replace with your data

# Make predictions
predictions = model.predict(input_data)
print(predictions)
```

## Model Information

Check the metadata.json file for detailed information about:
- Model architecture
- Input/output shapes
- Export timestamp
- Framework version

## Requirements

- TensorFlow >= 2.0
- NumPy
- Any other dependencies used during training

## Notes

- Ensure input data is preprocessed the same way as during training
- The model expects inputs in the same format as during training
- Check the input shape in metadata.json for correct dimensions
"""

    def _create_onnx_usage_instructions(self):
        """Create usage instructions for ONNX model"""
        return """# ONNX Model Usage Instructions

## Installation

```bash
pip install onnxruntime
# For GPU support (optional)
pip install onnxruntime-gpu
```

## Loading and Using the Model

```python
import onnxruntime as ort
import numpy as np

# Create inference session
session = ort.InferenceSession('model.onnx')

# Get input/output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Prepare input data (adjust according to your model)
input_data = np.array([[...]], dtype=np.float32)  # Replace with your data

# Run inference
result = session.run([output_name], {input_name: input_data})
predictions = result[0]
print(predictions)
```

## Model Information

Check the metadata file for:
- Model architecture details
- Input/output shapes
- Export information

## Platform Compatibility

ONNX models can be used with:
- ONNX Runtime (Python, C++, C#, Java)
- Various inference engines
- Different hardware platforms

## Notes

- Ensure input data preprocessing matches training
- ONNX models are platform-independent
- Check input data types (usually float32)
"""

    def create_deployment_package(self, model, export_format, output_dir, include_examples=True):
        """Create a complete deployment package"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        package_dir = os.path.join(output_dir, f'deployment_package_{timestamp}')
        os.makedirs(package_dir, exist_ok=True)
        
        # Export the model
        model_path = self.export_model(model, export_format, package_dir)
        
        # Create deployment script
        deployment_script = self._create_deployment_script(export_format)
        script_path = os.path.join(package_dir, 'deploy.py')
        with open(script_path, 'w') as f:
            f.write(deployment_script)
        
        # Create requirements file
        requirements = self._create_requirements_file(export_format)
        req_path = os.path.join(package_dir, 'requirements.txt')
        with open(req_path, 'w') as f:
            f.write(requirements)
        
        # Create example usage if requested
        if include_examples:
            example_script = self._create_example_script(export_format)
            example_path = os.path.join(package_dir, 'example_usage.py')
            with open(example_path, 'w') as f:
                f.write(example_script)
        
        # Create ZIP package
        zip_path = os.path.join(output_dir, f'deployment_package_{timestamp}.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(package_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, package_dir)
                    zipf.write(file_path, arcname)
        
        # Clean up
        shutil.rmtree(package_dir)
        
        return zip_path
    
    def _create_deployment_script(self, export_format):
        """Create a deployment script template"""
        if export_format == 'tensorflow':
            return """#!/usr/bin/env python3
'''
TensorFlow Model Deployment Script
Generated by Neural Network Trainer
'''

import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('saved_model')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = np.array(data['input'])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        return jsonify({
            'prediction': prediction.tolist(),
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
"""
        else:
            return "# Deployment script template not available for this format"
    
    def _create_requirements_file(self, export_format):
        """Create requirements.txt for deployment"""
        if export_format == 'tensorflow':
            return """tensorflow>=2.10.0
numpy>=1.21.0
flask>=2.0.0
"""
        elif export_format == 'onnx':
            return """onnxruntime>=1.12.0
numpy>=1.21.0
flask>=2.0.0
"""
        else:
            return "# Requirements for this format not specified"
    
    def _create_example_script(self, export_format):
        """Create example usage script"""
        return """#!/usr/bin/env python3
'''
Example usage of the exported model
'''

import numpy as np

# Example input data (replace with your actual data structure)
example_input = np.random.random((1, 10))  # Adjust shape as needed

print("Example input shape:", example_input.shape)
print("Example input:", example_input)

# Load and use your model here
# (See the README.md for specific loading instructions)

print("\\nReplace this with actual model loading and prediction code")
print("Check the README.md file for detailed usage instructions")
"""
