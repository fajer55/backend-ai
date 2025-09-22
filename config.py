# Configuration file for the Neural Network Trainer

# Flask Configuration
DEBUG = True
SECRET_KEY = 'your-secret-key-here'
MAX_CONTENT_LENGTH = 1024 * 1024 * 1024  # 1000MB

# File Upload Settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'png', 'jpg', 'jpeg', 'zip'}

# Model Settings
DEFAULT_MODEL_CONFIG = {
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.001,
    'validation_split': 0.2,
    'optimizer': 'adam'
}

# Training Settings
MAX_EPOCHS = 1000
MIN_EPOCHS = 1
MAX_BATCH_SIZE = 512
MIN_BATCH_SIZE = 1

# Export Settings
EXPORT_FORMATS = ['tensorflow', 'onnx', 'keras', 'tflite']
MODEL_SAVE_PATH = 'trained_models'

# Performance Settings
ENABLE_GPU = True
MEMORY_LIMIT = 8192  # MB

# Logging
LOG_LEVEL = 'INFO'
LOG_FILE = 'neural_trainer.log'
