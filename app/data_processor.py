import pandas as pd
import numpy as np
import os
import zipfile
from PIL import Image
import json
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

try:
    import cv2
except ImportError:
    cv2 = None
    print("Warning: OpenCV not available. Image processing will be limited.")

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def process_file(self, filepath, session_id=None):
        """Process uploaded file and return preview data"""
        file_extension = os.path.splitext(filepath)[1].lower()
        
        if file_extension == '.csv':
            return self._process_csv(filepath)
        elif file_extension in ['.zip']:
            return self._process_image_zip(filepath, session_id)
        elif file_extension in ['.png', '.jpg', '.jpeg']:
            return self._process_single_image(filepath, session_id)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def _process_csv(self, filepath):
        """Process CSV file"""
        try:
            # Read CSV with error handling
            df = pd.read_csv(filepath)
            
            # Basic info
            rows, columns = df.shape
            
            # Sample data for preview
            sample_data = df.head(10).fillna('').to_dict('records')
            
            # Detect data types and potential target column
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
            
            preview_data = {
                'type': 'csv',
                'rows': rows,
                'columns': columns,
                'sample': sample_data,
                'numeric_columns': numeric_columns,
                'categorical_columns': categorical_columns,
                'column_names': df.columns.tolist()
            }
            
            return preview_data
            
        except Exception as e:
            raise Exception(f"Error processing CSV file: {str(e)}")
    
    def _process_image_zip(self, filepath, session_id=None):
        """Process ZIP file containing images"""
        try:
            extract_path = filepath.replace('.zip', '_extracted')
            os.makedirs(extract_path, exist_ok=True)
            
            # Extract ZIP file
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            
            # Find image files
            image_files = []
            image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
            
            for root, dirs, files in os.walk(extract_path):
                for file in files:
                    if os.path.splitext(file)[1].lower() in image_extensions:
                        image_files.append(os.path.join(root, file))
            
            if not image_files:
                raise Exception("No image files found in ZIP archive")
            
            # Detect class structure (if organized in folders)
            classes = set()
            for img_path in image_files:
                parent_dir = os.path.basename(os.path.dirname(img_path))
                if parent_dir != os.path.basename(extract_path):
                    classes.add(parent_dir)
            
            # Extract session_id from filepath if not provided
            if session_id is None:
                session_id = os.path.basename(filepath).split('_')[0]
            
            # Sample random images for preview (up to 8)
            # Try to get a balanced sample from different classes if they exist
            sample_images = []
            max_samples = 8
            
            if classes and len(classes) > 1:
                # Group images by class
                images_by_class = {}
                for img_path in image_files:
                    class_name = os.path.basename(os.path.dirname(img_path))
                    if class_name != os.path.basename(extract_path):  # Skip if it's the root folder
                        if class_name not in images_by_class:
                            images_by_class[class_name] = []
                        images_by_class[class_name].append(img_path)
                
                # Try to sample evenly from each class
                images_per_class = max(1, max_samples // len(classes))
                remaining_slots = max_samples
                
                # First pass: get at least one image from each class
                for class_name, class_images in images_by_class.items():
                    if remaining_slots <= 0:
                        break
                    
                    # Sample from this class
                    sample_size = min(images_per_class, len(class_images), remaining_slots)
                    selected_images = random.sample(class_images, sample_size)
                    
                    for img_path in selected_images:
                        try:
                            filename = os.path.basename(img_path)
                            sample_images.append({
                                'url': f'/api/preview-image/{session_id}/{filename}',
                                'label': class_name
                            })
                            remaining_slots -= 1
                        except:
                            continue
                
                # Second pass: fill remaining slots randomly
                if remaining_slots > 0:
                    all_remaining_images = []
                    for class_images in images_by_class.values():
                        all_remaining_images.extend(class_images)
                    
                    # Remove already selected images
                    selected_filenames = {sample['url'].split('/')[-1] for sample in sample_images}
                    remaining_images = [img for img in all_remaining_images 
                                      if os.path.basename(img) not in selected_filenames]
                    
                    if remaining_images:
                        additional_count = min(remaining_slots, len(remaining_images))
                        additional_images = random.sample(remaining_images, additional_count)
                        
                        for img_path in additional_images:
                            try:
                                filename = os.path.basename(img_path)
                                class_name = os.path.basename(os.path.dirname(img_path))
                                sample_images.append({
                                    'url': f'/api/preview-image/{session_id}/{filename}',
                                    'label': class_name
                                })
                            except:
                                continue
            else:
                # No class structure or single class - just random sample
                sample_count = min(max_samples, len(image_files))
                random_image_files = random.sample(image_files, sample_count)
                
                for i, img_path in enumerate(random_image_files):
                    try:
                        filename = os.path.basename(img_path)
                        class_label = os.path.basename(os.path.dirname(img_path)) if classes else f'Image {i+1}'
                        sample_images.append({
                            'url': f'/api/preview-image/{session_id}/{filename}',
                            'label': class_label
                        })
                    except:
                        continue
            
            preview_data = {
                'type': 'images',
                'count': len(image_files),
                'classes': len(classes) if classes else None,
                'class_names': list(classes) if classes else None,
                'samples': sample_images,
                'extract_path': extract_path
            }
            
            return preview_data
            
        except Exception as e:
            raise Exception(f"Error processing image ZIP: {str(e)}")
    
    def _process_single_image(self, filepath, session_id=None):
        """Process single image file"""
        try:
            img = Image.open(filepath)
            width, height = img.size
            
            # Extract session_id from filepath if not provided
            if session_id is None:
                session_id = os.path.basename(filepath).split('_')[0]
            
            preview_data = {
                'type': 'images',
                'count': 1,
                'classes': None,
                'samples': [{
                    'url': f'/api/preview-image/{session_id}/{os.path.basename(filepath)}',
                    'label': 'Single Image'
                }],
                'dimensions': f"{width}x{height}",
                'mode': img.mode,
                'extract_path': os.path.dirname(filepath)  # For consistency with zip processing
            }
            
            return preview_data
            
        except Exception as e:
            raise Exception(f"Error processing image: {str(e)}")
    
    def prepare_training_data(self, filepath, validation_split=0.2, task_type='classification'):
        """Prepare data for training"""
        preview_data = self.process_file(filepath)
        
        if preview_data['type'] == 'csv':
            return self._prepare_csv_data(filepath, validation_split, task_type)
        elif preview_data['type'] == 'images':
            return self._prepare_image_data(preview_data, validation_split, task_type)
        else:
            raise ValueError("Unsupported data type for training")
    
    def _prepare_csv_data(self, filepath, validation_split, task_type):
        """Prepare CSV data for training"""
        df = pd.read_csv(filepath)
        
        feature_columns = df.columns[:-1]
        target_column = df.columns[-1]
        
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Handle categorical features
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            X.loc[:, col] = LabelEncoder().fit_transform(X[col].astype(str))
        
        # ✅ حفظ أسماء الفئات الأصلية
        original_class_names = None
        if task_type == 'classification':
            if y.dtype == 'object':
                # ✅ احتفظ بالأسماء الأصلية قبل التحويل
                original_class_names = sorted(y.unique().tolist())
                y = self.label_encoder.fit_transform(y)
            else:
                # إذا كانت رقمية، احتفظ بالقيم الفريدة
                original_class_names = sorted(y.unique().tolist())
            
            # إعادة ترقيم التصنيفات
            unique_classes = np.unique(y)
            num_classes = len(unique_classes)
            
            y_mapped = np.zeros_like(y)
            for new_idx, old_class in enumerate(unique_classes):
                y_mapped[y == old_class] = new_idx
            
            y = y_mapped.astype(np.int32)
            
            print(f"📊 Original classes: {original_class_names}")
            print(f"📊 Mapped to: {np.arange(num_classes)}")
            
        else:  # regression
            y = pd.to_numeric(y, errors='coerce')
            if y.isna().sum() > 0:
                valid_indices = ~y.isna()
                X = X[valid_indices]
                y = y[valid_indices]
            y = y.values.astype(np.float32)
            num_classes = 1
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, 
            stratify=y if task_type == 'classification' else None
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        data_info = {
            'input_shape': X_train_scaled.shape[1],
            'num_classes': num_classes,
            'task_type': task_type,
            'data_type': 'tabular',
            'class_names': original_class_names,  # ✅ الأسماء الأصلية
            'feature_names': feature_columns.tolist(),
            'target_name': target_column
        }
        
        return X_train_scaled, X_val_scaled, y_train, y_val, data_info
    def _prepare_image_data(self, preview_data, validation_split, task_type):
        """Prepare image data for training"""
        if 'extract_path' in preview_data:
            extract_path = preview_data['extract_path']
        else:
            # Handle single image case
            raise Exception("Single image training not implemented yet")
        
        # Load images and labels
        images = []
        labels = []
        
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        target_size = (224, 224)  # Standard size for most models
        
        # If organized in folders (class structure)
        if preview_data.get('class_names'):
            label_to_idx = {label: idx for idx, label in enumerate(preview_data['class_names'])}
            
            for root, dirs, files in os.walk(extract_path):
                class_name = os.path.basename(root)
                if class_name in label_to_idx:
                    for file in files:
                        if os.path.splitext(file)[1].lower() in image_extensions:
                            img_path = os.path.join(root, file)
                            try:
                                if cv2 is not None:
                                    img = cv2.imread(img_path)
                                    img = cv2.resize(img, target_size)
                                    img = img / 255.0  # Normalize
                                else:
                                    # Fallback to PIL
                                    img = Image.open(img_path)
                                    img = img.resize(target_size)
                                    img = np.array(img) / 255.0
                                    if len(img.shape) == 2:  # Grayscale
                                        img = np.stack([img] * 3, axis=-1)  # Convert to RGB
                                
                                images.append(img)
                                labels.append(label_to_idx[class_name])
                            except Exception as e:
                                print(f"Error processing image {img_path}: {e}")
                                continue
        else:
            # No class structure - create dummy labels for unsupervised/autoencoder tasks
            for root, dirs, files in os.walk(extract_path):
                for file in files:
                    if os.path.splitext(file)[1].lower() in image_extensions:
                        img_path = os.path.join(root, file)
                        try:
                            if cv2 is not None:
                                img = cv2.imread(img_path)
                                img = cv2.resize(img, target_size)
                                img = img / 255.0
                            else:
                                # Fallback to PIL
                                img = Image.open(img_path)
                                img = img.resize(target_size)
                                img = np.array(img) / 255.0
                                if len(img.shape) == 2:  # Grayscale
                                    img = np.stack([img] * 3, axis=-1)  # Convert to RGB
                            
                            images.append(img)
                            labels.append(0)  # Dummy label
                        except Exception as e:
                            print(f"Error processing image {img_path}: {e}")
                            continue
        
        if not images:
            raise Exception("No valid images found for training")
        
        X = np.array(images)
        y = np.array(labels)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        data_info = {
            'input_shape': X_train.shape[1:],  # (height, width, channels)
            'num_classes': len(np.unique(y)) if task_type == 'classification' else 1,
            'task_type': task_type,
            'data_type': 'images'
        }
        
        return X_train, X_val, y_train, y_val, data_info
