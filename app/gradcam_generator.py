import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class GradCAMGenerator:
    """
    🎓 Grad-CAM: Shows what the model focuses on in images
    """
    
    def __init__(self, model, class_names, img_size=(256, 256)):
        self.model = model
        self.class_names = class_names
        self.img_size = img_size
        self.num_classes = len(class_names)
    
    def load_and_preprocess_image(self, image_path):
        """
        تحميل ومعالجة الصورة
        """
        try:
            # تحميل الصورة
            img = keras.preprocessing.image.load_img(
                image_path, 
                target_size=self.img_size
            )
            
            # تحويل إلى array
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            # تطبيع (معايير EfficientNetV2)
            img_preprocessed = tf.keras.applications.efficientnet_v2.preprocess_input(
                img_array.copy()
            )
            
            return img_array, img_preprocessed
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return None, None
    
    def make_gradcam_heatmap(self, img_preprocessed, pred_class_idx):
        """
        حساب Grad-CAM heatmap
        """
        try:
            # إيجاد آخر طبقة Conv
            last_conv_layer = None
            for layer in reversed(self.model.layers):
                if isinstance(layer, keras.layers.Conv2D):
                    last_conv_layer = layer
                    break
                # إذا كانت طبقة تحتوي على طبقات فرعية
                if hasattr(layer, 'layers'):
                    for sublayer in reversed(layer.layers):
                        if isinstance(sublayer, keras.layers.Conv2D):
                            last_conv_layer = sublayer
                            break
                    if last_conv_layer:
                        break
            
            if last_conv_layer is None:
                logger.warning("No Conv2D layer found, using fallback method")
                return self._simple_saliency_map(img_preprocessed, pred_class_idx)
            
            logger.info(f"Using layer: {last_conv_layer.name}")
            
            # إنشاء نموذج مخصص
            grad_model = keras.Model(
                inputs=self.model.input,
                outputs=[last_conv_layer.output, self.model.output]
            )
            
            # حساب gradients
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_preprocessed, training=False)
                class_channel = predictions[:, pred_class_idx]
            
            # الـ gradients
            grads = tape.gradient(class_channel, conv_outputs)
            
            if grads is None:
                logger.warning("Gradients are None, using fallback method")
                return self._simple_saliency_map(img_preprocessed, pred_class_idx)
            
            # حساب الأوزان
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # إنشاء heatmap
            conv_output = conv_outputs[0]
            heatmap = conv_output @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            
            # تطبيع
            heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
            
            return heatmap.numpy()
        
        except Exception as e:
            logger.error(f"Error in make_gradcam_heatmap: {str(e)}")
            return self._simple_saliency_map(img_preprocessed, pred_class_idx)
    
    def _simple_saliency_map(self, img_preprocessed, pred_class_idx):
        """
        طريقة بديلة: Saliency map بسيطة
        """
        try:
            with tf.GradientTape() as tape:
                img_tensor = tf.cast(img_preprocessed, tf.float32)
                tape.watch(img_tensor)
                predictions = self.model(img_tensor, training=False)
                class_channel = predictions[:, pred_class_idx]
            
            grads = tape.gradient(class_channel, img_tensor)
            saliency = tf.reduce_max(tf.abs(grads), axis=-1)
            saliency = saliency[0]
            saliency = (saliency - tf.reduce_min(saliency)) / (tf.reduce_max(saliency) - tf.reduce_min(saliency) + 1e-10)
            
            return saliency.numpy()
        except Exception as e:
            logger.error(f"Error in saliency map: {str(e)}")
            # إرجع خريطة سوداء في أسوأ الحالات
            return np.zeros((self.img_size[0], self.img_size[1]))
    
    def process_image(self, image_path):
        """
        معالجة كاملة لصورة واحدة
        """
        try:
            # تحميل الصورة
            img_array, img_preprocessed = self.load_and_preprocess_image(image_path)
            
            if img_array is None:
                return None
            
            # التنبؤ
            preds = self.model.predict(img_preprocessed, verbose=0)[0]
            pred_class_idx = np.argmax(preds)
            confidence = preds[pred_class_idx]
            
            # حساب Grad-CAM
            heatmap = self.make_gradcam_heatmap(img_preprocessed, pred_class_idx)
            
            # تحميل الصورة الأصلية
            original_img = keras.preprocessing.image.load_img(image_path)
            original_array = keras.preprocessing.image.img_to_array(original_img) / 255.0
            
            # تكبير heatmap
            heatmap_resized = cv2.resize(
                heatmap, 
                (original_array.shape[1], original_array.shape[0])
            )
            
            # تلوين heatmap
            heatmap_colored = cv2.applyColorMap(
                np.uint8(255 * heatmap_resized), 
                cv2.COLORMAP_JET
            )
            heatmap_colored = heatmap_colored / 255.0
            
            # دمج الصور
            overlay = original_array * 0.5 + heatmap_colored * 0.5
            
            return {
                'original': original_array,
                'heatmap': heatmap_resized,
                'overlay': overlay,
                'predictions': preds,
                'pred_class_idx': int(pred_class_idx),
                'confidence': float(confidence),
                'class_name': self.class_names[pred_class_idx]
            }
        
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return None
    
    def convert_to_base64(self, img_array):
        """
        تحويل صورة numpy إلى base64 للـ JSON
        """
        try:
            import base64
            from io import BytesIO
            from PIL import Image
            
            # تحويل إلى uint8
            img_uint8 = np.uint8(img_array * 255)
            
            # تحويل إلى صورة PIL
            pil_img = Image.fromarray(img_uint8)
            
            # حفظ في BytesIO
            buffer = BytesIO()
            pil_img.save(buffer, format='JPEG', quality=85)
            
            # تحويل إلى base64
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/jpeg;base64,{img_base64}"
        
        except Exception as e:
            logger.error(f"Error converting to base64: {str(e)}")
            return None
    
    def generate_gradcam_samples(self, image_paths, output_dir):
        """
        معالجة مجموعة صور وحفظ النتائج
        """
        try:
            results = []
            
            for idx, img_path in enumerate(image_paths):
                logger.info(f"Processing image {idx + 1}/{len(image_paths)}: {os.path.basename(img_path)}")
                
                result = self.process_image(img_path)
                
                if result is None:
                    logger.warning(f"Failed to process {img_path}")
                    continue
                
                # تحويل إلى base64
                result['original_base64'] = self.convert_to_base64(result['original'])
                result['heatmap_base64'] = self.convert_to_base64(
                    cv2.applyColorMap(np.uint8(255 * result['heatmap']), cv2.COLORMAP_JET) / 255.0
                )
                result['overlay_base64'] = self.convert_to_base64(result['overlay'])
                
                # حذف البيانات الكبيرة (arrays)
                del result['original']
                del result['heatmap']
                del result['overlay']
                
                # تحويل التنبؤات
                result['all_predictions'] = {
                    self.class_names[i]: float(result['predictions'][i]) 
                    for i in range(len(self.class_names))
                }
                del result['predictions']
                
                results.append(result)
            
            # حفظ البيانات
            os.makedirs(output_dir, exist_ok=True)
            
            output_data = {
                'generated_at': datetime.now().isoformat(),
                'num_samples': len(results),
                'samples': results
            }
            
            json_path = os.path.join(output_dir, 'gradcam_data.json')
            with open(json_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"Saved {len(results)} samples to {json_path}")
            
            return output_data
        
        except Exception as e:
            logger.error(f"Error generating Grad-CAM samples: {str(e)}")
            return None