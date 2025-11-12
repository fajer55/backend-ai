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
    Improved version based on working Colab implementation
    """

    def __init__(self, model, class_names, img_size=(256, 256)):
        self.model = model
        self.class_names = class_names
        self.img_size = img_size
        self.num_classes = len(class_names)
        logger.info(f"✅ GradCAMGenerator initialized with {self.num_classes} classes")

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

    def make_gradcam_heatmap(self, img_array, model):
        """
        Grad-CAM مبسط - يعمل مع أي نموذج
        Based on working Colab implementation ✅
        """
        try:
            # معالجة الصورة
            img_preprocessed = tf.keras.applications.efficientnet_v2.preprocess_input(img_array.copy())

            # إيجاد آخر طبقة Conv في النموذج
            last_conv_layer = None
            for layer in reversed(model.layers):
                # إذا كانت طبقة تحتوي على طبقات فرعية (base_model)
                if hasattr(layer, 'layers'):
                    for sublayer in reversed(layer.layers):
                        if isinstance(sublayer, keras.layers.Conv2D):
                            last_conv_layer = sublayer
                            break
                    if last_conv_layer:
                        break
                # أو إذا كانت Conv2D مباشرة
                elif isinstance(layer, keras.layers.Conv2D):
                    last_conv_layer = layer
                    break

            if last_conv_layer is None:
                logger.warning("No Conv2D layer found! Using fallback method")
                return self._fallback_saliency_map(img_preprocessed), None

            logger.info(f"  ✓ Using Conv layer: {last_conv_layer.name}")

            # إنشاء نموذج للوصول إلى الطبقة
            outputs = []

            # نحفظ الـ output أثناء forward pass
            original_call = last_conv_layer.call

            def new_call(inputs, **kwargs):
                x = original_call(inputs, **kwargs)
                outputs.append(x)
                return x

            last_conv_layer.call = new_call

            # Forward pass مع GradientTape
            with tf.GradientTape() as tape:
                # مسح outputs السابقة
                outputs.clear()

                # Forward pass
                preds = model(img_preprocessed, training=False)
                pred_index = tf.argmax(preds[0])
                class_channel = preds[:, pred_index]

                # الحصول على conv_output المحفوظ
                if len(outputs) > 0:
                    conv_output = outputs[0]
                    tape.watch(conv_output)
                else:
                    # Fallback: استخدام طريقة أخرى
                    last_conv_layer.call = original_call
                    grad_model = keras.Model(model.input, [last_conv_layer.output, model.output])

                    with tf.GradientTape() as tape2:
                        conv_output, preds2 = grad_model(img_preprocessed)
                        pred_index = tf.argmax(preds2[0])
                        class_channel = preds2[:, pred_index]

                    grads = tape2.gradient(class_channel, conv_output)
                    last_conv_layer.call = original_call

                    if grads is None:
                        logger.warning("Gradients are None, using fallback")
                        return self._fallback_saliency_map(img_preprocessed), preds[0].numpy()

                    # حساب الأوزان
                    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

                    # إنشاء heatmap
                    conv_output = conv_output[0]
                    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
                    heatmap = tf.squeeze(heatmap)

                    # تطبيع
                    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)

                    return heatmap.numpy(), preds2[0].numpy()

            # إرجاع الدالة الأصلية
            last_conv_layer.call = original_call

            # حساب gradients
            grads = tape.gradient(class_channel, conv_output)

            if grads is None:
                logger.warning("Gradients are None after first attempt, trying alternative method")
                # طريقة بديلة: استخدام النموذج الكامل
                grad_model = keras.Model(model.input, [conv_output, model.output])

                with tf.GradientTape() as tape:
                    conv_output, preds = grad_model(img_preprocessed)
                    pred_index = tf.argmax(preds[0])
                    class_channel = preds[:, pred_index]

                grads = tape.gradient(class_channel, conv_output)

                if grads is None:
                    logger.warning("Still None, using fallback")
                    return self._fallback_saliency_map(img_preprocessed), preds[0].numpy()

            # حساب الأوزان
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            # إنشاء heatmap
            conv_output = conv_output[0]
            heatmap = conv_output @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)

            # تطبيع
            heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)

            logger.info("✅ Grad-CAM computed successfully")
            return heatmap.numpy(), preds[0].numpy()

        except Exception as e:
            logger.error(f"Error in Grad-CAM: {str(e)}")
            import traceback
            traceback.print_exc()
            # استخدم fallback
            try:
                img_preprocessed = tf.keras.applications.efficientnet_v2.preprocess_input(img_array.copy())
                preds = model.predict(img_preprocessed, verbose=0)
                return self._fallback_saliency_map(img_preprocessed), preds[0]
            except:
                return np.zeros((self.img_size[0], self.img_size[1])), None

    def _fallback_saliency_map(self, img_preprocessed):
        """
        طريقة بديلة: Saliency map بسيطة
        """
        try:
            with tf.GradientTape() as tape:
                img_tensor = tf.cast(img_preprocessed, tf.float32)
                tape.watch(img_tensor)
                predictions = self.model(img_tensor, training=False)
                pred_class_idx = tf.argmax(predictions[0])
                class_channel = predictions[:, pred_class_idx]

            grads = tape.gradient(class_channel, img_tensor)

            if grads is None:
                logger.warning("Gradients are None in fallback")
                return np.zeros((self.img_size[0], self.img_size[1]))

            # Enhanced processing
            grads_abs = tf.abs(grads)
            saliency = tf.reduce_mean(grads_abs, axis=-1)
            saliency = saliency[0]

            # تطبيع
            saliency_np = saliency.numpy()
            saliency_np = (saliency_np - saliency_np.min()) / (saliency_np.max() - saliency_np.min() + 1e-10)

            logger.info("✅ Fallback saliency map computed")
            return saliency_np

        except Exception as e:
            logger.error(f"Error in fallback saliency map: {str(e)}")
            return np.zeros((self.img_size[0], self.img_size[1]))

    def process_image(self, image_path):
        """
        معالجة كاملة لصورة واحدة - Improved version ✅
        """
        try:
            logger.info(f"📸 Processing: {os.path.basename(image_path)}")

            # تحميل الصورة
            img_array, img_preprocessed = self.load_and_preprocess_image(image_path)

            if img_array is None:
                return None

            # حساب Grad-CAM (الطريقة الجديدة)
            heatmap, preds = self.make_gradcam_heatmap(img_array, self.model)

            # إذا ما حصلنا على predictions من Grad-CAM، احصل عليها منفصلة
            if preds is None:
                preds = self.model.predict(img_preprocessed, verbose=0)[0]

            pred_class_idx = np.argmax(preds)
            confidence = preds[pred_class_idx]

            # Get top 3 predictions
            top3_indices = np.argsort(preds)[-3:][::-1]
            top3_predictions = [
                {
                    'class_name': self.class_names[idx],
                    'confidence': float(preds[idx]),
                    'index': int(idx)
                }
                for idx in top3_indices
            ]

            logger.info(f"  🎯 Prediction: {self.class_names[pred_class_idx]} ({confidence:.1%})")

            # تحميل الصورة الأصلية
            original_img = keras.preprocessing.image.load_img(image_path)
            original_array = keras.preprocessing.image.img_to_array(original_img) / 255.0

            # تكبير heatmap لحجم الصورة الأصلية
            heatmap_resized = cv2.resize(
                heatmap,
                (original_array.shape[1], original_array.shape[0])
            )

            # تلوين heatmap (نفس طريقة Colab - COLORMAP_JET)
            heatmap_colored = cv2.applyColorMap(
                np.uint8(255 * heatmap_resized),
                cv2.COLORMAP_JET
            )
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)  # تحويل من BGR إلى RGB
            heatmap_colored = heatmap_colored / 255.0

            # دمج الصور (نفس نسبة Colab: 50% original + 50% heatmap)
            overlay = original_array * 0.5 + heatmap_colored * 0.5

            return {
                'original': original_array,
                'heatmap': heatmap_resized,
                'heatmap_colored': heatmap_colored,
                'overlay': overlay,
                'predictions': preds,
                'pred_class_idx': int(pred_class_idx),
                'confidence': float(confidence),
                'class_name': self.class_names[pred_class_idx],
                'top3_predictions': top3_predictions,
                'image_name': os.path.basename(image_path)
            }

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def convert_to_base64(self, img_array):
        """
        تحويل صورة numpy إلى base64 للـ JSON
        """
        try:
            import base64
            from io import BytesIO
            from PIL import Image

            # تأكد أن القيم بين 0-255
            if img_array.max() <= 1.0:
                img_uint8 = np.uint8(img_array * 255)
            else:
                img_uint8 = np.uint8(img_array)

            # تحويل إلى صورة PIL
            pil_img = Image.fromarray(img_uint8)

            # حفظ في BytesIO
            buffer = BytesIO()
            pil_img.save(buffer, format='PNG', quality=95)  # PNG بدل JPEG للجودة الأعلى

            # تحويل إلى base64
            img_base64 = base64.b64encode(buffer.getvalue()).decode()

            return f"data:image/png;base64,{img_base64}"

        except Exception as e:
            logger.error(f"Error converting to base64: {str(e)}")
            return None

    def generate_gradcam_samples(self, image_paths, output_dir):
        """
        معالجة مجموعة صور وحفظ النتائج - Enhanced version ✅
        """
        try:
            logger.info(f"🔮 Starting Grad-CAM generation for {len(image_paths)} images...")
            results = []

            for idx, img_path in enumerate(image_paths):
                logger.info(f"Processing {idx + 1}/{len(image_paths)}: {os.path.basename(img_path)}")

                result = self.process_image(img_path)

                if result is None:
                    logger.warning(f"Failed to process {img_path}")
                    continue

                # تحويل الصور إلى base64
                result['original_base64'] = self.convert_to_base64(result['original'])
                result['heatmap_base64'] = self.convert_to_base64(result['heatmap_colored'])
                result['overlay_base64'] = self.convert_to_base64(result['overlay'])

                # حذف البيانات الكبيرة (arrays) لتوفير الذاكرة
                del result['original']
                del result['heatmap']
                del result['heatmap_colored']
                del result['overlay']

                # تحويل جميع التنبؤات
                result['all_predictions'] = {
                    self.class_names[i]: float(result['predictions'][i])
                    for i in range(len(self.class_names))
                }
                del result['predictions']

                results.append(result)
                logger.info(f"  ✅ Completed {idx + 1}/{len(image_paths)}")

            # حفظ البيانات
            os.makedirs(output_dir, exist_ok=True)

            output_data = {
                'generated_at': datetime.now().isoformat(),
                'num_samples': len(results),
                'class_names': self.class_names,
                'img_size': list(self.img_size),
                'samples': results
            }

            json_path = os.path.join(output_dir, 'gradcam_data.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            logger.info(f"🎉 Successfully saved {len(results)} samples to {json_path}")

            return output_data

        except Exception as e:
            logger.error(f"Error generating Grad-CAM samples: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
