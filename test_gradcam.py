"""
🧪 Quick Grad-CAM Test Script
استخدم هذا السكريبت لاختبار Grad-CAM بشكل مباشر
"""

import os
import sys
import json

# أضف المسار
sys.path.insert(0, os.path.dirname(__file__))

def test_gradcam(session_id):
    """اختبار Grad-CAM لجلسة محددة"""
    print(f"\n{'='*70}")
    print(f"🧪 Testing Grad-CAM for session: {session_id}")
    print(f"{'='*70}\n")

    import tensorflow as tf
    from app.gradcam_generator import GradCAMGenerator

    # المسارات
    MODELS_FOLDER = 'trained_models'
    model_path = os.path.join(MODELS_FOLDER, f'efficientnetv2_{session_id}', 'model')
    metadata_path = os.path.join(MODELS_FOLDER, f'efficientnetv2_{session_id}', 'metadata.json')

    # 1. فحص وجود الملفات
    print("📁 Step 1: Checking files...")
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False
    print(f"✅ Model found: {model_path}")

    if not os.path.exists(metadata_path):
        print(f"❌ Metadata not found: {metadata_path}")
        return False
    print(f"✅ Metadata found: {metadata_path}")

    # 2. تحميل metadata
    print("\n📋 Step 2: Loading metadata...")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    class_names = metadata.get('class_names', [])
    extract_path = metadata.get('extract_path')

    print(f"   Classes: {class_names}")
    print(f"   Extract path: {extract_path}")

    if not class_names:
        print("❌ No class names in metadata")
        return False

    if not extract_path or not os.path.exists(extract_path):
        print(f"❌ Extract path not found: {extract_path}")
        return False

    # 3. جمع الصور
    print("\n📸 Step 3: Finding sample images...")
    sample_images = []
    for class_name in class_names:
        class_path = os.path.join(extract_path, class_name)
        if os.path.exists(class_path):
            images = [f for f in os.listdir(class_path)
                     if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if images:
                img_full_path = os.path.join(class_path, images[0])
                sample_images.append(img_full_path)
                print(f"   ✓ {class_name}: {images[0]}")

    if not sample_images:
        print("❌ No sample images found")
        return False

    print(f"\n✅ Found {len(sample_images)} images")

    # 4. تحميل النموذج
    print("\n🤖 Step 4: Loading model...")
    print("   ⏳ This may take 30-60 seconds for large models...")

    import time
    start_time = time.time()

    try:
        model = tf.keras.models.load_model(model_path)
        load_time = time.time() - start_time

        print(f"✅ Model loaded successfully in {load_time:.1f}s")
        print(f"   Type: {type(model).__name__}")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print(f"   Total layers: {len(model.layers)}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 5. حساب Grad-CAM
    print("\n🔮 Step 5: Computing Grad-CAM...")
    print("   ⏳ Processing images (may take 1-2 min)...")

    try:
        start_time = time.time()

        print("   📝 Creating Grad-CAM generator...")
        gradcam_gen = GradCAMGenerator(model, class_names, img_size=(256, 256))

        output_dir = os.path.join(MODELS_FOLDER, f'gradcam_test_{session_id}')
        os.makedirs(output_dir, exist_ok=True)

        print(f"   📁 Output dir: {output_dir}")
        print(f"   🖼️  Processing {len(sample_images)} images...")

        # معالجة صورة واحدة فقط للاختبار السريع
        test_images = sample_images[:1]  # فقط أول صورة
        print(f"   💡 Quick test: processing only 1 image")

        gradcam_data = gradcam_gen.generate_gradcam_samples(test_images, output_dir)

        process_time = time.time() - start_time

        if gradcam_data and gradcam_data.get('num_samples', 0) > 0:
            print(f"\n✅ SUCCESS! (completed in {process_time:.1f}s)")
            print(f"   Samples: {gradcam_data['num_samples']}")
            print(f"   Output: {os.path.join(output_dir, 'gradcam_data.json')}")
            print(f"\n   📊 Sample data:")
            if gradcam_data.get('samples'):
                sample = gradcam_data['samples'][0]
                print(f"      - Image: {sample.get('image_name')}")
                print(f"      - Predicted: {sample.get('class_name')}")
                print(f"      - Confidence: {sample.get('confidence', 0):.1%}")
            return True
        else:
            print(f"\n❌ FAILED - No samples generated (took {process_time:.1f}s)")
            print(f"   Data: {gradcam_data}")
            return False

    except Exception as e:
        print(f"\n❌ Error during Grad-CAM: {e}")
        import traceback
        traceback.print_exc()
        return False


def list_sessions():
    """عرض جميع الجلسات المتاحة"""
    print("\n📋 Available sessions:")
    MODELS_FOLDER = 'trained_models'

    if not os.path.exists(MODELS_FOLDER):
        print("❌ No trained_models folder")
        return []

    sessions = []
    for folder in os.listdir(MODELS_FOLDER):
        if folder.startswith('efficientnetv2_'):
            session_id = folder.replace('efficientnetv2_', '')
            sessions.append(session_id)
            print(f"   • {session_id}")

    return sessions


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🔬 Grad-CAM Test Utility")
    print("="*70)

    # عرض الجلسات المتاحة
    sessions = list_sessions()

    if not sessions:
        print("\n❌ No sessions found. Train a model first!")
        sys.exit(1)

    # إذا في argument، استخدمه
    if len(sys.argv) > 1:
        session_id = sys.argv[1]
    else:
        # استخدم آخر جلسة
        session_id = sorted(sessions)[-1]
        print(f"\n💡 Using latest session: {session_id}")
        print("   (You can specify a session: python test_gradcam.py SESSION_ID)")

    # اختبار
    success = test_gradcam(session_id)

    print("\n" + "="*70)
    if success:
        print("✅ TEST PASSED!")
    else:
        print("❌ TEST FAILED!")
    print("="*70 + "\n")

    sys.exit(0 if success else 1)
