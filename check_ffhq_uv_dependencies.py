print("🔍 FFHQ-UV Dependency Check\n")

# PyTorch
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"   ┗ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   ┗ CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"   ┗ CUDA version (compiled): {torch.version.cuda}")
except Exception as e:
    print(f"❌ PyTorch Error: {e}")

# TensorFlow
try:
    import tensorflow as tf
    print(f"✅ TensorFlow: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"   ┗ GPU Available: {bool(gpus)} ({[gpu.name for gpu in gpus]})")
    cuda_version = tf.sysconfig.get_build_info().get("cuda_version", "Unknown")
    cudnn_version = tf.sysconfig.get_build_info().get("cudnn_version", "Unknown")
    print(f"   ┗ Built with CUDA {cuda_version}, cuDNN {cudnn_version}")
except Exception as e:
    print(f"❌ TensorFlow Error: {e}")

# dlib
try:
    import dlib
    print(f"✅ dlib: {dlib.__version__}")
except Exception as e:
    print(f"❌ dlib Error: {e}")

# PyTorch3D
try:
    import pytorch3d
    print(f"✅ PyTorch3D: {pytorch3d.__version__}")
except Exception as e:
    print(f"❌ PyTorch3D Error: {e}")

# Nvdiffrast
try:
    import nvdiffrast
    print(f"✅ Nvdiffrast: {nvdiffrast.__version__}")
except Exception as e:
    print(f"❌ Nvdiffrast Error: {e}")

# OpenCV
try:
    import cv2
    print(f"✅ OpenCV: {cv2.__version__}")
except Exception as e:
    print(f"❌ OpenCV Error: {e}")

# TensorBoard
try:
    import tensorboard
    print(f"✅ TensorBoard: {tensorboard.__version__}")
except Exception as e:
    print(f"❌ TensorBoard Error: {e}")

# Requests (for MS Face API)
try:
    import requests
    print(f"✅ Requests: {requests.__version__}")
except Exception as e:
    print(f"❌ Requests Error: {e}")

# Skimage
try:
    import skimage
    print(f"✅ skimage: {skimage.__version__}")
except Exception as e:
    print(f"❌ skimage Error: {e}")

# PIL (Pillow)
try:
    from PIL import Image
    import PIL
    print(f"✅ Pillow: {PIL.__version__}")
except Exception as e:
    print(f"❌ Pillow Error: {e}")

print("\n🎯 Dependency Check Complete.")
