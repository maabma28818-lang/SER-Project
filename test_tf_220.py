# test_tf_220.py - Fixed version
import numpy as np
import tensorflow as tf

print("🧪 Testing TensorFlow 2.20.0 Installation...")

try:
    print(f"✅ TensorFlow version: {tf.__version__}")
    
    # Test basic operations
    print("🔧 Testing basic operations...")
    a = tf.constant([1, 2, 3])
    b = tf.constant([4, 5, 6])
    result = a + b
    print(f"✅ Basic operations work: {result.numpy()}")
    
    # Test Keras model creation
    print("🧠 Testing Keras model creation...")
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, input_shape=(5,))
    ])
    print("✅ Keras model creation works!")
    
    # Test model compilation
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    print("✅ Model compilation works!")
    
    # Test prediction
    test_data = np.random.random((1, 5))
    prediction = model.predict(test_data, verbose=0)
    print(f"✅ Prediction works: {prediction.shape}")
    
    print("\n🎉 TensorFlow 2.20.0 is working perfectly!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()