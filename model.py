import tensorflow as tf

# Load the smaller pre-trained model
model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),  # Same input shape as before
    include_top=True,          # Include the classification head
    weights="imagenet"         # Use pre-trained weights
)

# Save the model to a directory
model.save("mobilenet_model")
