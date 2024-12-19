import tensorflow as tf

# Load the MobileNetV2 model
model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=True,
    weights="imagenet"
)

# Save the model in HDF5 format
model.save("mobilenet_model.h5")  # Specify the `.h5` extension
