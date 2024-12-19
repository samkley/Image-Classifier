import tensorflow as tf

# Load the MobileNetV2 model
model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=True,
    weights="imagenet"
)


model.save("mobilenet_model.keras") 
