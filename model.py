from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import load_model

# Load a pre-trained VGG16 model from Keras with ImageNet weights
model = VGG16(weights='imagenet')

# Optionally, save it as an .h5 file for later use
model.save('vgg16_model.h5')

print("Pre-trained VGG16 model saved as 'vgg16_model.h5'")
