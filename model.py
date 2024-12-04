from tensorflow.keras.applications import VGG16

# Load a pre-trained VGG16 model with ImageNet weights
model = VGG16(weights='imagenet')

# Save the model in both Keras and H5 formats
model.save('vgg16_model.keras', save_format='keras')  # Native Keras format (recommended)
model.save('vgg16_model.h5')  # H5 format for compatibility

print("Models saved as 'vgg16_model.keras' and 'vgg16_model.h5'")
