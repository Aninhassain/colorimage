from keras.models import Model
from keras.layers import Input, Conv2D, concatenate

# Define the input layer
input_layer = Input(shape=(224, 224, 1))

# Define the convolutional layers
conv1 = Conv2D(64, (3, 3), activation='relu')(input_layer)
conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)

# Define the output layer
output_layer = Conv2D(3, (3, 3), activation='tanh')(conv2)

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)