import numpy as np
from scipy.ndimage import distance_transform_edt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, models, regularizers

field = np.load('field.npy') # n x Ny x Nx x 3
geometry = np.load('geo.npy') # n x Ny x Nx

vx = field[:,:,:,1] # n x Ny x Nx
vy = field[:,:,:,2] # n x Ny x Nx
rho = field[:,:,:,0]


def binary_to_sdf(binary_image):
    dist_out = distance_transform_edt(1 - binary_image)  # Background
    dist_in = distance_transform_edt(binary_image)       # Foreground
    sdf = dist_out - dist_in
    return sdf

sdf_geo = np.zeros_like(geometry, dtype=float)

for i in range(geometry.shape[0]):
  sdf_geo[i] = binary_to_sdf(geometry[i])

geometry = sdf_geo


# Neural Network

input_layer = layers.Input(shape=(geometry.shape[1], geometry.shape[2], 1))

# Downsampling Path
conv1 = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(input_layer)
conv1 = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(conv1)
pool1 = layers.MaxPooling2D((2, 2))(conv1)

conv2 = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(pool1)
conv2 = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(conv2)
pool2 = layers.MaxPooling2D((2, 2))(conv2)

conv3 = layers.Conv2D(128, (5, 5), activation='relu', padding='same')(pool2)
conv3 = layers.Conv2D(128, (5, 5), activation='relu', padding='same')(conv3)
pool3 = layers.MaxPooling2D((3, 3))(conv3)

conv4 = layers.Conv2D(256, (5, 5), activation='relu', padding='same')(pool3)
conv4 = layers.Conv2D(256, (5, 5), activation='relu', padding='same')(conv4)

# Upsampling Path
up1 = layers.Conv2DTranspose(128, (5, 5), strides=(3, 3), padding='same', activation='relu')(conv4)
concat1 = layers.Concatenate()([up1, conv3])

up2 = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', activation='relu')(concat1)
concat2 = layers.Concatenate()([up2, conv2])

up3 = layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', activation='relu')(concat2)
concat3 = layers.Concatenate()([up3, conv1])

# Output layers for the x and y components of the velocity
output_x = layers.Conv2D(1,(1,1), padding='same', activation='linear', name='velocity_x')(concat3)
output_y = layers.Conv2D(1,(1,1), padding='same', activation='linear', name='velocity_y')(concat3)
output_rho = layers.Conv2D(1,(1,1), padding='same', activation='linear', name='density')(concat3)


# Create the model
model = models.Model(inputs=input_layer, outputs=[output_x, output_y, output_rho])

model.summary()

model.compile(
    optimizer='adam',
    loss='mean_squared_error',  # Switched from custom_loss_function to mean_squared_error
    metrics={'velocity_x': ['mae'], 'velocity_y': ['mae'], 'density' : ['mae']}
)


Input_data = geometry.reshape((geometry.shape[0], geometry.shape[1], geometry.shape[2], 1))
Vx_data = vx.reshape((vx.shape[0], vx.shape[1], vx.shape[2],1))
Vy_data = vy.reshape((vy.shape[0], vy.shape[1], vy.shape[2],1))
Rho_data = rho.reshape((rho.shape[0], rho.shape[1], rho.shape[2],1))

np.random.seed(42)  # For reproducibility
indices = np.arange(Input_data.shape[0])
np.random.shuffle(indices)

train_size = int(0.8 * len(indices))
train_indices = indices[:train_size]
test_indices = indices[train_size:]

Input_train = Input_data[train_indices]
Input_test = Input_data[test_indices]

Vx_train = Vx_data[train_indices]
Vx_test = Vx_data[test_indices]

Vy_train = Vy_data[train_indices]
Vy_test = Vy_data[test_indices]

Rho_train = Rho_data[train_indices]
Rho_test = Rho_data[test_indices]


early_stopping = EarlyStopping(monitor='val_loss', patience=500, verbose=1, restore_best_weights=True)


history = model.fit(Input_train, {'velocity_x': Vx_train, 'velocity_y': Vy_train, 'density':Rho_train}, epochs=10000, batch_size=32, validation_split=0.2,callbacks=[early_stopping])

test_scores = model.evaluate(Input_test, [Vx_test, Vy_test, Rho_test])

# Save predictions
np.save('test_scores2.npy',test_scores)
predictions = model.predict(Input_test)
np.save('predictions2.npy', predictions)
np.save('Vx_test2.npy', Vx_test)
np.save('Vy_test2.npy', Vy_test)
np.save('Rho_test2.npy', Rho_test)


# Prepare data for saving
epochs = np.arange(1, len(history.history['loss']) + 1)
loss = np.array(history.history['loss'])
val_loss = np.array(history.history['val_loss'])
velocity_x_mae = np.array(history.history['velocity_x_mae'])
val_velocity_x_mae = np.array(history.history['val_velocity_x_mae'])
velocity_y_mae = np.array(history.history['velocity_y_mae'])
val_velocity_y_mae = np.array(history.history['val_velocity_y_mae'])
density_mae = np.array(history.history['density_mae'])
val_density_mae = np.array(history.history['val_density_mae'])

# Save metrics to .npz file
np.savez('training_history_data2.npz',
         epochs=epochs,
         loss=loss,
         val_loss=val_loss,
         velocity_x_mae=velocity_x_mae,
         val_velocity_x_mae=val_velocity_x_mae,
         velocity_y_mae=velocity_y_mae,
         val_velocity_y_mae=val_velocity_y_mae,
         density_mae=density_mae,
         val_density_mae=val_density_mae
        )

from tensorflow.keras.models import Model
import numpy as np

# Create a list to hold models for each convolutional layer's output
intermediate_layer_models = []

# Extract the outputs of each convolutional layer
layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name]

# Create a model for each layer output
for output in layer_outputs:
    intermediate_model = Model(inputs=model.input, outputs=output)
    intermediate_layer_models.append(intermediate_model)

# Choose a sample input image from the test set
input_sample = Input_test[1:2]  # Ensure it's a batch of one

# Dictionary to hold output feature maps for each layer
output_feature_maps = {}

for i, intermediate_model in enumerate(intermediate_layer_models):
    # Get the output of the intermediate model
    outputs = intermediate_model.predict(input_sample)
    # Save outputs to dictionary
    output_feature_maps[f"layer_{i}"] = outputs

# Save the feature maps to a file
np.savez('feature_maps_all.npz', **output_feature_maps)
