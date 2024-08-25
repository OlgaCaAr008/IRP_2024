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

# First model - binary
input_layer = layers.Input(shape=(geometry.shape[1], geometry.shape[2], 1))
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dense(256, activation='relu')(x)
output_x = layers.Dense(vx.shape[1] * vx.shape[2], activation='linear', name='velocity_x')(x)
output_y = layers.Dense(vy.shape[1] * vy.shape[2], activation='linear', name='velocity_y')(x)
output_rho = layers.Dense(rho.shape[1] * rho.shape[2], activation='linear', name='density')(x)
output_x1 = layers.Reshape((vx.shape[1], vx.shape[2], 1))(output_x)
output_y1 = layers.Reshape((vy.shape[1], vy.shape[2], 1))(output_y)
output_rho1 = layers.Reshape((rho.shape[1], rho.shape[2], 1))(output_rho)
model1 = models.Model(inputs=input_layer, outputs=[output_x1, output_y1, output_rho1])

Input_data1 = geometry.reshape((geometry.shape[0], geometry.shape[1], geometry.shape[2], 1))

def binary_to_sdf(binary_image):
    dist_out = distance_transform_edt(1 - binary_image)  # Background
    dist_in = distance_transform_edt(binary_image)       # Foreground
    sdf = dist_out - dist_in
    return sdf

sdf_geo = np.zeros_like(geometry, dtype=float)

for i in range(geometry.shape[0]):
  sdf_geo[i] = binary_to_sdf(geometry[i])

geometry = sdf_geo

# Second model - SDF
input_layer = layers.Input(shape=(geometry.shape[1], geometry.shape[2], 1))
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
output_x = layers.Conv2D(1,(1,1), padding='same', activation='linear', name='velocity_x2')(concat3)
output_y = layers.Conv2D(1,(1,1), padding='same', activation='linear', name='velocity_y2')(concat3)
output_rho = layers.Conv2D(1,(1,1), padding='same', activation='linear', name='density2')(concat3)

# Create the model
model2 = models.Model(inputs=input_layer, outputs=[output_x, output_y, output_rho])

Input_data2 = geometry.reshape((geometry.shape[0], geometry.shape[1], geometry.shape[2], 1))

# Fusion model
output1 = model1.output
output2 = model2.output

weight_model1 = 0.7  
weight_model2 = 0.3  

# Weighted average output fusion model
weighted_average_output_x = layers.Lambda(lambda x: x[0] * weight_model1 + x[1] * weight_model2)([output1[0], output2[0]])
weighted_average_output_y = layers.Lambda(lambda x: x[0] * weight_model1 + x[1] * weight_model2)([output1[1], output2[1]])
weighted_average_output_rho = layers.Lambda(lambda x: x[0] * weight_model1 + x[1] * weight_model2)([output1[2], output2[2]])

fusion_model = models.Model(inputs=[model1.input, model2.input], outputs=[weighted_average_output_x, weighted_average_output_y, weighted_average_output_rho])

# Compile and train or use the fusion model
fusion_model.compile(optimizer='adam', loss='mse')


Vx_data = vx.reshape((vx.shape[0], vx.shape[1], vx.shape[2],1))
Vy_data = vy.reshape((vy.shape[0], vy.shape[1], vy.shape[2],1))
Rho_data = rho.reshape((rho.shape[0], rho.shape[1], rho.shape[2],1))

np.random.seed(42)  # For reproducibility
indices = np.arange(Input_data1.shape[0])
np.random.shuffle(indices)

train_size = int(0.8 * len(indices))
train_indices = indices[:train_size]
test_indices = indices[train_size:]

Input_train1 = Input_data1[train_indices]
Input_test1 = Input_data1[test_indices]

Input_train2 = Input_data2[train_indices]
Input_test2 = Input_data2[test_indices]

Vx_train = Vx_data[train_indices]
Vx_test = Vx_data[test_indices]

Vy_train = Vy_data[train_indices]
Vy_test = Vy_data[test_indices]

Rho_train = Rho_data[train_indices]
Rho_test = Rho_data[test_indices]



# Fit the fusion model
history = fusion_model.fit(
    [Input_train1, Input_train2],  
    [Vx_train, Vy_train, Rho_train],  
    batch_size=32,
    epochs=10000,
    validation_split=0.2,  
    callbacks=[EarlyStopping(monitor='val_loss', patience=500, verbose=1, restore_best_weights=True)]
)


test_loss = fusion_model.evaluate(
    [Input_test1, Input_test2],  # Test inputs for both models
    [Vx_test, Vy_test, Rho_test],  # True values for each output
    batch_size=32 
)

predictions = fusion_model.predict([Input_test1, Input_test2])

# Save predictions
np.save('test_scores_w1.npy',test_loss)

np.save('predictions_w1.npy', predictions)
np.save('Vx_test_w1.npy', Vx_test)
np.save('Vy_test_w1.npy', Vy_test)
np.save('Rho_test_w1.npy', Rho_test)


