from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input, Conv2D, Activation
from tensorflow.keras.optimizers import Adam
import numpy as np

#def create_model():
#    model=Sequential()
#            
#    model.add(Flatten(input_shape=(4,4)))
#    model.add(Dense(16))
#    model.add(Activation('relu'))
#    model.add(Dense(32))
#    model.add(Activation('relu'))
#    model.add(Dense(64))
#    model.add(Activation('relu'))
#    model.add(Dense(1))
#    model.add(Activation('linear'))
#    model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=0.005))
#    print(model.summary())
#    return model

def create_model():
    model = Sequential()
    
    # Convolutional Layers
    model.add(Conv2D(128, kernel_size=(2, 2), activation='relu', input_shape=(4, 4, 1)))
    model.add(Conv2D(128, kernel_size=(2, 2), activation='relu'))
    #model.add(Conv2D(256, kernel_size=(2, 2), activation='relu'))
    
    # Flatten layer
    model.add(Flatten())
    
    # Dense Layers
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    
    # Output Layer
    model.add(Dense(4, activation='softmax'))  # 4 possible moves: up, down, left, right
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Print model summary
    print(model.summary())
    
    return model

def check_accuracy(model, validation_data):
    print(f"Starting Accuracy Test.\n\nValidationdata len: {len(validation_data)}\n\nHold on. . .\n")
    
    states = np.array([np.array(data[0]) for data in validation_data])
    actions = np.array([data[1] for data in validation_data])
    
    # Evaluate the model on the validation data
    loss, accuracy = model.evaluate(states, actions)
    
    print(f"Validation Loss: {loss}")
    print(f"Validation Accuracy: {accuracy}")
    input("\nConfirm Data")

    return loss, accuracy

if __name__ == "__main__":
    model = create_model((4, 4))
    model.summary()
