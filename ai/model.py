from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input, Conv2D
from tensorflow.keras.optimizers import Adam
import numpy as np

def create_model():
    model=Sequential()
            
    model.add(Flatten(input_shape=(4,4)))
    model.add(Dense(units=1024,activation="relu"))
    model.add(Dense(units=512,activation="relu"))
    model.add(Dense(units=256,activation="relu"))
    model.add(Dense(units=4))
    model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=0.005))
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
