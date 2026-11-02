import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import streamlit as st
def cnn_project():

    from tensorflow.keras.datasets import mnist

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    print("Training Images:", X_train.shape)
    print("Testing Images:", X_test.shape)

    #Displaying the sample image

    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(X_train[i], cmap='gray')
        plt.title(y_train[i])
    plt.show()
    plt.close()
    print("Image Displayed")

    #Preprocessing
    #normalize:
    X_train = X_train/255.0
    X_test = X_test/255.0

    #CNN needs 4 dimensions:
    X_train = X_train.reshape(-1,28,28,1)
    X_test = X_test.reshape(-1,28,28,1)

    #converts labels
    from tensorflow.keras.utils import to_categorical
    y_train = to_categorical(y_train,10)
    y_test = to_categorical(y_test,10)

    #Build CNN
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D
    from tensorflow.keras.layers import MaxPooling2D
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Dense

    model = Sequential()
    model.add(
        Conv2D(
            32,
            (3,3),
            activation='relu',
            input_shape=(28,28,1)
        )
    )
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dense(10,activation='softmax'))

    #compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    #summary of the model
    model.summary()

    #Trianing the model
    history = model.fit(
        X_train,
        y_train,
        epochs=5,
        validation_split=0.2
    )

    #Accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title("Accuracy")
    plt.legend(['Train','Validation'])
    plt.show()

    #Loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("Loss")
    plt.legend(['Train','Validation'])
    plt.show()

    #Model evaluation
    loss, accuracy = model.evaluate(
        X_test,
        y_test
    )
    print("Accuracy:",accuracy)

    #To make predictions
    import numpy as np
    predictions = model.predict(X_test)

    #To make First predictions
    actual = np.argmax(y_test[0])
    predicted = np.argmax(predictions[0])
    print("Actual:",actual)
    print("Predicted:",predicted)

    #To show correct predictions
    for i in range(5):

        plt.imshow(
            X_test[i].reshape(28,28),
            cmap='gray'
        )

        plt.title(
            f"Actual:{np.argmax(y_test[i])} Predicted:{np.argmax(predictions[i])}"
        )

        plt.show()

    #To show incorrect predictions
    for i in range(100):

        if np.argmax(y_test[i]) != np.argmax(predictions[i]):

            plt.imshow(
                X_test[i].reshape(28,28),
                cmap='gray'
            )
                    
            plt.title(
                f"Actual:{np.argmax(y_test[i])} Predicted:{np.argmax(predictions[i])}"
            )

            plt.show()
            
            break

#Streamlit app
#Students marks calculator

#streamlit code
def streamlit_app():

    st.title("Student Marks Calculator")

    st.write("Enter marks of 5 subjects")

    s1 = st.number_input("Subject 1",0,100)
    s2 = st.number_input("Subject 2",0,100)
    s3 = st.number_input("Subject 3",0,100)
    s4 = st.number_input("Subject 4",0,100)
    s5 = st.number_input("Subject 5",0,100)

    if st.button("Calculate"):

        total = s1+s2+s3+s4+s5
        percentage = total/5

        st.write("Total Marks:", total)
        st.write("Percentage:", percentage)

        if percentage >= 90:
            st.success("Grade A")

        elif percentage >= 75:
            st.success("Grade B")

        elif percentage >= 60:
            st.warning("Grade C")

        else:
            st.error("Grade D")

#Main section
if __name__ == "__main__":
    streamlit_app()