from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import mnist_reader


def draw(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], "r--")
    plt.plot(history.history['val_' + 'accuracy'], "g--")
    plt.ylabel('accuracy')
    plt.xlabel('Epoch')
    plt.ylim((0.8, 1.00))
    plt.legend(['train', 'test'], loc='best')


    plt.show()



# PREPARE

X_train, y_train = mnist_reader.load_mnist('fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('fashion', kind='t10k')

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# ARCHITECTURE

model = Sequential()

model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# COMPILE

model.compile(optimizer='adamax',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# TRAIN

EarlyStop = EarlyStopping(monitor='val_loss',
                          patience=4,
                          verbose=1)

ModelCheck = ModelCheckpoint(filepath='best_model.h5',
                             monitor='val_loss',
                             save_best_only=True)

memory = model.fit(X_train,
            y_train,
            epochs=40,
            verbose=1,
            batch_size=64,
            validation_split=0.3,
            callbacks=[EarlyStop, ModelCheck]
            )

# RUN

#model.evaluate(X_test, y_test, batch_size=32)

modelek = load_model('best_model.h5')
modelek.evaluate(X_test, y_test, batch_size=32)
draw(memory)
