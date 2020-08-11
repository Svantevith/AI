import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from ann_training_set import scaled_train_samples, train_labels
from ann_testing_set import scaled_test_samples, test_labels
from confusion_matrix_plot import plot_confusion_matrix

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# print('Num GPUs Available: ', len(physical_devices))
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

ann_model = Sequential([
    # input_shape=(1,), because it is connected only to 1 hidden layer
    # Dense means that it is fully connected
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
])

ann_model.summary()

ann_model.compile(
    Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# supervised learning based on given labels
ann_model.fit(x=scaled_train_samples,
              y=train_labels,
              validation_split=0.1,
              batch_size=10,
              epochs=30,
              shuffle=True,
              verbose=2
              )

# unsupervised learning based on predictions without given labels
# verbosity is 0 because there is no output there
ann_predictions = ann_model.predict(
    x=scaled_test_samples,
    batch_size=10,
    verbose=0
)

for p in ann_predictions:
    print(p)

rounded_predictions = np.argmax(ann_predictions, axis=-1)

for rp in rounded_predictions:
    print(rp)


confusionMatrix = confusion_matrix(y_true=test_labels, y_pred=rounded_predictions)
cm_plot_labels = ['No_side_effects', 'Side_effects_occurred']

plot_confusion_matrix(cm=confusionMatrix, classes=cm_plot_labels, title='Confusion Matrix')
