import keras.layers as l
from keras.models import Model
from keras.optimizers import Adam

from layers import *


def create_dense_model(input_shape, output_shape, neurons, dropout_rate=0.1, lr=1e-4):
    input = l.Input(shape=(input_shape,))
    next = input
    next = l.BatchNormalization()(next)
    next = l.Dense(neurons, activation='elu')(next)
    next = l.BatchNormalization()(next)
    next = l.Dropout(dropout_rate)(next)
    next = l.Dense(neurons / 2, activation='elu')(next)
    next = l.BatchNormalization()(next)
    next = l.Dropout(dropout_rate)(next)
    next = l.Dense(neurons / 4, activation='elu')(next)
    next = l.BatchNormalization()(next)
    next = l.Dropout(dropout_rate)(next)
    next = l.Dense(output_shape, activation='softmax')(next)

    model = Model(input=input, output=next)

    model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    return model, {}


def create_disc_model(input_shape, output_shape, neurons, dropout_rate=0.1, lr=1e-4):
    input = l.Input(shape=(input_shape,))
    next = input
    next = DiscretizationLayer(input_shape)(next)
    next = l.BatchNormalization()(next)
    next = l.Dense(neurons, activation='elu')(next)
    next = l.BatchNormalization()(next)
    next = l.Dropout(dropout_rate)(next)
    next = l.Dense(neurons / 2, activation='elu')(next)
    next = l.BatchNormalization()(next)
    next = l.Dropout(dropout_rate)(next)
    next = l.Dense(neurons / 4, activation='elu')(next)
    next = l.BatchNormalization()(next)
    next = l.Dropout(dropout_rate)(next)
    next = l.Dense(output_shape, activation='softmax')(next)

    model = Model(input=input, output=next)

    model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    return model, {}


def create_disc_model_lite(input_shape, output_shape, neurons, dropout_rate=0.1, lr=1e-4):
    input = l.Input(shape=(input_shape,))
    next = input
    next = DiscretizationLayerLite(input_shape)(next)
    next = l.BatchNormalization()(next)
    next = l.Dense(neurons, activation='elu')(next)
    next = l.BatchNormalization()(next)
    next = l.Dropout(dropout_rate)(next)
    next = l.Dense(neurons / 2, activation='elu')(next)
    next = l.BatchNormalization()(next)
    next = l.Dropout(dropout_rate)(next)
    next = l.Dense(neurons / 4, activation='elu')(next)
    next = l.BatchNormalization()(next)
    next = l.Dropout(dropout_rate)(next)
    next = l.Dense(output_shape, activation='softmax')(next)

    model = Model(input=input, output=next)

    model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    return model, {}


def create_laplace_model(input_shape, output_shape, neurons, dropout_rate=0.1, lr=1e-4):
    input = l.Input(shape=(input_shape,))
    next = input
    next = DiscretizationLayerLaplace(input_shape)(next)
    next = l.BatchNormalization()(next)
    next = l.Dense(neurons, activation='elu')(next)
    next = l.BatchNormalization()(next)
    next = l.Dropout(dropout_rate)(next)
    next = l.Dense(neurons / 2, activation='elu')(next)
    next = l.BatchNormalization()(next)
    next = l.Dropout(dropout_rate)(next)
    next = l.Dense(neurons / 4, activation='elu')(next)
    next = l.BatchNormalization()(next)
    next = l.Dropout(dropout_rate)(next)
    next = l.Dense(output_shape, activation='softmax')(next)

    model = Model(input=input, output=next)

    model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    return model, {}


def create_laplace_model_lite(input_shape, output_shape, neurons, dropout_rate=0.1, lr=1e-4):
    input = l.Input(shape=(input_shape,))
    next = input
    next = DiscretizationLayerLaplaceLite(input_shape)(next)
    next = l.BatchNormalization()(next)
    next = l.Dense(neurons, activation='elu')(next)
    next = l.BatchNormalization()(next)
    next = l.Dropout(dropout_rate)(next)
    next = l.Dense(neurons / 2, activation='elu')(next)
    next = l.BatchNormalization()(next)
    next = l.Dropout(dropout_rate)(next)
    next = l.Dense(neurons / 4, activation='elu')(next)
    next = l.BatchNormalization()(next)
    next = l.Dropout(dropout_rate)(next)
    next = l.Dense(output_shape, activation='softmax')(next)

    model = Model(input=input, output=next)

    model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    return model, {}


models = {
    'dense': create_dense_model,
    'disc': create_disc_model,
    'disc_lite': create_disc_model_lite,
    'laplace': create_laplace_model,
    'laplace_lite': create_laplace_model_lite
}