"""
Phase 6 — Face-recognition CNN architecture (trained from scratch)
==================================================================

Defines the convolutional neural network that classifies an aligned face crop
into one of the N enrolled students. The network is built layer-by-layer here
and its weights are trained (Phase 7) exclusively on project data — no
pre-trained weights, no transfer learning. Using Keras to *assemble and train*
an architecture we designed is "from scratch" in the sense required: the
learned parameters come only from our data. (A deep-learning framework is a
numerical library, exactly as NumPy is for the HOG detector.)

Architecture (VGG-style, compact — chosen for a small dataset)
--------------------------------------------------------------
    input  112 x 112 x C
    block1 [Conv3x3(32)  -> BN -> ReLU] x2 -> MaxPool     ->  56 x 56
    block2 [Conv3x3(64)  -> BN -> ReLU] x2 -> MaxPool     ->  28 x 28
    block3 [Conv3x3(128) -> BN -> ReLU] x2 -> MaxPool     ->  14 x 14
    block4 [Conv3x3(256) -> BN -> ReLU] x2 -> MaxPool     ->   7 x  7
    GlobalAveragePooling  ->  Dropout
    Dense(256) -> BN -> ReLU -> Dropout
    Dense(N, softmax)

Design rationale (dissertation §4)
----------------------------------
* Small 3x3 filters stacked in blocks (VGG idiom): two 3x3 convs share the
  receptive field of one 5x5 at fewer parameters and with an extra
  nonlinearity — good capacity-per-parameter on a modest dataset.
* Batch normalization after every conv stabilizes and speeds training of a
  from-scratch network and adds mild regularization.
* **Global average pooling instead of Flatten+Dense**: collapses the 7x7x256
  map to a 256-vector, removing the huge fully-connected layer that would
  otherwise dominate the parameter count and overfit a few-hundred-image
  dataset. This single choice keeps the model near ~1.5M parameters.
* Dropout before and within the classifier head — the primary explicit
  regularizer, tuned in Phase 7 alongside augmentation strength.
* Softmax over N students: closed-set classification (fixed cohort). Unknown
  rejection is handled downstream by a confidence threshold, not by the
  architecture.

Usage:
    python src/model.py                 # build, summarize, verify a forward pass
    from model import build_model, compile_model
"""

from __future__ import annotations

import keras
from keras import layers

import config


def _conv_block(x, filters: int, block: int):
    """Two (Conv3x3 -> BN -> ReLU) layers followed by 2x2 max-pooling."""
    for i in (1, 2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                          kernel_initializer="he_normal",
                          name=f"block{block}_conv{i}")(x)
        x = layers.BatchNormalization(name=f"block{block}_bn{i}")(x)
        x = layers.Activation("relu", name=f"block{block}_relu{i}")(x)
    return layers.MaxPooling2D(2, name=f"block{block}_pool")(x)


def build_model(num_classes: int,
                input_size: int = config.INPUT_SIZE,
                channels: int = config.INPUT_CHANNELS,
                base_filters: int = 32,
                head_units: int = 256,
                dropout: float = 0.5,
                name: str = "face_cnn") -> keras.Model:
    """Construct (untrained) the recognition CNN.

    Parameters
    ----------
    num_classes : number of enrolled students (softmax outputs).
    input_size, channels : crop geometry; must match the processed dataset.
    base_filters : channel width of block 1; doubles each block.
    head_units : width of the dense layer before the classifier.
    dropout : rate before the head; the in-head rate is 0.6x this.
    """
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")
    keras.utils.set_random_seed(config.RANDOM_SEED)

    inputs = keras.Input(shape=(input_size, input_size, channels), name="crop")
    x = inputs
    for block, mult in enumerate((1, 2, 4, 8), start=1):
        x = _conv_block(x, base_filters * mult, block)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="head_drop1")(x)
    x = layers.Dense(head_units, use_bias=False,
                     kernel_initializer="he_normal", name="head_dense")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Activation("relu", name="head_relu")(x)
    x = layers.Dropout(dropout * 0.6, name="head_drop2")(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           name="predictions")(x)
    return keras.Model(inputs, outputs, name=name)


def compile_model(model: keras.Model,
                  learning_rate: float = 1e-3,
                  label_smoothing: float = 0.0) -> keras.Model:
    """Attach optimizer, loss and metrics. Uses integer (sparse) labels.

    label_smoothing > 0 requires one-hot labels; kept 0 here so the training
    pipeline can feed integer class indices directly.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc"),
                 keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3")],
    )
    return model


def _self_test() -> None:
    """Build the model, print a summary, and verify one forward pass."""
    import numpy as np

    num_classes = 56  # current DSLR cohort; real N comes from labels at train
    model = compile_model(build_model(num_classes))
    model.summary()

    batch = np.random.rand(4, config.INPUT_SIZE, config.INPUT_SIZE,
                           config.INPUT_CHANNELS).astype("float32")
    probs = model.predict(batch, verbose=0)
    row_sums = probs.sum(axis=1)

    print("\n============ MODEL SELF-TEST ============")
    print(f"input shape          : {batch.shape}")
    print(f"output shape         : {probs.shape}  (expected (4, {num_classes}))")
    print(f"softmax rows sum ~1  : {np.allclose(row_sums, 1.0, atol=1e-5)}")
    print(f"total parameters     : {model.count_params():,}")
    trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    print(f"trainable parameters : {trainable:,}")
    assert probs.shape == (4, num_classes)
    assert np.allclose(row_sums, 1.0, atol=1e-5)
    print("status               : OK")
    print("=========================================")


if __name__ == "__main__":
    _self_test()
