from keras import backend as k
from keras.models import Sequential
from keras.layers import Dense, Deconv2D, Dropout

import tensorflow as tf
from tensorflow.python.saved_model import builder as save_model_builder, tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def


def save_TF_model(model, export_path):
    """Convert the Keras HDF5 model into TensorFlow SavedModel."""

    builder = save_model_builder.SavedModelBuilder(export_path)
    signature = predict_signature_def(
        inputs={'input': model.inputs[0]},
        outputs={'output': model.output[0]}
    )

    with k.get_session() as session:
        builder.add_meta_graph_and_variables(
            sess=session,
            tags=[tag_constants.SERVING],
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
            }
        )

        builder.save()

def model():
    model = Sequential()
    return model