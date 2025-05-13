import tensorflow as tf

# Convert first innings model
model1 = tf.keras.models.load_model("odi_target_predictor_lstm.h5", custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
converter1 = tf.lite.TFLiteConverter.from_keras_model(model1)

# Use Select TF ops and disable lowering of tensor list operations
converter1.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter1._experimental_lower_tensor_list_ops = False

tflite_model1 = converter1.convert()
with open("odi_target_predictor_lstm.tflite", "wb") as f:
    f.write(tflite_model1)

# Convert second innings model
model2 = tf.keras.models.load_model("odi_chase_predictor_new.h5")
converter2 = tf.lite.TFLiteConverter.from_keras_model(model2)

# Same conversion settings
converter2.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter2._experimental_lower_tensor_list_ops = False

tflite_model2 = converter2.convert()
with open("odi_chase_predictor_new.tflite", "wb") as f:
    f.write(tflite_model2)