from modelutil import load_model


# This uses your current load_model(), which knows how to read the checkpoint
model = load_model()
# Export to an H5 weights file that Keras 3 / TF 2.20+ can load easily
model.save_weights("models-checkpoint/lipnet.weights.h5")
print("Saved H5 weights to models-checkpoint/lipnet.weights.h5")