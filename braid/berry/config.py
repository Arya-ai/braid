from easydict import EasyDict

BerryKeys = EasyDict()

# Contains some auxiliary variables such as 'p' in Dropout layer
BerryKeys.AUX_INPUTS = "aux_inputs"

# Contains all the layer activations/outputs for easy access
BerryKeys.LAYER_OUTPUTS = "layer_outputs"
