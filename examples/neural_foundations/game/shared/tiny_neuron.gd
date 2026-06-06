class_name TinyNeuron
extends RefCounted

var weights := PackedFloat32Array([0.0, 0.0])
var bias := 0.0

func forward(inputs: PackedFloat32Array) -> float:
	assert(inputs.size() == weights.size())
	var weighted_sum := bias
	for index in range(inputs.size()):
		weighted_sum += inputs[index] * weights[index]
	return tanh(weighted_sum)
