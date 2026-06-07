class_name TinyNeuron
extends RefCounted

var weights := PackedFloat32Array([0.0, 0.0])
var bias := 0.0
var activation := "tanh"

func forward(inputs: PackedFloat32Array) -> float:
	assert(inputs.size() == weights.size())
	var weighted_sum := bias
	for index in range(inputs.size()):
		weighted_sum += inputs[index] * weights[index]
	match activation:
		"sigmoid":
			return 1.0 / (1.0 + exp(-weighted_sum))
		"tanh":
			return tanh(weighted_sum)
		_:
			assert(false, "Unknown activation: %s" % activation)
			return 0.0
