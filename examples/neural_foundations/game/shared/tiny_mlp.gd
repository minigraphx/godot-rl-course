class_name TinyMLP
extends RefCounted

var w_hidden: Array[PackedFloat32Array] = [
	PackedFloat32Array([0.1, -0.2, 0.3]),
	PackedFloat32Array([0.4, 0.1, -0.1]),
	PackedFloat32Array([-0.3, 0.2, 0.2]),
	PackedFloat32Array([0.2, 0.3, -0.4]),
]
var b_hidden := PackedFloat32Array([0.0, 0.0, 0.0, 0.0])
var w_output: Array[PackedFloat32Array] = [
	PackedFloat32Array([0.2, -0.1, 0.3, 0.1]),
	PackedFloat32Array([-0.2, 0.4, 0.1, -0.3]),
]
var b_output := PackedFloat32Array([0.0, 0.0])


func forward(inputs: PackedFloat32Array) -> PackedFloat32Array:
	var hidden := _hidden_activations(inputs)
	var outputs := PackedFloat32Array()
	for output_index in range(w_output.size()):
		var weighted_sum := b_output[output_index]
		for hidden_index in range(hidden.size()):
			weighted_sum += w_output[output_index][hidden_index] * hidden[hidden_index]
		outputs.append(tanh(weighted_sum))
	return outputs


func mse_loss(prediction: PackedFloat32Array, target: PackedFloat32Array) -> float:
	assert(prediction.size() == target.size())
	var total := 0.0
	for index in range(prediction.size()):
		var error := prediction[index] - target[index]
		total += error * error
	return total / prediction.size()


func train_sample(
	inputs: PackedFloat32Array,
	target: PackedFloat32Array,
	learning_rate: float
) -> float:
	var hidden_pre := _hidden_pre_activations(inputs)
	var hidden := PackedFloat32Array()
	for value in hidden_pre:
		hidden.append(tanh(value))

	var output_pre := PackedFloat32Array()
	var prediction := PackedFloat32Array()
	for output_index in range(w_output.size()):
		var weighted_sum := b_output[output_index]
		for hidden_index in range(hidden.size()):
			weighted_sum += w_output[output_index][hidden_index] * hidden[hidden_index]
		output_pre.append(weighted_sum)
		prediction.append(tanh(weighted_sum))

	var loss := mse_loss(prediction, target)
	var d_output_pre := PackedFloat32Array()
	for output_index in range(prediction.size()):
		var d_prediction := 2.0 * (prediction[output_index] - target[output_index]) / target.size()
		d_output_pre.append(d_prediction * (1.0 - prediction[output_index] * prediction[output_index]))

	var d_hidden := PackedFloat32Array()
	for hidden_index in range(hidden.size()):
		var gradient := 0.0
		for output_index in range(w_output.size()):
			gradient += w_output[output_index][hidden_index] * d_output_pre[output_index]
		d_hidden.append(gradient)

	for output_index in range(w_output.size()):
		for hidden_index in range(hidden.size()):
			w_output[output_index][hidden_index] -= learning_rate * d_output_pre[output_index] * hidden[hidden_index]
		b_output[output_index] -= learning_rate * d_output_pre[output_index]

	for hidden_index in range(w_hidden.size()):
		var d_hidden_pre := d_hidden[hidden_index] * (1.0 - hidden[hidden_index] * hidden[hidden_index])
		for input_index in range(inputs.size()):
			w_hidden[hidden_index][input_index] -= learning_rate * d_hidden_pre * inputs[input_index]
		b_hidden[hidden_index] -= learning_rate * d_hidden_pre

	return loss


func _hidden_activations(inputs: PackedFloat32Array) -> PackedFloat32Array:
	var hidden_pre := _hidden_pre_activations(inputs)
	var hidden := PackedFloat32Array()
	for value in hidden_pre:
		hidden.append(tanh(value))
	return hidden


func _hidden_pre_activations(inputs: PackedFloat32Array) -> PackedFloat32Array:
	var values := PackedFloat32Array()
	for hidden_index in range(w_hidden.size()):
		assert(inputs.size() == w_hidden[hidden_index].size())
		var weighted_sum := b_hidden[hidden_index]
		for input_index in range(inputs.size()):
			weighted_sum += w_hidden[hidden_index][input_index] * inputs[input_index]
		values.append(weighted_sum)
	return values
