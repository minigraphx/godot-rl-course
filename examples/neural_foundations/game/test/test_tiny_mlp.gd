extends SceneTree

const Harness = preload("res://test/harness.gd")


func _init() -> void:
	call_deferred("_run")


func _run() -> void:
	var harness := Harness.new()
	var tiny_mlp_script := load("res://shared/tiny_mlp.gd")
	if tiny_mlp_script == null:
		harness.assert_true(false, "TinyMLP script exists")
		harness.finish(self)
		return

	var network: RefCounted = tiny_mlp_script.new()
	var hidden_weights: Array[PackedFloat32Array] = [
		PackedFloat32Array([0.2, -0.4]),
		PackedFloat32Array([0.7, 0.1]),
	]
	network.w_hidden = hidden_weights
	network.b_hidden = PackedFloat32Array([0.0, 0.1])
	var output_weights: Array[PackedFloat32Array] = [
		PackedFloat32Array([0.6, -0.3]),
	]
	network.w_output = output_weights
	network.b_output = PackedFloat32Array([0.05])

	var inputs := PackedFloat32Array([0.5, -0.25])
	var hidden_0 := tanh(0.2 * 0.5 + -0.4 * -0.25)
	var hidden_1 := tanh(0.7 * 0.5 + 0.1 * -0.25 + 0.1)
	var expected := tanh(0.6 * hidden_0 + -0.3 * hidden_1 + 0.05)
	var prediction: PackedFloat32Array = network.forward(inputs)

	harness.assert_close(prediction[0], expected, 0.000001, "forward pass")

	var target := PackedFloat32Array([0.8])
	var gradients: Dictionary = network.gradient_sample(inputs, target)
	var gradient_hidden: Array = gradients["w_hidden"]
	var gradient_output: Array = gradients["w_output"]
	var gradient_b_hidden: PackedFloat32Array = gradients["b_hidden"]
	var gradient_b_output: PackedFloat32Array = gradients["b_output"]
	harness.assert_close(
		gradient_hidden[0][0],
		-0.4325939877,
		0.000001,
		"hidden gradient row 0 col 0"
	)
	harness.assert_close(
		gradient_hidden[0][1],
		0.2162969938,
		0.000001,
		"hidden gradient row 0 col 1"
	)
	harness.assert_close(
		gradient_hidden[1][0],
		0.1888499534,
		0.000001,
		"hidden gradient row 1 col 0"
	)
	harness.assert_close(
		gradient_hidden[1][1],
		-0.0944249767,
		0.000001,
		"hidden gradient row 1 col 1"
	)
	harness.assert_close(
		gradient_b_hidden[0],
		-0.8651879753,
		0.000001,
		"hidden bias gradient 0"
	)
	harness.assert_close(
		gradient_b_hidden[1],
		0.3776999069,
		0.000001,
		"hidden bias gradient 1"
	)
	harness.assert_close(
		gradient_output[0][0],
		-0.2961483109,
		0.000001,
		"output gradient col 0"
	)
	harness.assert_close(
		gradient_output[0][1],
		-0.6018748484,
		0.000001,
		"output gradient col 1"
	)
	harness.assert_close(
		gradient_b_output[0],
		-1.5004323265,
		0.000001,
		"output bias gradient"
	)

	var first_loss: float = network.mse_loss(prediction, target)
	for _index in range(25):
		network.train_sample(inputs, target, 0.2)
	var final_loss: float = network.mse_loss(network.forward(inputs), target)
	harness.assert_true(final_loss < first_loss, "training lowers loss")

	harness.finish(self)
