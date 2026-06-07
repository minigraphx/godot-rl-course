extends SceneTree

const Harness = preload("res://test/harness.gd")
const TinyNeuron = preload("res://shared/tiny_neuron.gd")
const JumperScene = preload("res://unit_01_jumper/unit_01_jumper.tscn")

func _init() -> void:
	call_deferred("_run")


func _run() -> void:
	var harness := Harness.new()
	var neuron := TinyNeuron.new()
	var section_inputs := PackedFloat32Array([0.5, 0.25])
	neuron.weights = PackedFloat32Array([0.8, 1.2])
	neuron.bias = -0.5
	neuron.activation = "sigmoid"
	var section_output := neuron.forward(section_inputs)
	harness.assert_close(
		section_output,
		1.0 / (1.0 + exp(-0.2)),
		0.000001,
		"forward pass"
	)
	_print_fixed_number_neuron(
		section_inputs,
		neuron.weights,
		neuron.bias,
		section_output
	)
	var demo := JumperScene.instantiate()
	root.add_child(demo)
	demo.refresh_lab()
	harness.assert_true(
		not demo.should_jump(0.30, 0.80),
		"slow and far waits"
	)
	harness.assert_true(
		demo.should_jump(0.90, 0.45),
		"fast and medium jumps"
	)
	harness.assert_true(
		demo.should_jump(0.30, 0.10),
		"slow and near jumps"
	)
	harness.assert_true(
		demo.get_node("Interface/LabPanel/Rows/SpeedCalculation").text.begins_with(
			"Speed input:"
		),
		"speed contribution is visible"
	)
	harness.assert_true(
		demo.get_node("Interface/LabPanel/Rows/ClosenessCalculation").text.begins_with(
			"Closeness input:"
		),
		"closeness contribution is visible"
	)
	harness.assert_true(
		demo.get_node("Interface/LabPanel/Rows/OutputCalculation").text.begins_with(
			"Sigmoid output:"
		),
		"sigmoid output is visible"
	)
	harness.assert_true(
		demo.get_node("Interface/LabPanel/Rows/CaseScore").text == "3 / 3 cases pass",
		"default parameters pass all guided cases"
	)
	_print_jumper_demo(demo)
	demo.queue_free()
	harness.finish(self)


func _print_fixed_number_neuron(
	inputs: PackedFloat32Array,
	weights: PackedFloat32Array,
	bias: float,
	output: float
) -> void:
	var weighted_sum := bias
	var names := ["speed", "closeness"]
	print("Fixed-number neuron (same numbers as section 1):")
	for index in range(inputs.size()):
		var contribution := inputs[index] * weights[index]
		weighted_sum += contribution
		print(
			"  %s input %.2f x weight %+.2f  ->  %+.2f"
			% [names[index], inputs[index], weights[index], contribution]
		)
	print("  bias                 %+.2f" % bias)
	print("  sum (z) = %+.3f" % weighted_sum)
	var fires := "fires: JUMP" if output > 0.5 else "quiet: WAIT"
	print("  sigmoid(sum) = %.3f  (%s)" % [output, fires])
	print()


func _print_jumper_demo(demo: Node2D) -> void:
	print("Cliff jumper demo (default scene wiring):")
	print("  %s" % demo.get_node("Interface/LabPanel/Rows/SpeedCalculation").text)
	print("  %s" % demo.get_node("Interface/LabPanel/Rows/ClosenessCalculation").text)
	print("  %s" % demo.get_node("Interface/LabPanel/Rows/BiasCalculation").text)
	print("  %s" % demo.get_node("Interface/LabPanel/Rows/SumCalculation").text)
	print("  %s" % demo.get_node("Interface/LabPanel/Rows/OutputCalculation").text)
	print("  Decision: %s" % demo.get_node("Interface/LabPanel/Rows/Decision").text)
	print("  Guided check: %s" % demo.get_node("Interface/LabPanel/Rows/CaseScore").text)
	print()
