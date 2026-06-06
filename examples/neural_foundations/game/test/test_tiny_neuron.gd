extends SceneTree

const Harness = preload("res://test/harness.gd")
const TinyNeuron = preload("res://shared/tiny_neuron.gd")
const EnemyScene = preload("res://unit_01_enemy/unit_01_enemy.tscn")

func _init() -> void:
	call_deferred("_run")


func _run() -> void:
	var harness := Harness.new()
	var neuron := TinyNeuron.new()
	var section_inputs := PackedFloat32Array([0.5, -0.25])
	neuron.weights = PackedFloat32Array([0.8, -0.4])
	neuron.bias = 0.1
	var section_output := neuron.forward(section_inputs)
	harness.assert_close(
		section_output,
		tanh(0.6),
		0.000001,
		"forward pass"
	)
	_print_frozen_neuron(section_inputs, neuron.weights, neuron.bias, section_output)
	var demo := EnemyScene.instantiate()
	root.add_child(demo)
	demo._physics_process(1.0 / 60.0)
	var displayed_player: Vector2 = demo.get("displayed_player_position")
	var displayed_enemy: Vector2 = demo.get("displayed_enemy_position")
	var displayed_distance: float = demo.get("displayed_distance")
	var displayed_normalized_distance: float = demo.get(
		"displayed_normalized_distance"
	)
	harness.assert_close(
		displayed_distance,
		displayed_player.distance_to(displayed_enemy),
		0.000001,
		"displayed line distance"
	)
	harness.assert_close(
		displayed_normalized_distance,
		clampf(displayed_distance / demo.get("normalization_distance"), 0.0, 1.0),
		0.000001,
		"displayed normalized distance"
	)
	var distance_label: Label = demo.get_node(
		"Interface/Panel/Margin/Rows/DistanceCalculation"
	)
	harness.assert_true(
		distance_label.text.begins_with(
			"Distance: %.2f" % displayed_normalized_distance
		),
		"distance label uses the displayed decision input"
	)
	_print_enemy_demo(demo, displayed_distance, displayed_normalized_distance)
	demo.queue_free()
	harness.finish(self)


func _print_frozen_neuron(
	inputs: PackedFloat32Array,
	weights: PackedFloat32Array,
	bias: float,
	output: float
) -> void:
	var weighted_sum := bias
	print("Frozen neuron (same numbers as section 1):")
	for index in range(inputs.size()):
		var contribution := inputs[index] * weights[index]
		weighted_sum += contribution
		print(
			"  x%d=%.2f × w%d=%+.2f  ->  %+.2f"
			% [index + 1, inputs[index], index + 1, weights[index], contribution]
		)
	print("  bias                 %+.2f" % bias)
	print("  z = %+.3f" % weighted_sum)
	var sign := "positive" if output >= 0.0 else "negative"
	print("  tanh(z) = %+.3f  (%s)" % [output, sign])
	print()


func _print_enemy_demo(
	demo: Node2D,
	distance_px: float,
	normalized_distance: float
) -> void:
	print("Enemy demo (default scene wiring):")
	print(
		"  player/enemy distance: %.0f px  ->  input %.2f"
		% [distance_px, normalized_distance]
	)
	print("  %s" % demo.get_node("Interface/Panel/Margin/Rows/HealthCalculation").text)
	print("  %s" % demo.get_node("Interface/Panel/Margin/Rows/DistanceCalculation").text)
	print("  %s" % demo.get_node("Interface/Panel/Margin/Rows/BiasCalculation").text)
	print("  %s" % demo.get_node("Interface/Panel/Margin/Rows/SumCalculation").text)
	print("  %s" % demo.get_node("Interface/Panel/Margin/Rows/OutputCalculation").text)
	print("  Decision: %s" % demo.get_node("Interface/Panel/Margin/Rows/Behavior").text)
	print()
