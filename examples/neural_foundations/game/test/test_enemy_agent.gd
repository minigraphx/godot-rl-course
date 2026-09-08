extends SceneTree

## The enemy demo is the second view of the same neuron as the jumper: the same
## tiny_neuron.gd, but with tanh instead of sigmoid, so its threshold is 0.0
## rather than 0.5. These checks pin that difference down, and pin the three
## behaviours the unit's stretch goal asks the reader to reproduce.

const Harness = preload("res://test/harness.gd")
const TinyNeuron = preload("res://shared/tiny_neuron.gd")
const EnemyScene = preload("res://unit_01_enemy/unit_01_enemy.tscn")

func _init() -> void:
	call_deferred("_run")


func _run() -> void:
	var harness := Harness.new()

	# Same shared neuron as the jumper, different activation.
	var neuron := TinyNeuron.new()
	harness.assert_true(
		neuron.activation == "tanh",
		"the shared neuron defaults to tanh"
	)

	var demo := EnemyScene.instantiate()
	root.add_child(demo)

	# Healthy and close: the enemy commits.
	harness.assert_true(
		demo.should_chase(1.0, 0.1),
		"healthy and near chases"
	)
	# Badly hurt: it backs off even when the player is right there.
	harness.assert_true(
		not demo.should_chase(0.0, 0.1),
		"hurt and near retreats"
	)
	# Distance carries a negative weight, so far away pushes toward retreat.
	harness.assert_true(
		not demo.should_chase(0.4, 1.0),
		"far away retreats"
	)

	# tanh is centred on zero: equal and opposite inputs cancel to exactly 0.0,
	# and the >= comparison makes that boundary case a chase.
	var zero_output: float = demo.neuron_output(0.0, 0.0)
	harness.assert_close(
		zero_output,
		tanh(demo.bias),
		0.000001,
		"zero inputs leave only the bias"
	)

	# The forward pass must stay visible — that is the whole point of the scene.
	harness.assert_true(
		demo.get_node("Interface/Panel/Margin/Rows/HealthCalculation")
			.text.begins_with("Health:"),
		"health contribution is visible"
	)
	harness.assert_true(
		demo.get_node("Interface/Panel/Margin/Rows/OutputCalculation")
			.text.begins_with("tanh(z):"),
		"the output label names tanh, not sigmoid"
	)

	_print_walkthrough(demo)
	harness.finish(self)


func _print_walkthrough(demo: Node) -> void:
	print("")
	print("Enemy demo (same neuron as the jumper, tanh instead of sigmoid):")
	print("  health %+.2f x weight %+.2f" % [1.0, demo.health_weight])
	print("  distance %+.2f x weight %+.2f" % [0.1, demo.distance_weight])
	print("  bias                        %+.2f" % demo.bias)
	print("  tanh(z) = %+.3f  ->  %s" % [
		demo.neuron_output(1.0, 0.1),
		"CHASE" if demo.should_chase(1.0, 0.1) else "RETREAT",
	])
	print("")
