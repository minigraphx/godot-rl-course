extends SceneTree

const Harness = preload("res://test/harness.gd")
const TinyNeuron = preload("res://shared/tiny_neuron.gd")
const EnemyScene = preload("res://unit_01_enemy/unit_01_enemy.tscn")

func _init() -> void:
	call_deferred("_run")


func _run() -> void:
	var harness := Harness.new()
	var neuron := TinyNeuron.new()
	neuron.weights = PackedFloat32Array([0.8, -0.4])
	neuron.bias = 0.1
	harness.assert_close(
		neuron.forward(PackedFloat32Array([0.5, -0.25])),
		tanh(0.6),
		0.000001,
		"forward pass"
	)
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
	demo.queue_free()
	quit(harness.failures)
