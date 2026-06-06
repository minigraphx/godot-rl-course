extends SceneTree

const Harness = preload("res://test/harness.gd")


func _init() -> void:
	call_deferred("_run")


func _run() -> void:
	var harness := Harness.new()
	var agent_script := load("res://unit_02_collector/collector_agent.gd")
	if agent_script == null:
		harness.assert_true(false, "collector agent script exists")
		harness.finish(self)
		return

	var collector: Node2D = agent_script.new()
	collector.sensor_radius = 10.0

	var inputs: PackedFloat32Array = collector.observation_inputs(
		Vector2.ZERO,
		Vector2(3.0, 4.0),
		Vector2(6.0, 8.0)
	)
	harness.assert_true(inputs.size() == 4, "collector exposes four learnable inputs")
	if inputs.size() != 4:
		collector.free()
		harness.finish(self)
		return
	harness.assert_close(inputs[0], 0.6, 0.000001, "gem direction x")
	harness.assert_close(inputs[1], 0.8, 0.000001, "gem direction y")
	harness.assert_close(inputs[2], 0.6, 0.000001, "hazard offset x")
	harness.assert_close(inputs[3], 0.8, 0.000001, "hazard offset y")

	var close_hazard_inputs: PackedFloat32Array = collector.observation_inputs(
		Vector2.ZERO,
		Vector2(3.0, 4.0),
		Vector2(3.0, 4.0)
	)
	harness.assert_close(close_hazard_inputs[2], 0.3, 0.000001, "close hazard offset x")
	harness.assert_close(close_hazard_inputs[3], 0.4, 0.000001, "close hazard offset y")

	var velocity: Vector2 = collector.prediction_to_velocity(
		PackedFloat32Array([2.0, -0.5]),
		100.0
	)
	harness.assert_close(velocity.length(), 100.0, 0.0001, "velocity is clamped")
	harness.assert_true(velocity.x > 0.0, "velocity keeps x direction")
	harness.assert_true(velocity.y < 0.0, "velocity keeps y direction")

	collector.free()
	harness.finish(self)
