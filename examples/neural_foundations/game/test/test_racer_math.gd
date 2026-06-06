extends SceneTree

const Harness = preload("res://test/harness.gd")


func _init() -> void:
	call_deferred("_run")


func _run() -> void:
	var harness := Harness.new()
	var racer_script := load("res://unit_03_racer/racer.gd")
	if racer_script == null:
		harness.assert_true(false, "racer script exists")
		harness.finish(self)
		return

	var racer: Node2D = racer_script.new()
	racer.track_bounds = Rect2(Vector2.ZERO, Vector2(100.0, 100.0))
	racer.start_position = Vector2(10.0, 50.0)
	racer.start_rotation = 0.0
	racer.ray_length = 90.0
	racer.checkpoints = PackedVector2Array([Vector2(30.0, 50.0), Vector2(60.0, 50.0)])
	racer.reset_racer()

	var rays: PackedFloat32Array = racer.normalized_ray_distances()
	harness.assert_true(rays.size() == 3, "racer exposes three ray distances")
	harness.assert_close(rays[1], 1.0, 0.000001, "forward ray normalizes to max distance")
	harness.assert_true(rays[0] >= 0.0 and rays[0] <= 1.0, "left ray is normalized")
	harness.assert_true(rays[2] >= 0.0 and rays[2] <= 1.0, "right ray is normalized")

	harness.assert_close(
		absf(racer.normalized_heading_error(Vector2(0.0, 50.0))),
		1.0,
		0.000001,
		"heading error wraps to normalized half turn"
	)

	racer.set_action({"drive": [2.0, -2.0]})
	harness.assert_close(racer.steering, 1.0, 0.000001, "steering clamps high")
	harness.assert_close(racer.throttle, -1.0, 0.000001, "throttle clamps low")

	racer.global_position = Vector2(30.0, 50.0)
	var checkpoint_reward: float = racer.apply_checkpoint_reward()
	harness.assert_close(checkpoint_reward, racer.checkpoint_reward, 0.000001, "ordered checkpoint reward")
	harness.assert_true(racer.next_checkpoint_index == 1, "checkpoint advances in order")

	racer.global_position = Vector2(-1.0, 50.0)
	harness.assert_close(
		racer.collision_penalty_if_needed(),
		racer.collision_penalty,
		0.000001,
		"collision penalty"
	)
	harness.assert_true(racer.done, "collision ends episode")

	racer.reset_racer()
	racer.speed = 0.0
	for _step in range(racer.immobility_limit):
		racer.update_immobility()
	harness.assert_true(racer.done, "immobility timeout ends episode")

	racer.reset_racer()
	harness.assert_close(racer.global_position.x, 10.0, 0.000001, "reset x")
	harness.assert_close(racer.global_position.y, 50.0, 0.000001, "reset y")
	harness.assert_close(racer.rotation, 0.0, 0.000001, "reset rotation")
	harness.assert_true(not racer.done, "reset clears done")

	racer.free()
	harness.finish(self)
