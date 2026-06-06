extends SceneTree

const Harness = preload("res://test/harness.gd")


func _init() -> void:
	call_deferred("_run")


func _run() -> void:
	var harness := Harness.new()
	var teacher_script := load("res://unit_02_collector/teacher.gd")
	if teacher_script == null:
		harness.assert_true(false, "collector teacher script exists")
		quit(harness.failures)
		return

	var teacher: RefCounted = teacher_script.new()
	teacher.hazard_radius = 5.0

	var toward_gem: Vector2 = teacher.target_movement(
		Vector2.ZERO,
		Vector2(10.0, 0.0),
		Vector2(100.0, 0.0)
	)
	harness.assert_close(toward_gem.x, 1.0, 0.000001, "far hazard moves toward gem x")
	harness.assert_close(toward_gem.y, 0.0, 0.000001, "far hazard moves toward gem y")

	var away_from_hazard: Vector2 = teacher.target_movement(
		Vector2.ZERO,
		Vector2(10.0, 0.0),
		Vector2(1.0, 0.0)
	)
	harness.assert_close(away_from_hazard.x, -1.0, 0.000001, "close hazard moves away x")
	harness.assert_close(away_from_hazard.y, 0.0, 0.000001, "close hazard moves away y")

	quit(harness.failures)
