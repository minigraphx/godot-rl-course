class_name CollectorAgent
extends Node2D

var sensor_radius := 240.0


func observation_inputs(
	agent_position: Vector2,
	gem_position: Vector2,
	hazard_position: Vector2
) -> PackedFloat32Array:
	var gem_offset := gem_position - agent_position
	var gem_direction := Vector2.ZERO
	if gem_offset.length() > 0.0:
		gem_direction = gem_offset.normalized()

	var hazard_distance := agent_position.distance_to(hazard_position)
	var hazard_proximity := 1.0 - clampf(hazard_distance / sensor_radius, 0.0, 1.0)

	return PackedFloat32Array([
		gem_direction.x,
		gem_direction.y,
		hazard_proximity,
	])


func prediction_to_velocity(
	prediction: PackedFloat32Array,
	speed: float
) -> Vector2:
	assert(prediction.size() == 2)
	var direction := Vector2(
		clampf(prediction[0], -1.0, 1.0),
		clampf(prediction[1], -1.0, 1.0)
	)
	if direction.length() > 1.0:
		direction = direction.normalized()
	return direction * speed
