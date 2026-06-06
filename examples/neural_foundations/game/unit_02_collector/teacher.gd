class_name CollectorTeacher
extends RefCounted

var hazard_radius := 120.0


func target_movement(
	agent_position: Vector2,
	gem_position: Vector2,
	hazard_position: Vector2
) -> Vector2:
	var hazard_offset := agent_position - hazard_position
	if hazard_offset.length() <= hazard_radius and hazard_offset.length() > 0.0:
		return hazard_offset.normalized()

	var gem_offset := gem_position - agent_position
	if gem_offset.length() == 0.0:
		return Vector2.ZERO
	return gem_offset.normalized()
