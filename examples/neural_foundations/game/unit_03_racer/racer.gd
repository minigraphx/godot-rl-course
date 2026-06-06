class_name FoundationsRacer
extends Node2D

@export var track_bounds := Rect2(Vector2(80.0, 60.0), Vector2(800.0, 420.0))
@export var start_position := Vector2(160.0, 270.0)
@export var start_rotation := 0.0
@export var ray_length := 120.0
@export var ray_angle := PI / 4.0
@export var acceleration := 180.0
@export var turn_speed := 2.8
@export var drag := 0.92
@export var max_speed := 220.0
@export var checkpoint_radius := 36.0
@export var checkpoint_reward := 1.0
@export var collision_penalty := -1.0
@export var step_penalty := -0.01
@export var immobility_limit := 90
@export var immobility_speed_threshold := 3.0

var checkpoints := PackedVector2Array([
	Vector2(320.0, 140.0),
	Vector2(720.0, 140.0),
	Vector2(760.0, 400.0),
	Vector2(260.0, 410.0),
])
var velocity := Vector2.ZERO
var steering := 0.0
var throttle := 0.0
var speed := 0.0
var done := false
var episode_return := 0.0
var step_reward := 0.0
var next_checkpoint_index := 0
var immobile_steps := 0
var steps := 0


func reset_racer() -> void:
	global_position = start_position
	rotation = start_rotation
	velocity = Vector2.ZERO
	steering = 0.0
	throttle = 0.0
	speed = 0.0
	done = false
	episode_return = 0.0
	step_reward = 0.0
	next_checkpoint_index = 0
	immobile_steps = 0
	steps = 0


func set_action(action) -> void:
	var drive = action
	if action is Dictionary:
		drive = action.get("drive", [0.0, 0.0])
	if drive is PackedFloat32Array or drive is Array:
		steering = clampf(float(drive[0]), -1.0, 1.0)
		throttle = clampf(float(drive[1]), -1.0, 1.0)


func normalized_ray_distances() -> PackedFloat32Array:
	return PackedFloat32Array([
		normalized_ray_distance(rotation - ray_angle),
		normalized_ray_distance(rotation),
		normalized_ray_distance(rotation + ray_angle),
	])


func normalized_ray_distance(angle: float) -> float:
	var direction := Vector2.RIGHT.rotated(angle)
	var distances: Array[float] = []
	if direction.x > 0.0:
		distances.append((track_bounds.end.x - global_position.x) / direction.x)
	elif direction.x < 0.0:
		distances.append((track_bounds.position.x - global_position.x) / direction.x)
	if direction.y > 0.0:
		distances.append((track_bounds.end.y - global_position.y) / direction.y)
	elif direction.y < 0.0:
		distances.append((track_bounds.position.y - global_position.y) / direction.y)
	var nearest := ray_length
	for distance in distances:
		if distance >= 0.0:
			nearest = minf(nearest, distance)
	return clampf(nearest / ray_length, 0.0, 1.0)


func normalized_heading_error(target: Vector2) -> float:
	var target_angle := global_position.angle_to_point(target)
	return wrapf(target_angle - rotation, -PI, PI) / PI


func apply_checkpoint_reward() -> float:
	if checkpoints.is_empty():
		return 0.0
	var checkpoint := checkpoints[next_checkpoint_index]
	if global_position.distance_to(checkpoint) > checkpoint_radius:
		return 0.0
	next_checkpoint_index = (next_checkpoint_index + 1) % checkpoints.size()
	return checkpoint_reward


func collision_penalty_if_needed() -> float:
	if track_bounds.has_point(global_position):
		return 0.0
	done = true
	return collision_penalty


func update_immobility() -> void:
	if absf(speed) < immobility_speed_threshold:
		immobile_steps += 1
	else:
		immobile_steps = 0
	if immobile_steps >= immobility_limit:
		done = true


func target_checkpoint() -> Vector2:
	if checkpoints.is_empty():
		return global_position
	return checkpoints[next_checkpoint_index]


func observation_values() -> PackedFloat32Array:
	var rays := normalized_ray_distances()
	return PackedFloat32Array([
		rays[0],
		rays[1],
		rays[2],
		normalized_heading_error(target_checkpoint()),
		clampf(speed / max_speed, -1.0, 1.0),
		float(next_checkpoint_index) / max(1.0, float(checkpoints.size())),
	])


func simulate_step(delta: float) -> void:
	if done:
		return
	rotation += steering * turn_speed * delta
	speed = clampf(speed + throttle * acceleration * delta, -max_speed * 0.35, max_speed)
	velocity = Vector2.RIGHT.rotated(rotation) * speed
	global_position += velocity * delta
	speed *= drag
	steps += 1
	step_reward = step_penalty
	step_reward += apply_checkpoint_reward()
	step_reward += collision_penalty_if_needed()
	update_immobility()
	episode_return += step_reward


func _ready() -> void:
	reset_racer()


func _physics_process(delta: float) -> void:
	simulate_step(delta)
