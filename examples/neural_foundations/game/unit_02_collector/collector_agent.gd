class_name CollectorAgent
extends Node2D

const CollectorTeacherScript = preload("res://unit_02_collector/teacher.gd")
const TinyMLPScript = preload("res://shared/tiny_mlp.gd")

var sensor_radius := 240.0
var agent_speed := 120.0
var learning_rate := 0.08

var teacher: RefCounted = CollectorTeacherScript.new()
var network: RefCounted = TinyMLPScript.new()
var last_inputs := PackedFloat32Array()
var last_target := PackedFloat32Array()
var last_prediction := PackedFloat32Array()
var last_loss := 0.0
var training_steps := 0
var gems_collected := 0
var hazards_hit := 0
var replay_index := 0
var replay_states := [
	{
		"agent": Vector2(420.0, 300.0),
		"gem": Vector2(720.0, 260.0),
		"hazard": Vector2(330.0, 360.0),
	},
	{
		"agent": Vector2(520.0, 260.0),
		"gem": Vector2(260.0, 180.0),
		"hazard": Vector2(580.0, 320.0),
	},
	{
		"agent": Vector2(360.0, 380.0),
		"gem": Vector2(760.0, 420.0),
		"hazard": Vector2(410.0, 330.0),
	},
]

@onready var agent_body: Polygon2D = get_node_or_null("Agent")
@onready var gem: Polygon2D = get_node_or_null("Gem")
@onready var hazard: Polygon2D = get_node_or_null("Hazard")
@onready var gem_direction_line: Line2D = get_node_or_null("GemDirectionLine")
@onready var hazard_direction_line: Line2D = get_node_or_null("HazardDirectionLine")
@onready var target_arrow: Line2D = get_node_or_null("TargetArrow")
@onready var prediction_arrow: Line2D = get_node_or_null("PredictionArrow")
@onready var input_label: Label = get_node_or_null("Interface/Panel/Margin/Rows/Inputs")
@onready var target_label: Label = get_node_or_null("Interface/Panel/Margin/Rows/Target")
@onready var prediction_label: Label = get_node_or_null("Interface/Panel/Margin/Rows/Prediction")
@onready var loss_label: Label = get_node_or_null("Interface/Panel/Margin/Rows/Loss")
@onready var counts_label: Label = get_node_or_null("Interface/Panel/Margin/Rows/Counts")
@onready var target_x_bar: ProgressBar = get_node_or_null("Interface/Panel/Margin/Rows/TargetXBar")
@onready var target_y_bar: ProgressBar = get_node_or_null("Interface/Panel/Margin/Rows/TargetYBar")
@onready var prediction_x_bar: ProgressBar = get_node_or_null("Interface/Panel/Margin/Rows/PredictionXBar")
@onready var prediction_y_bar: ProgressBar = get_node_or_null("Interface/Panel/Margin/Rows/PredictionYBar")


func _ready() -> void:
	teacher.hazard_radius = sensor_radius * 0.55
	_apply_replay_state()
	_update_demo(0.0)


func _physics_process(delta: float) -> void:
	_update_demo(delta)


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
	var hazard_offset := (hazard_position - agent_position) / sensor_radius
	if hazard_distance > sensor_radius and hazard_offset.length() > 0.0:
		hazard_offset = hazard_offset.normalized()

	return PackedFloat32Array([
		gem_direction.x,
		gem_direction.y,
		hazard_offset.x,
		hazard_offset.y,
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


func _update_demo(delta: float) -> void:
	if not _has_scene_nodes():
		return

	var target_vector := _capture_current_outputs()
	last_loss = network.train_sample(last_inputs, last_target, learning_rate)

	if delta > 0.0:
		training_steps += 1
		agent_body.position += prediction_to_velocity(last_prediction, agent_speed) * delta
		agent_body.position.x = clampf(agent_body.position.x, 40.0, 920.0)
		agent_body.position.y = clampf(agent_body.position.y, 80.0, 500.0)
		if _update_replay_counters():
			target_vector = _capture_current_outputs()

	_update_lines(target_vector)
	_update_output_bars()
	_update_labels()


func _capture_current_outputs() -> Vector2:
	last_inputs = observation_inputs(
		agent_body.position,
		gem.position,
		hazard.position
	)
	var target_vector: Vector2 = teacher.target_movement(
		agent_body.position,
		gem.position,
		hazard.position
	)
	last_target = PackedFloat32Array([target_vector.x, target_vector.y])
	last_prediction = network.forward(last_inputs)
	last_loss = network.mse_loss(last_prediction, last_target)
	return target_vector


func _has_scene_nodes() -> bool:
	return (
		agent_body != null
		and gem != null
		and hazard != null
		and gem_direction_line != null
		and hazard_direction_line != null
		and target_arrow != null
		and prediction_arrow != null
	)


func _update_lines(target_vector: Vector2) -> void:
	var origin := agent_body.position
	var gem_vector := Vector2(last_inputs[0], last_inputs[1])
	var hazard_vector := Vector2(last_inputs[2], last_inputs[3])
	gem_direction_line.points = PackedVector2Array([
		origin,
		origin + gem_vector * 80.0,
	])
	hazard_direction_line.points = PackedVector2Array([
		origin,
		origin + hazard_vector * 80.0,
	])
	target_arrow.points = PackedVector2Array([
		origin,
		origin + target_vector * 70.0,
	])
	var prediction_vector := Vector2(last_prediction[0], last_prediction[1])
	prediction_arrow.points = PackedVector2Array([
		origin,
		origin + prediction_vector * 70.0,
	])


func _update_output_bars() -> void:
	_set_bar_value(target_x_bar, last_target[0])
	_set_bar_value(target_y_bar, last_target[1])
	_set_bar_value(prediction_x_bar, last_prediction[0])
	_set_bar_value(prediction_y_bar, last_prediction[1])


func _set_bar_value(bar: ProgressBar, value: float) -> void:
	if bar == null:
		return
	bar.value = (clampf(value, -1.0, 1.0) + 1.0) * 50.0


func _update_labels() -> void:
	if input_label != null:
		input_label.text = "Inputs: gem=(%+.2f, %+.2f) hazard=(%+.2f, %+.2f)" % [
			last_inputs[0],
			last_inputs[1],
			last_inputs[2],
			last_inputs[3],
		]
	if target_label != null:
		target_label.text = "Target: (%+.2f, %+.2f)" % [
			last_target[0],
			last_target[1],
		]
	if prediction_label != null:
		prediction_label.text = "Prediction: (%+.2f, %+.2f)" % [
			last_prediction[0],
			last_prediction[1],
		]
	if loss_label != null:
		loss_label.text = "Loss: %.4f" % last_loss
	if counts_label != null:
		counts_label.text = "Steps: %d  Gems: %d  Hazards: %d  Replay: %d" % [
			training_steps,
			gems_collected,
			hazards_hit,
			replay_index + 1,
		]


func _update_replay_counters() -> bool:
	if agent_body.position.distance_to(gem.position) < 28.0:
		gems_collected += 1
		_advance_replay()
		return true
	elif agent_body.position.distance_to(hazard.position) < 30.0:
		hazards_hit += 1
		_advance_replay()
		return true
	return false


func _advance_replay() -> void:
	replay_index = (replay_index + 1) % replay_states.size()
	_apply_replay_state()


func _apply_replay_state() -> void:
	if agent_body == null or gem == null or hazard == null:
		return
	var state: Dictionary = replay_states[replay_index]
	agent_body.position = state["agent"]
	gem.position = state["gem"]
	hazard.position = state["hazard"]
