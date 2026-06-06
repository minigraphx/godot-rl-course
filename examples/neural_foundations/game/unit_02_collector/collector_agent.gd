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

@onready var agent_body: Polygon2D = get_node_or_null("Agent")
@onready var gem: Polygon2D = get_node_or_null("Gem")
@onready var hazard: Polygon2D = get_node_or_null("Hazard")
@onready var target_arrow: Line2D = get_node_or_null("TargetArrow")
@onready var prediction_arrow: Line2D = get_node_or_null("PredictionArrow")
@onready var input_label: Label = get_node_or_null("Interface/Panel/Margin/Rows/Inputs")
@onready var target_label: Label = get_node_or_null("Interface/Panel/Margin/Rows/Target")
@onready var prediction_label: Label = get_node_or_null("Interface/Panel/Margin/Rows/Prediction")
@onready var loss_label: Label = get_node_or_null("Interface/Panel/Margin/Rows/Loss")
@onready var counts_label: Label = get_node_or_null("Interface/Panel/Margin/Rows/Counts")


func _ready() -> void:
	teacher.hazard_radius = sensor_radius * 0.55
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
	last_loss = network.train_sample(last_inputs, last_target, learning_rate)

	if delta > 0.0:
		agent_body.position += prediction_to_velocity(last_prediction, agent_speed) * delta
		agent_body.position.x = clampf(agent_body.position.x, 40.0, 920.0)
		agent_body.position.y = clampf(agent_body.position.y, 80.0, 500.0)

	_update_arrows(target_vector)
	_update_labels()


func _has_scene_nodes() -> bool:
	return (
		agent_body != null
		and gem != null
		and hazard != null
		and target_arrow != null
		and prediction_arrow != null
	)


func _update_arrows(target_vector: Vector2) -> void:
	var origin := agent_body.position
	target_arrow.points = PackedVector2Array([
		origin,
		origin + target_vector * 70.0,
	])
	var prediction_vector := Vector2(last_prediction[0], last_prediction[1])
	prediction_arrow.points = PackedVector2Array([
		origin,
		origin + prediction_vector * 70.0,
	])


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
		counts_label.text = "Teacher: move away near hazard, otherwise move to gem"
