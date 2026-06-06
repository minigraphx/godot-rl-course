extends Node2D

const TinyNeuronScript = preload("res://shared/tiny_neuron.gd")

@export_range(-3.0, 3.0, 0.05) var health_weight := 1.4
@export_range(-3.0, 3.0, 0.05) var distance_weight := -1.2
@export_range(-2.0, 2.0, 0.05) var bias := -0.1
@export_range(20.0, 300.0, 5.0) var enemy_speed := 90.0
@export_range(100.0, 800.0, 10.0) var normalization_distance := 500.0

@onready var player: Polygon2D = $Player
@onready var enemy: Polygon2D = $Enemy
@onready var health_bar: ProgressBar = $Interface/Panel/Margin/Rows/HealthBar
@onready var health_slider: HSlider = $Interface/Panel/Margin/Rows/HealthSlider
@onready var health_calculation: Label = $Interface/Panel/Margin/Rows/HealthCalculation
@onready var distance_calculation: Label = $Interface/Panel/Margin/Rows/DistanceCalculation
@onready var bias_calculation: Label = $Interface/Panel/Margin/Rows/BiasCalculation
@onready var sum_calculation: Label = $Interface/Panel/Margin/Rows/SumCalculation
@onready var output_calculation: Label = $Interface/Panel/Margin/Rows/OutputCalculation
@onready var behavior_label: Label = $Interface/Panel/Margin/Rows/Behavior
@onready var pause_button: Button = $Interface/Buttons/Pause

var neuron: RefCounted = TinyNeuronScript.new()
var paused := false
var displayed_player_position := Vector2.ZERO
var displayed_enemy_position := Vector2.ZERO
var displayed_distance := 0.0
var displayed_normalized_distance := 0.0


func _ready() -> void:
	health_slider.value_changed.connect(_on_health_changed)
	pause_button.pressed.connect(_on_pause_pressed)
	$Interface/Buttons/Reset.pressed.connect(_reset_demo)
	_reset_demo()


func _physics_process(delta: float) -> void:
	_move_player(delta)
	neuron.weights = PackedFloat32Array([health_weight, distance_weight])
	neuron.bias = bias

	var health := health_slider.value / health_slider.max_value
	var distance := player.position.distance_to(enemy.position)
	var normalized_distance := clampf(distance / normalization_distance, 0.0, 1.0)
	displayed_player_position = player.position
	displayed_enemy_position = enemy.position
	displayed_distance = distance
	displayed_normalized_distance = normalized_distance
	var inputs := PackedFloat32Array([health, normalized_distance])
	var health_contribution := health * health_weight
	var distance_contribution := normalized_distance * distance_weight
	var weighted_sum := health_contribution + distance_contribution + bias
	var output: float = neuron.forward(inputs)
	var chasing: bool = output >= 0.0

	if not paused:
		var direction := enemy.position.direction_to(player.position)
		if not chasing:
			direction *= -1.0
		enemy.position += direction * enemy_speed * delta
		enemy.position.x = clampf(enemy.position.x, 30.0, 930.0)
		enemy.position.y = clampf(enemy.position.y, 50.0, 510.0)

	health_bar.value = health_slider.value
	enemy.color = Color("#ef8354") if chasing else Color("#4f83cc")
	health_calculation.text = (
		"Health:   %.2f × %+.2f = %+.3f"
		% [health, health_weight, health_contribution]
	)
	distance_calculation.text = (
		"Distance: %.2f × %+.2f = %+.3f"
		% [normalized_distance, distance_weight, distance_contribution]
	)
	bias_calculation.text = "Bias:                    %+.3f" % bias
	sum_calculation.text = "Weighted sum z:          %+.3f" % weighted_sum
	output_calculation.text = "tanh(z):                 %+.3f" % output
	behavior_label.text = "CHASE" if chasing else "RETREAT"
	behavior_label.modulate = enemy.color
	queue_redraw()


func _draw() -> void:
	if not is_instance_valid(player) or not is_instance_valid(enemy):
		return
	draw_dashed_line(
		displayed_enemy_position,
		displayed_player_position,
		Color("#d8dee9"),
		2.0,
		10.0,
	)
	var midpoint := (displayed_player_position + displayed_enemy_position) * 0.5
	draw_string(
		ThemeDB.fallback_font,
		midpoint + Vector2(8.0, -8.0),
		"%.0f px · input %.2f" % [
			displayed_distance,
			displayed_normalized_distance,
		],
		HORIZONTAL_ALIGNMENT_LEFT,
		-1.0,
		16,
		Color.WHITE,
	)


func _move_player(delta: float) -> void:
	var input := Input.get_vector("ui_left", "ui_right", "ui_up", "ui_down")
	player.position += input * 170.0 * delta
	player.position.x = clampf(player.position.x, 30.0, 930.0)
	player.position.y = clampf(player.position.y, 50.0, 510.0)


func _on_health_changed(value: float) -> void:
	health_bar.value = value


func _on_pause_pressed() -> void:
	paused = not paused
	pause_button.text = "Resume" if paused else "Pause"


func _reset_demo() -> void:
	player.position = Vector2(720.0, 300.0)
	enemy.position = Vector2(380.0, 300.0)
	displayed_player_position = player.position
	displayed_enemy_position = enemy.position
	displayed_distance = player.position.distance_to(enemy.position)
	displayed_normalized_distance = clampf(
		displayed_distance / normalization_distance,
		0.0,
		1.0,
	)
	health_slider.value = 75.0
	paused = false
	pause_button.text = "Pause"
