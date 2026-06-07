extends Node2D

const TinyNeuronScript = preload("res://shared/tiny_neuron.gd")
const TILESET_PATH := "res://unit_01_jumper/assets/cliff_jumper_tileset.png"

const GUIDED_CASES := [
	{"speed": 0.30, "distance": 0.80, "jump": false},
	{"speed": 0.90, "distance": 0.45, "jump": true},
	{"speed": 0.30, "distance": 0.10, "jump": true},
]
const RUN_START_X := 500.0
const CLIFF_X := 880.0
const GROUND_Y := 560.0
const MAX_APPROACH_DISTANCE := CLIFF_X - RUN_START_X
const GRAVITY := 980.0
const JUMP_VELOCITY := -430.0

@export_range(-3.0, 3.0, 0.05) var speed_weight := 1.2
@export_range(-3.0, 3.0, 0.05) var closeness_weight := 2.0
@export_range(-3.0, 3.0, 0.05) var bias := -1.6

@onready var speed_slider: HSlider = $Interface/LabPanel/Rows/SpeedSlider
@onready var distance_slider: HSlider = $Interface/LabPanel/Rows/DistanceSlider
@onready var speed_weight_slider: HSlider = $Interface/LabPanel/Rows/SpeedWeightSlider
@onready var closeness_weight_slider: HSlider = $Interface/LabPanel/Rows/ClosenessWeightSlider
@onready var bias_slider: HSlider = $Interface/LabPanel/Rows/BiasSlider
@onready var speed_calculation: Label = $Interface/LabPanel/Rows/SpeedCalculation
@onready var closeness_calculation: Label = $Interface/LabPanel/Rows/ClosenessCalculation
@onready var bias_calculation: Label = $Interface/LabPanel/Rows/BiasCalculation
@onready var sum_calculation: Label = $Interface/LabPanel/Rows/SumCalculation
@onready var output_calculation: Label = $Interface/LabPanel/Rows/OutputCalculation
@onready var decision_label: Label = $Interface/LabPanel/Rows/Decision
@onready var case_score: Label = $Interface/LabPanel/Rows/CaseScore
@onready var run_status: Label = $Interface/RunStatus

var neuron: RefCounted = TinyNeuronScript.new()
var run_active := false
var run_speed_input := 0.5
var runner_position := Vector2(RUN_START_X, GROUND_Y - 38.0)
var runner_velocity_y := 0.0
var jump_triggered := false
var jump_distance := 1.0
var predicted_result := ""
var tileset_texture: ImageTexture


func _ready() -> void:
	var tileset_image := Image.load_from_file(TILESET_PATH)
	if not tileset_image.is_empty():
		tileset_texture = ImageTexture.create_from_image(tileset_image)
	get_viewport().size_changed.connect(_on_viewport_changed)
	for slider in [
		speed_slider,
		distance_slider,
		speed_weight_slider,
		closeness_weight_slider,
		bias_slider,
	]:
		slider.value_changed.connect(_on_slider_changed)
	$Interface/Buttons/TestRun.pressed.connect(start_test_run)
	$Interface/Buttons/Reset.pressed.connect(reset_demo)
	speed_weight_slider.value = speed_weight
	closeness_weight_slider.value = closeness_weight
	bias_slider.value = bias
	refresh_lab()
	reset_demo()


func _process(delta: float) -> void:
	if not run_active:
		return

	var speed_px := lerpf(115.0, 245.0, run_speed_input)
	runner_position.x += speed_px * delta
	var remaining_distance := clampf(
		(CLIFF_X - runner_position.x) / MAX_APPROACH_DISTANCE,
		0.0,
		1.0,
	)

	if not jump_triggered and should_jump(run_speed_input, remaining_distance):
		jump_triggered = true
		jump_distance = remaining_distance
		runner_velocity_y = JUMP_VELOCITY
		predicted_result = _classify_takeoff(run_speed_input, jump_distance)

	if jump_triggered:
		runner_velocity_y += GRAVITY * delta
		runner_position.y += runner_velocity_y * delta

	if not jump_triggered and runner_position.x >= CLIFF_X:
		_finish_run("TOO LATE: the neuron did not fire before the edge.")
	elif jump_triggered and runner_position.y >= GROUND_Y - 38.0:
		if predicted_result == "LANDED" and runner_position.x >= CLIFF_X + 70.0:
			_finish_run("LANDED: speed and distance produced useful timing.")
		elif predicted_result == "TOO EARLY":
			_finish_run("TOO EARLY: the jump arc ended before the platform.")
		elif predicted_result == "TOO LATE":
			_finish_run("TOO LATE: there was not enough time to cross the gap.")

	if runner_position.y > get_viewport_rect().size.y + 80.0:
		_finish_run("FELL INTO LAVA: adjust the weights or bias.")

	queue_redraw()


func should_jump(speed_input: float, distance_input: float) -> bool:
	return _evaluate(speed_input, distance_input)["sum"] > 0.000001


func refresh_lab() -> void:
	if not is_node_ready():
		return
	speed_weight = speed_weight_slider.value
	closeness_weight = closeness_weight_slider.value
	bias = bias_slider.value

	var speed_input := speed_slider.value
	var distance_input := distance_slider.value
	var values := _evaluate(speed_input, distance_input)
	speed_calculation.text = (
		"Speed input:     %.2f x %+.2f = %+.3f"
		% [speed_input, speed_weight, values["speed_contribution"]]
	)
	closeness_calculation.text = (
		"Closeness input: %.2f x %+.2f = %+.3f"
		% [values["closeness"], closeness_weight, values["closeness_contribution"]]
	)
	bias_calculation.text = "Bias:                         %+.3f" % bias
	sum_calculation.text = "Sum (z):                      %+.3f" % values["sum"]
	output_calculation.text = "Sigmoid output:               %.3f" % values["output"]
	var fires: bool = values["sum"] > 0.000001
	decision_label.text = "JUMP - neuron fires" if fires else "WAIT - neuron stays quiet"
	decision_label.modulate = Color("#f6c453") if fires else Color("#8bd3dd")
	case_score.text = "%d / 3 cases pass" % _guided_case_score()
	queue_redraw()


func start_test_run() -> void:
	run_speed_input = [0.30, 0.60, 0.90].pick_random()
	runner_position = Vector2(RUN_START_X, GROUND_Y - 38.0)
	runner_velocity_y = 0.0
	jump_triggered = false
	jump_distance = 1.0
	predicted_result = ""
	run_active = true
	run_status.text = "RUNNING: speed input %.2f" % run_speed_input
	queue_redraw()


func reset_demo() -> void:
	run_active = false
	runner_position = Vector2(RUN_START_X, GROUND_Y - 38.0)
	runner_velocity_y = 0.0
	jump_triggered = false
	run_status.text = "LAB MODE: set the neuron, then press Test run."
	queue_redraw()


func _evaluate(speed_input: float, distance_input: float) -> Dictionary:
	var closeness := 1.0 - clampf(distance_input, 0.0, 1.0)
	var speed_contribution := speed_input * speed_weight
	var closeness_contribution := closeness * closeness_weight
	var weighted_sum := speed_contribution + closeness_contribution + bias
	neuron.weights = PackedFloat32Array([speed_weight, closeness_weight])
	neuron.bias = bias
	neuron.activation = "sigmoid"
	var output: float = neuron.forward(
		PackedFloat32Array([speed_input, closeness])
	)
	return {
		"closeness": closeness,
		"speed_contribution": speed_contribution,
		"closeness_contribution": closeness_contribution,
		"sum": weighted_sum,
		"output": output,
	}


func _guided_case_score() -> int:
	var passed := 0
	for test_case in GUIDED_CASES:
		if should_jump(test_case["speed"], test_case["distance"]) == test_case["jump"]:
			passed += 1
	return passed


func _classify_takeoff(speed_input: float, distance_input: float) -> String:
	var ideal_distance := lerpf(0.10, 0.38, speed_input)
	if distance_input > ideal_distance + 0.13:
		return "TOO EARLY"
	if distance_input < ideal_distance - 0.10:
		return "TOO LATE"
	return "LANDED"


func _finish_run(message: String) -> void:
	run_active = false
	run_status.text = message
	queue_redraw()


func _on_slider_changed(_value: float) -> void:
	refresh_lab()


func _on_viewport_changed() -> void:
	queue_redraw()


func _draw() -> void:
	var viewport_size := get_viewport_rect().size
	draw_rect(Rect2(Vector2.ZERO, viewport_size), Color("#111827"))

	var ground_destination := Rect2(455.0, GROUND_Y - 5.0, 425.0, 180.0)
	var ground_source := Rect2(65.0, 75.0, 190.0, 175.0)
	if tileset_texture:
		draw_texture_rect_region(tileset_texture, ground_destination, ground_source)
	else:
		draw_rect(ground_destination, Color("#8b5a2b"))

	var platform_destination := Rect2(965.0, GROUND_Y - 5.0, 250.0, 120.0)
	var platform_source := Rect2(1005.0, 115.0, 180.0, 135.0)
	if tileset_texture:
		draw_texture_rect_region(tileset_texture, platform_destination, platform_source)
	else:
		draw_rect(platform_destination, Color("#9ca3af"))

	var lava_destination := Rect2(865.0, GROUND_Y + 20.0, 115.0, 150.0)
	var lava_source := Rect2(70.0, 310.0, 180.0, 180.0)
	if tileset_texture:
		draw_texture_rect_region(tileset_texture, lava_destination, lava_source)
	else:
		draw_rect(lava_destination, Color("#ef3e23"))

	draw_line(
		Vector2(CLIFF_X, GROUND_Y - 150.0),
		Vector2(CLIFF_X, GROUND_Y + 80.0),
		Color("#f6c453"),
		2.0,
	)
	draw_string(
		ThemeDB.fallback_font,
		Vector2(CLIFF_X - 38.0, GROUND_Y - 165.0),
		"EDGE",
		HORIZONTAL_ALIGNMENT_LEFT,
		-1.0,
		16,
		Color("#f6c453"),
	)

	var runner_source := Rect2(95.0, 770.0, 170.0, 175.0)
	var runner_destination := Rect2(runner_position - Vector2(34.0, 48.0), Vector2(68.0, 70.0))
	if tileset_texture:
		draw_texture_rect_region(tileset_texture, runner_destination, runner_source)
	else:
		draw_rect(runner_destination, Color("#3aaed8"))

	if run_active:
		var distance_px := maxf(CLIFF_X - runner_position.x, 0.0)
		draw_line(
			Vector2(runner_position.x, GROUND_Y - 75.0),
			Vector2(CLIFF_X, GROUND_Y - 75.0),
			Color("#8bd3dd"),
			2.0,
		)
		draw_string(
			ThemeDB.fallback_font,
			Vector2(runner_position.x + 8.0, GROUND_Y - 86.0),
			"distance %.2f" % (distance_px / MAX_APPROACH_DISTANCE),
			HORIZONTAL_ALIGNMENT_LEFT,
			-1.0,
			15,
			Color.WHITE,
		)
