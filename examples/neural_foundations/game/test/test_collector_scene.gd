extends SceneTree

const Harness = preload("res://test/harness.gd")


func _init() -> void:
	call_deferred("_run")


func _run() -> void:
	var harness := Harness.new()
	var scene := load("res://unit_02_collector/unit_02_collector.tscn")
	if scene == null:
		harness.assert_true(false, "collector scene exists")
		harness.finish(self)
		return

	var demo: Node2D = scene.instantiate()
	root.add_child(demo)
	demo._physics_process(1.0 / 60.0)

	var inputs: PackedFloat32Array = demo.get("last_inputs")
	var target: PackedFloat32Array = demo.get("last_target")
	var prediction: PackedFloat32Array = demo.get("last_prediction")
	var loss: float = demo.get("last_loss")

	harness.assert_true(inputs.size() == 4, "scene exposes four inputs")
	harness.assert_true(target.size() == 2, "scene exposes two target values")
	harness.assert_true(prediction.size() == 2, "scene exposes two predictions")
	harness.assert_true(loss >= 0.0, "scene exposes non-negative loss")

	var loss_label: Label = demo.get_node("Interface/Panel/Margin/Rows/Loss")
	var target_label: Label = demo.get_node("Interface/Panel/Margin/Rows/Target")
	var prediction_label: Label = demo.get_node("Interface/Panel/Margin/Rows/Prediction")
	harness.assert_true(loss_label.text.begins_with("Loss:"), "loss label updates")
	harness.assert_true(target_label.text.begins_with("Target:"), "target label updates")
	harness.assert_true(
		prediction_label.text.begins_with("Prediction:"),
		"prediction label updates"
	)

	var target_arrow: Line2D = demo.get_node("TargetArrow")
	var prediction_arrow: Line2D = demo.get_node("PredictionArrow")
	var gem_direction_line: Line2D = demo.get_node("GemDirectionLine")
	var hazard_direction_line: Line2D = demo.get_node("HazardDirectionLine")
	harness.assert_true(target_arrow.points.size() == 2, "target arrow updates")
	harness.assert_true(prediction_arrow.points.size() == 2, "prediction arrow updates")
	harness.assert_true(gem_direction_line.points.size() == 2, "gem direction line updates")
	harness.assert_true(
		hazard_direction_line.points.size() == 2,
		"hazard direction line updates"
	)

	var target_x_bar: ProgressBar = demo.get_node("Interface/Panel/Margin/Rows/TargetXBar")
	var prediction_x_bar: ProgressBar = demo.get_node("Interface/Panel/Margin/Rows/PredictionXBar")
	var counts_label: Label = demo.get_node("Interface/Panel/Margin/Rows/Counts")
	harness.assert_true(target_x_bar.value >= 0.0, "target x bar updates")
	harness.assert_true(prediction_x_bar.value >= 0.0, "prediction x bar updates")
	harness.assert_true(counts_label.text.begins_with("Steps:"), "counter label updates")

	demo.queue_free()
	harness.finish(self)
