extends SceneTree

const Harness = preload("res://test/harness.gd")


func _init() -> void:
	call_deferred("_run")


func _run() -> void:
	var harness := Harness.new()
	var scene := load("res://unit_02_collector/unit_02_collector.tscn")
	if scene == null:
		harness.assert_true(false, "collector scene exists")
		quit(harness.failures)
		return

	var demo: Node2D = scene.instantiate()
	root.add_child(demo)
	demo._physics_process(1.0 / 60.0)

	var inputs: PackedFloat32Array = demo.get("last_inputs")
	var target: PackedFloat32Array = demo.get("last_target")
	var prediction: PackedFloat32Array = demo.get("last_prediction")
	var loss: float = demo.get("last_loss")

	harness.assert_true(inputs.size() == 3, "scene exposes three inputs")
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
	harness.assert_true(target_arrow.points.size() == 2, "target arrow updates")
	harness.assert_true(prediction_arrow.points.size() == 2, "prediction arrow updates")

	demo.queue_free()
	quit(harness.failures)
