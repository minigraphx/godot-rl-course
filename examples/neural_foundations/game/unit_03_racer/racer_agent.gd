extends "res://addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd"

@export var racer_path: NodePath


func _racer():
	return get_node(racer_path)


func _to_array(values: PackedFloat32Array) -> Array:
	var result: Array = []
	for value in values:
		result.append(float(value))
	return result


func _sync_from_racer() -> void:
	var racer = _racer()
	reward = racer.step_reward
	done = racer.done


func get_obs() -> Dictionary:
	return {"obs": _to_array(_racer().observation_values())}


func get_reward() -> float:
	_sync_from_racer()
	return reward


func get_action_space() -> Dictionary:
	return {
		"drive": {
			"size": 2,
			"action_type": "continuous",
			"squash": true,
		},
	}


func set_action(action) -> void:
	_racer().set_action(action)


func reset() -> void:
	super.reset()
	_racer().reset_racer()


func get_info() -> Dictionary:
	var racer = _racer()
	return {
		"checkpoint": racer.next_checkpoint_index,
		"episode_return": racer.episode_return,
		"steps": racer.steps,
	}
