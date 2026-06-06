extends RefCounted

var failures := 0

func assert_close(actual: float, expected: float, tolerance: float, label: String) -> void:
	if (
		not is_finite(actual)
		or not is_finite(expected)
		or not is_finite(tolerance)
		or tolerance < 0.0
		or absf(actual - expected) > tolerance
	):
		failures += 1
		push_error("%s: expected %f, got %f" % [label, expected, actual])

func assert_true(condition: bool, label: String) -> void:
	if not condition:
		failures += 1
		push_error(label)
