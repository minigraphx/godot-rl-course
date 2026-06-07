# Cliff Jumper Unit 1 Design

## Goal

Replace the weak chase/retreat example in Neural Foundations 1 with a
time-critical cliff-jump trigger whose decision depends on both running speed
and remaining distance.

## Teaching Boundary

Unit 1 teaches the forward pass. It does not train the neuron automatically.
The learner acts as the optimizer by changing two weights and one bias, then
observing which situations fire the neuron.

The opening numerical example uses descriptive names before introducing
mathematical shorthand:

- input value;
- weight;
- contribution;
- bias;
- sum (later named \(z\));
- sigmoid output.

The phrase "frozen neuron" is replaced by "fixed-number neuron". The notation
\(x_1\), \(x_2\), \(w_1\), and \(w_2\) appears only after the learner has seen
the descriptive calculation. Omega is not used.

## Godot Example

The new `unit_01_jumper` scene has two modes:

1. **Lab:** sliders control normalized speed and remaining distance. Sliders
   also control `speed_weight`, `closeness_weight`, and `bias`. Every change
   immediately updates contributions, sum, sigmoid output, and the
   `WAIT`/`JUMP` decision.
2. **Test run:** the runner approaches the cliff at a randomly selected speed.
   The current neuron parameters trigger one jump. The scene reports too
   early, too late, or landed.

Distance is converted to `closeness = 1 - normalized_distance`, so both
positive weights have an intuitive reading:

- more speed pushes toward jumping earlier;
- more closeness to the edge pushes toward jumping now.

The action fires when `sigmoid(sum) > 0.5`.

## Guided Success Check

Three fixed laboratory cases prevent learners from solving only one situation:

| Case | Speed | Distance | Expected |
|---|---:|---:|---|
| Slow and far | 0.30 | 0.80 | WAIT |
| Fast and medium | 0.90 | 0.45 | JUMP |
| Slow and near | 0.30 | 0.10 | JUMP |

The interface displays how many of the three cases pass. The documentation
guides parameter changes in this order: distance/closeness weight, speed
weight, then bias.

## Documentation

English and German Unit 1 pages are updated together. The research
classification path remains because it gives a static decision-boundary view.
The comparison section maps the research boundary to the jumper trigger
boundary. Curriculum planning references are updated from chase/retreat to
cliff-jump timing.

## Verification

- Godot headless tests verify sigmoid output, the three guided cases, labels,
  and scene wiring.
- Python tests verify the same named fixed-number calculation.
- `mkdocs build --strict` verifies both language versions and navigation.
