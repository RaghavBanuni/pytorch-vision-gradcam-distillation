# Does the Model Look at the Object? — PyTorch, Grad-CAM, Distillation

A classifier that reaches 99% test accuracy and is still wrong about *why*. This repo
builds that failure on purpose, measures it, fixes it, and then compresses the fixed model
into a student ten times smaller.

The dataset is procedurally generated, so the ground truth includes something real image
datasets never give you: **the bounding box of the object**, and **a background cue that is
deliberately correlated with the label**. That turns explainability from a picture you squint
at into a number you can put in a table.

```
generator (shape + position + bbox + background cue)
        |
        +--> train / val / test        cue agrees with the label  (p = 0.95)
        +--> test_cue_broken           cue randomised
        +--> test_cue_inverted         cue points at the wrong class
        |                              (all three rendered from the SAME objects)
        v
train: SmallResNet, cosine schedule, label smoothing, early stopping
        |
        +--> accuracy, macro-F1, confusion matrix
        +--> ECE + reliability bins            (is the confidence honest?)
        +--> robustness gap across the 3 test sets
        +--> Grad-CAM pointing game            (is the evidence inside the object?)
        |
        v
distillation: teacher -> student (~10x fewer parameters), KL on softened logits
```

## The finding this repo is built to produce

Train on data where the background tint agrees with the label 95% of the time and the model
learns the tint, not the shape. It scores near-perfectly on a test set drawn from the same
distribution, then **collapses on `test_cue_inverted`** - and the Grad-CAM pointing-game score
shows why long before the robustness test does: the evidence sits outside the object box.

Two interventions are implemented and compared on the same three test sets:

- **cue-breaking augmentation** - randomised background/colour jitter, so the shortcut stops
  paying during training.
- **balanced training data** - regenerate with `--cue-strength 0.25`, which is the honest fix
  and the one you rarely get to make in production.

The point of reporting both is that they cost different things, and neither is free.

## Why the metrics are the ones here

**Accuracy on an i.i.d. test split cannot detect a shortcut.** The shortcut is *in* the test
split. That is exactly why a single held-out number is not evidence of generalisation, and
why the harness always evaluates three test sets.

**The three test regimes are a paired comparison.** They are rendered from the same latent
objects - same shapes, positions, rotations, colours, even the same noise draw - and differ
*only* in the background tint. Sampling them independently would confound the cue with
ordinary sampling noise; pairing them means the gap has exactly one cause.

**Grad-CAM is scored, not admired.** Every generated image knows where its object is, so the
class activation map gets two numbers: the *pointing game* hit-rate (does the peak of the map
fall inside the object box?) and *mask energy inside the box* as a share of total energy. Both
are reported against the share of the frame the box occupies, which is the score a uniform map
would get. Grad-CAM screenshots in a notebook prove nothing; a pointing-game score of 0.31
against a 0.14 baseline is an argument.

**Confidence is checked, not assumed.** A model can be accurate and still badly calibrated,
which matters the moment a downstream system thresholds on the probability. Expected
calibration error and per-bin reliability are reported next to accuracy.

**Distillation is judged on the whole trade, not the accuracy delta.** The student is compared
against the same student trained from scratch on hard labels only - because if plain training
matches the distilled version, the distillation added nothing but complexity. Parameter count,
and the accuracy gap on all three test sets, are reported together.

## Layout

```
visionlab/
  data.py        procedural images: shapes, boxes, background cue, paired test regimes
  augment.py     tensor augmentations, including the cue-breaking ones
  models.py      SmallResNet, TinyCNN student, torchvision backbone adapter, freezing
  train.py       Trainer: seeding, warmup + cosine, label smoothing, early stopping
  evaluate.py    accuracy, macro-F1, confusion matrix, ECE, robustness gaps
  gradcam.py     Grad-CAM via hooks + pointing game and mask-energy scoring
  distill.py     KD loss, student training, compression report
  cli.py         data / train / explain / robustness / distill
tests/           gradients, Grad-CAM properties, KD loss maths, split hygiene, metrics
```

## Running it

```bash
pip install -r requirements.txt

python -m visionlab.cli data                  # generate and describe the three regimes
python -m visionlab.cli train --epochs 8      # train and report all metrics
python -m visionlab.cli explain               # Grad-CAM pointing game, per class
python -m visionlab.cli robustness --balanced # shortcut gap, with and without the fixes
python -m visionlab.cli distill               # teacher -> student, against a scratch student

pytest -q
```

Everything runs on CPU in minutes at the default 48x48 resolution; `--device cuda` is honoured
when available. Pretrained torchvision weights are supported through
`--model resnet18 --pretrained`, which downloads them - the default is a self-contained
architecture so nothing here depends on a network call.

## Honest limitations

- Procedural shapes are not photographs. The shortcut is planted, which is what makes it
  measurable; in real data you rarely know it is there, and that is the actual difficulty.
- Grad-CAM is a low-resolution, last-conv-layer explanation and is known to be relatively
  insensitive to the class it is asked about in some architectures. The pointing-game score
  is a sanity check on where evidence lies, not proof of a causal mechanism.
- ECE depends on the binning scheme; it is reported with the bin count next to it for that
  reason, and the reliability table is printed rather than summarised away.
- The distillation results here are on a small model and a small dataset. The ranking of
  methods is what generalises; the absolute numbers are not a benchmark.

MIT licensed.
