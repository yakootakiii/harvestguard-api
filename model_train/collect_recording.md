# Collecting a recording that can actually validate this model

Everything in `models/` is still trained on synthetic labels. The pipeline is now
honest about that — class bands are calibrated against a *measured* noise floor
rather than invented rates — but no real spoilage event has ever been recorded, so
the absolute gas rate that corresponds to real spoilage remains unknown.

This is the one gap that cannot be closed by better modelling. It needs data.

## What is wrong with `Banana D1.csv`

| Property | Current | Needed |
|---|---|---|
| Duration | 240 s (226 usable after warm-up) | 48–72 h |
| Spoilage event | None — a banana does not spoil in 4 minutes | At least one full progression |
| Labels | None | Timestamped observations |
| Environmental channels | None | Real temperature + humidity |
| Replicates | 1 | 3+ per condition |

At 226 usable seconds the train and test segments cannot even share the same drift
statistics, which is what currently holds Critical recall near 0.62.

## Protocol

**Duration.** 72 hours at minimum, logging continuously. Spoilage is a multi-hour
process; a window that resolves it must span hours, not the 60 s used now.

**Warm-up.** Power the MQ array for 10 minutes before the fruit goes in and discard
that period. `harvestguard_data.detect_warmup_end` finds it automatically, but only
if it is actually in the file. The current recording ramps at +1.10 units/s for its
first 15 s — an order of magnitude above anything the fruit does.

**Sampling.** 1 Hz is ample. Log every MQ channel, not just MQ3: MQ135 (ammonia) and
MQ4 (methane) are more direct spoilage proxies than MQ3 (alcohol), and they are
already wired. Retraining on one is a flag:

```bash
.venv/bin/python model_train/train.py --csv "model_train/Mango 72h.csv" --channel MQ135
```

**Environmental channels.** Log real SHT31 temperature and humidity in the same file,
at the same cadence. Both channels in the model are currently `np.random.uniform`
output; the model correctly ignores them, and it will keep ignoring them until real
readings exist.

**Labels.** Photograph the fruit every 2 hours and record a timestamped judgement —
`fresh` / `early spoilage` / `visibly spoiled`. Coarse labels tied to real times are
worth far more than fine-grained guesses. These become the real class boundaries,
replacing `GAS_BANDS_SIGMA` entirely.

**Conditions.** At least two: ambient (~28 °C) and refrigerated (~4 °C). The
refrigerated run matters twice over — it is a realistic transport condition *and* it
is currently 10σ outside the model's fitted domain, so the API refuses to answer for
it at all.

**Replicates.** Three runs per condition. One run cannot separate fruit-to-fruit
variation from sensor drift.

## File format

Match the existing schema so the loader works unchanged:

```
Ticks,MQ2,MQ3,MQ4,MQ5,MQ6,MQ7,MQ8,MQ9,MQ135,temperature,humidity
1648640854091,54,82,79,19,70,36,63,61,493,28.4,71.2
```

`Ticks` is Unix milliseconds. `load_real_gas` resamples to 1 Hz medians, so a jittery
logging interval is fine.

## What changes once the data exists

1. `GAS_BANDS_SIGMA` is deleted. Classes come from the observation log instead of
   from a synthetic generator.
2. `generate_window` is deleted. Windows are sliced from the real recording.
3. `WINDOW_SIZE` grows from 60 s to something matched to real spoilage kinetics —
   likely tens of minutes, which also drops the noise floor sharply.
4. `split_start_indices` splits by *run*, not by window position.
5. Held-out accuracy becomes a real measurement rather than a comparison against a
   theoretical ceiling.

Steps 1–3 are most of `harvestguard_data.py`. That is the intended outcome: the
synthetic generator is scaffolding, and a real recording is what removes it.
