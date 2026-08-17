# Code.md — house style for the phase-conditioned DVF generator project

This is the house style, applied specifically to this project: a U-Net+FiLM
generator producing DVFs from a single reference CT + phase pair, trained
adversarially against a 3D PatchGAN discriminator, supervised by Elastix
DVFs. The conventions below are the same ones the existing training script
follows — this doc just shows what they look like on *this* architecture,
not as a generic checklist.

## 1. Project layout

```
project/
  train_dvf_gan.py            # single entry point, generator + discriminator together
  utilities/
    generator.py               # UNetFiLM model class
    discriminator.py           # PatchDiscriminator model class
    film.py                    # FiLM trunk + per-scale heads, shared by generator
    svf.py                     # scaling-and-squaring integration layer
    losses.py                  # DVFLoss, ImageSimilarityLoss, SmoothnessLoss, GANLoss
  data/xcat/train              # per-phase CT .npy volumes + precomputed Elastix DVFs
  weights/                     # generator + discriminator checkpoints
  plots/                       # loss curves
```

- One script trains both networks — not two separate scripts — because
  they share a training loop and a single "does validation loss improve"
  checkpoint decision. Keep `network_a.py`'s one-script-per-experiment habit,
  just with two models inside it.
- `film.py` is its own file, not folded into `generator.py`, because the
  FiLM trunk + heads are conceptually a shared conditioning module the
  generator calls into at 5 points — giving it its own file keeps
  `generator.py` readable as "the U-Net" rather than "the U-Net plus a
  conditioning MLP interleaved through it."
- `svf.py` is separate for the same reason: it's a fixed, non-trainable
  integration operation, not part of the U-Net's learned weights — keeping
  it out of `generator.py` makes clear it's a deterministic post-processing
  step, not a layer with parameters.

## 2. Dataset class — `PhasePairDataset`

Follows the existing `__getitem__`-as-labeled-steps pattern, but the fields
change: instead of source/target *projections* it's source/target *phase
indices* plus the precomputed Elastix DVF for that directed pair.

```python
class PhasePairDataset(Dataset):
    def __init__(self, im_dir=None, im_size=None):
        self.im_dir = im_dir
        self.im_size = im_size

    def __len__(self):
        return len([n for n in os.listdir(self.im_dir) if n.endswith('_pair.npy')])

    def __getitem__(self, idx):
        # Find reference CT
        ...
        # Find target phase index
        ...
        # Find reference phase index
        ...
        # Find target DVF (Elastix ground truth for this phase pair)
        ...

        data = {'reference_ct': torch.from_numpy(reference_ct),
                'ref_phase': torch.tensor(ref_phase, dtype=torch.long),
                'target_phase': torch.tensor(target_phase, dtype=torch.long),
                'target_dvf': torch.from_numpy(target_dvf)}
        return data
```

- Normalize the CT inline with the same min-max formula as the sample —
  don't introduce a separate normalization utility for this project.
- Phase indices are `torch.long` (they index the FiLM embedding table),
  not floats — this is the one dtype choice specific to this project's
  data that differs from the sample's all-float fields.
- Preallocate the DVF array with explicit channel ordering, same as the
  sample's `target_flow`, so channel 0/1/2 = x/y/z is visible at the call
  site:

```python
target_dvf = np.zeros((3, self.im_size, self.im_size, self.im_size), dtype=np.float32)
```

## 3. Model construction

Two models, both built with short explicit constructor args — no config
object, matching `network_a.model(im_size, int_steps=10)`:

```python
generator = UNetFiLM(im_size=128, n_phases=10, int_steps=6)
discriminator = PatchDiscriminator(in_channels=4)
generator.to(device)
discriminator.to(device)
```

- `n_phases=10` and `int_steps=6` are named args on the constructor, not
  buried in a dict — same reasoning as the sample's `int_steps=10`.
- `PatchDiscriminator(in_channels=4)` spells out the 4 = ref_CT(1) + DVF(3)
  concatenation at the call site, so the conditioning choice is visible
  without reading `discriminator.py`.

## 4. Losses

Four loss objects, instantiated once before the loop, each with a `.loss(...)`
method taking raw tensors — same shape as `flow_mask = losses.flow_mask()`:

```python
dvf_loss = losses.DVFLoss()
img_loss = losses.ImageSimilarityLoss()
smooth_loss = losses.SmoothnessLoss()
gan_loss = losses.GANLoss()

lambda_img = 0.5
lambda_smooth = 0.1
lambda_adv = 0.05
```

- Loss weight constants (`lambda_img`, `lambda_smooth`, `lambda_adv`) are
  named module-level constants near the top of the script, same as the
  sample's `lr = 1e-5` — not passed via CLI args.

## 5. Optimizers

Two optimizers, not one — this is the one real structural departure from
the sample, since the sample only trains a single network:

```python
optimizer_g = optim.Adam(generator.parameters(), lr=1e-4)
optimizer_d = optim.Adam(discriminator.parameters(), lr=1e-4)
```

Keep them as two clearly-named locals, not a dict or list — every
`.zero_grad()` / `.step()` call site in the loop should read unambiguously
which network it belongs to.

## 6. Training loop

Same flat, no-trainer-class structure as the sample, extended to the two
networks. Order matters and should stay explicit in the code, not hidden
in a helper method:

```python
for epoch in range(1, epoch_num + 1):
    generator.train()
    discriminator.train()
    train_loss_g, train_loss_d = 0.0, 0.0

    for i, data in enumerate(trainloader, 0):
        reference_ct, ref_phase, target_phase, target_dvf = (
            data['reference_ct'].to(device),
            data['ref_phase'].to(device),
            data['target_phase'].to(device),
            data['target_dvf'].to(device),
        )

        # --- discriminator step ---
        optimizer_d.zero_grad()
        fake_dvf = generator(reference_ct, ref_phase, target_phase)
        loss_d = gan_loss.discriminator_loss(
            discriminator, reference_ct, target_dvf, fake_dvf.detach()
        )
        loss_d.backward()
        optimizer_d.step()
        train_loss_d += loss_d.item()

        # --- generator step ---
        optimizer_g.zero_grad()
        loss_g = (
            dvf_loss.loss(target_dvf, fake_dvf)
            + lambda_img * img_loss.loss(reference_ct, target_dvf_image, fake_dvf)
            + lambda_smooth * smooth_loss.loss(fake_dvf)
            + lambda_adv * gan_loss.generator_loss(discriminator, reference_ct, fake_dvf)
        )
        loss_g.backward()
        optimizer_g.step()
        train_loss_g += loss_g.item()
```

- Two accumulators (`train_loss_g`, `train_loss_d`), reported as two
  numbers in the epoch print line — don't collapse them into one combined
  loss for logging, since a healthy GAN run is judged by watching both
  curves relative to each other, not a single number trending down.
- Keep the discriminator step before the generator step in the loop body,
  and keep `fake_dvf.detach()` explicit at the call site (not hidden
  inside `gan_loss.discriminator_loss`) — this is the one place in GAN
  code where an easy-to-miss bug (forgetting `.detach()`) silently breaks
  training, so it stays visible rather than wrapped away.
- Optional discriminator-update throttling (1 D-step per 2–3 G-steps, per
  earlier discussion) — if added, gate it with a plain
  `if i % d_update_freq == 0:` around the discriminator block, not a
  separate scheduler class.

## 7. Validation

Mirror the training loop's unpack order exactly (same habit as the
sample), under `model.eval()` + `torch.no_grad()` for both networks:

```python
generator.eval()
discriminator.eval()
with torch.no_grad():
    for j, valdata in enumerate(valloader, 0):
        ...
```

Report validation loss using `loss_g` only (DVF + image + smoothness
terms, no adversarial term) — the discriminator's adversarial signal isn't
a meaningful "is the model getting better" metric on its own, so don't let
it drive the checkpoint decision.

## 8. Checkpointing

Save both networks' state dicts when validation `loss_g` improves —
same best-only, overwrite-in-place pattern as the sample, just two files
instead of one:

```python
if val_loss_g < min_val_loss:
    torch.save(generator.state_dict(), 'weights/' + filename + '_generator.pth')
    torch.save(discriminator.state_dict(), 'weights/' + filename + '_discriminator.pth')
    min_val_loss = val_loss_g
```

## 9. Logging and plots

Same per-epoch full-redraw plotting habit as the sample, extended to plot
generator and discriminator train curves as separate lines (not summed):

```python
print('Epoch: %d | G loss: %.4f | D loss: %.4f | val G loss: %.4f | total time: %d hours %d minutes' %
      (epoch, train_loss_g / len(trainset), train_loss_d / len(trainset), val_loss_g / len(valset), hours, minutes))
```

Plot `train_losses_g`, `train_losses_d`, `val_losses_g` as three lines on
one figure, legend labeled accordingly — re-plot and re-save the whole
curve every epoch, same as the sample, so the plot on disk is always
complete even if training is interrupted mid-run.

## 10. Note carried over from the sample review

Make sure `generator.train()` and `discriminator.train()` are called at
the top of every training epoch, not just once before the loop — this was
a latent bug in the original single-network script (`model.eval()` was
called for validation but `model.train()` was never called again
afterward). With two networks and InstanceNorm layers in both, this bug
would silently degrade both models, not just one, so it's worth a
deliberate line at the top of the epoch loop rather than assuming it
carries over from before the loop started.