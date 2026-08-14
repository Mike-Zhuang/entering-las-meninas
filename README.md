# Entering *Las Meninas*

### Art, geometry, and cognition through painting, compression, and reconstruction

> Every way of seeing has a bottleneck.

*Entering Las Meninas* is an art-and-computation project about what survives when an image moves between different ways of seeing. It begins with Diego Velázquez's *Las Meninas*, continues through a worn-cable ready-made and the drawing *The Tethered Viewer*, and culminates in a geometric autoencoder experiment, a transformed artwork, and a short parallax study.

The project does not ask a neural network to explain a painting. Instead, it uses a deliberately small model to make a narrower distinction visible: recurring geometry can survive compression even when handwriting, faces, color relationships, memory, and personal significance do not.

![Final artwork, Entering Las Meninas](outputs/artwork/entering-las-meninas-final.png)

## Project Arc

| Stage | Work | Role in the project |
| --- | --- | --- |
| Source | [*Las Meninas*](outputs/reference/las-meninas-reference.jpg) | Introduces the mirror, doorway, canvas, sightlines, and unstable position of the viewer. |
| Ready-made | [*Tether*](entering-las-meninas-10-minute-presentation/media/tether-overall.jpg) | Treats a worn cable and its traces of use as a self-portrait-like object. |
| Drawing | [*The Tethered Viewer*](OH_Programming/My_Own_Work.png) | Reorganizes the room around a red line connecting bodies, depth, and memory. |
| Experiment | [Geometric autoencoder](OH_Programming/README.md) | Tests which visual relationships pass through a constrained representation. |
| Final artwork | [*Entering Las Meninas*](outputs/artwork/entering-las-meninas-final.png) | Combines the painting's relational structure with frames, planes, nodes, and cognitive-map geometry. |
| Motion study | [Four-second parallax loop](entering-las-meninas-10-minute-presentation/media/parallax-loop.gif) | Uses depth-weighted motion to create the impression of stepping into the image. |

## What Survives the Bottleneck?

The central technical experiment uses a convolutional autoencoder with 27,983 trainable parameters. It is trained on 800 procedurally generated geometric images rather than on either artwork. The training vocabulary consists of perspective lines, nested rectangles, grids, arcs, radial connections, and occasional red polylines.

```text
3 × 128 × 128 input
        ↓ Conv + ReLU
12 × 64 × 64
        ↓ Conv + ReLU
24 × 32 × 32
        ↓ Conv + ReLU
32 × 16 × 16 bottleneck
        ↓ Transposed convolutions
3 × 128 × 128 reconstruction
```

| Setting | Value |
| --- | ---: |
| Synthetic training images | 800 |
| Training epochs | 18 |
| Batch size | 32 |
| Learning rate | 0.001 |
| Trainable parameters | 27,983 |
| Loss function | Mean squared error |
| Random seed | 139 |

![Actual layer-by-layer forward pass](OH_Programming/outputs/08-actual-forward-pass.png)

The reconstruction preserves the broad room frame, central opening, strong diagonals, and approximate figure positions. It is much less successful with handwriting, faces, fine contours, and the exact shape of the red tether. The model reconstructs *The Tethered Viewer* with an MSE of `0.003884`, compared with `0.036573` for *Las Meninas*. This gap primarily reflects the synthetic training distribution: the linear drawing is closer to the model's geometric vocabulary than the dark, chromatically complex painting.

The result is therefore not a measure of artistic quality or understanding. The model recovers familiar visual structure while exposing the distance between representing a pattern and knowing why it matters.

> The encoder compresses recurring visual patterns, not the personal significance I attach to them.

## Selected Outputs

- [Final presentation](entering-las-meninas-10-minute-presentation.pptx) — a ten-slide overview with the full English script in the speaker notes.
- [Autoencoder report](OH_Programming/submission/README-en.md) — complete method, figures, results, interpretation, and limitations.
- [Layer-output notebook](OH_Programming/submission/01-all-layer-outputs.ipynb) — the selected image traced through all twelve leaf layers.
- [Reconstruction notebook](OH_Programming/submission/02-autoencoder-reconstruction.ipynb) — encoder, bottleneck, decoder, and reconstruction analysis.
- [Final artwork](outputs/artwork/entering-las-meninas-final.png) and [parallax video](outputs/video/entering-las-meninas-parallax-h264.mp4).

## Repository Structure

```text
.
├── OH_Programming/        # Geometric autoencoder, checkpoint, notebooks, and figures
├── outputs/artwork/       # Final static artwork
├── outputs/video/         # Parallax study
├── style_transfer/        # Supporting VGG-19 style-transfer experiments
├── src/                   # Extended CNN and geometry analysis code
├── scripts/               # Pipeline, validation, and release utilities
├── tests/                 # Automated tests
├── report/                # Extended written analysis
└── presentation/          # Earlier extended-study presentation builder
```

The autoencoder project is the central experimental thread. The VGG-19 style-transfer work and the larger geometry/topology pipeline remain in the repository as supporting and extended studies. Their full context is available in [Extended CNN Study](EXTENDED-CNN-STUDY.md), while [Repository Guide](REPOSITORY-GUIDE.md) documents why path-dependent directories remain in place.

## Reproducing the Autoencoder Experiment

Run the experiment from `OH_Programming/` so that its relative paths continue to resolve correctly:

```bash
cd OH_Programming
python -m pip install -r requirements.txt
python autoencoder-experiment.py
python network-visualization.py
python export-individual-images.py
```

The scripts use random seed `139`, prefer Apple MPS when available, and fall back to the CPU. Graphviz is required for the automatically traced architecture export.

The submitted notebooks do not retrain the model. They load the included checkpoint and run inference on the CPU for portability.

## Limitations

- The autoencoder is trained on synthetic geometry, not on *Las Meninas* or *The Tethered Viewer*.
- Downsampling to 128 × 128 removes information before the image reaches the network.
- Feature maps and channel averages are numerical activations, not semantic concepts or attention maps.
- Reconstruction error describes this model's visual compression; it does not measure meaning, authorship, or artistic quality.
- The parallax animation is a two-dimensional, depth-weighted remapping rather than a three-dimensional reconstruction.

## License and Sources

Project-authored code is released under the [MIT License](LICENSE). Image rights, model weights, course materials, generated assets, and third-party dependencies are documented separately in [Third-Party Notices](THIRD-PARTY-NOTICES.md) and [Sources](report/sources.md).
