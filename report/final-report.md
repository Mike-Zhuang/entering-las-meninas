# Entering *Las Meninas*: How a Painting Changed the Way I Look at Art

## Abstract

This project asks what a convolutional neural network can separate in Diego Velázquez's *Las Meninas*, and what remains outside that separation. I treated the CNN as an incomplete viewer rather than an automatic art critic. Frozen VGG-19 features measured Gram-matrix style and spatially preserved composition; HED boundaries, Hough segments, and RANSAC tested geometry. At 100%, the non-neural style transformation produced a style distance of 0.3025 but a spatial cosine distance of 0.0333; the geometry transformation produced 0.1267 and 0.3525. Mirror deletion broke one of five manually defined relations while remaining extremely close in VGG space (0.0219 style, 0.0025 spatial). A matched edit preserved all five relations but produced a larger style distance (0.0249). This is the central result: CNN distance registers an edit without representing why the mirror matters. The geometry pipeline also failed to discover one stable vanishing point because estimates changed with the region of interest. I used these results to make *Entering Las Meninas*, which preserves the relation structure while replacing the royal reflection with a contemporary viewer. No human-participant study was conducted; I report model behavior and project-based interpretation, not a general claim about perception.

## 1. Introduction: from image surface to viewing position

At the beginning of this project, I could describe *Las Meninas* as a famous group portrait with an unusual mirror. Building the analysis made that description feel inadequate. The painting did not become more interesting because a model recognized more objects. It became more interesting when I had to specify how the objects connect: the painter faces an implied position outside the frame; the large canvas blocks part of the room; José Nieto occupies the bright rear doorway; the Infanta is surrounded by attendants; and the small mirror connects the depicted room to people who are not bodily present in it.

The Prado identifies the work as Velázquez's 1656 oil painting, 320.3 × 279.1 cm, inventory P001174. It identifies Philip IV and Mariana of Austria in the rear mirror and describes the work's space as a combination of scientific perspective, aerial perspective, and multiple sources of light. The museum also emphasizes that the mirror turns the painting into a reflection on seeing, representation, and the viewer's own role ([Museo Nacional del Prado](https://www.museodelprado.es/en/the-collection/art-work/the-family-of-felipe-iv-or-las-meninas/9fdc7800-9ade-48b0-ab8b-edee94ea877f)). That account shaped my question. If the painting's force depends on the viewer building a model that extends beyond the picture plane, then extracting texture or edges cannot be the whole analysis.

The high-resolution source was a public-domain faithful reproduction distributed through Wikimedia Commons. I preserved the original file separately and used a 1,779 × 2,048 working proxy for reproducible processing. The repository records the source and checksum instead of treating an internet image as an unexplained input ([Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Las_Meninas,_by_Diego_Vel%C3%A1zquez,_from_Prado_in_Google_Earth.jpg)).

![Public-domain working reference](../outputs/reference/las-meninas-reference.jpg)

My guiding research question became: **How far can a CNN separate style, geometry, and relational topology in *Las Meninas*, and what does that separation reveal about the difference between detecting visual features and understanding a painting's viewing structure?** “Relational topology” is an operational term in this project. It means a small graph of qualitative relations that survive exact changes in color or distance. It is not a claim about mathematical topology in full, and it is not a measurement of the brain.

## 2. Ready-made: *The Missing Viewer*

My ready-made proposal is an ordinary rectangular mirror titled *The Missing Viewer*. It would stand on the implied viewing axis in front of a reproduction or projection of *Las Meninas*. Its function comes from alignment: a present-day viewer looking into it would occupy the off-canvas position toward which the painter, mirror, and several figures direct attention. The ready-made makes the relation graph physical by connecting a real body to a position the painting constructs but does not directly show.

![AI-generated installation concept mockup, not documentary photography](../assets/installation/the-missing-viewer-mockup.png)

The image above is a concept mockup, not evidence that the installation was built. It tests scale, alignment, and gallery plausibility; no screen or sensor is required. The proposal also set a standard for the experiment: removing a few pixels from the painted mirror might barely change CNN distance while destroying the connection that makes the ready-made possible.

## 3. Experimental framework

I separated the project into three layers:

1. **Style**: color, texture, local mark, and CNN channel co-activation.
2. **Geometry**: coordinate, scale, angle, perspective, depth, and spatial layout.
3. **Relational topology**: qualitative connections such as reflects, contains, occludes, faces, and surrounds.

The design first used a 2 × 2 matrix: style retained or altered crossed with geometry retained or altered. The working proxy was G0S0. A deterministic color/tone/detail transform and a separate neural-style branch provided G0S1. A perspective homography provided G1S0. Using geometry-100 as the content input for neural style transfer provided G1S1. Five intensities (0, 25, 50, 75, and 100%) made each main transformation a trajectory rather than one before/after example.

![Deterministic screening version of the 2 × 2 transformation matrix](../outputs/figures/transformation-matrix.png)

The geometry transformation used a declared warp pivot at normalized coordinate (0.56, 0.53), convergence 0.14, and depth expansion 0.08. These are intervention parameters, not a discovered historical vanishing point. The code crops the safe interior after the homography and resizes it back to the original canvas, reducing reflected-border artifacts while keeping output dimensions constant. The deterministic style baseline changes tone, chroma, detail, and light quantization without moving pixel coordinates. It is labeled non-neural throughout the repository.

The topology experiment was separate from the 2 × 2 matrix. I defined five relations before interpreting the results: the rear mirror connects the room to the outside viewing position; the painter faces that position; the rear doorway contains José Nieto; the large canvas occludes the left side of the room; and the Infanta is surrounded by attendants. The mirror-deletion condition replaces only the mirror interior with locally matched wall texture. The matched control modifies the same box with a feathered color change scaled to the same root-mean-square pixel energy while retaining the reflected figures. A sham condition applies a subpixel shift and inverse shift to expose effects caused by resampling alone. The relation table is a manual visual audit, not CNN output.

## 4. CNN style and spatial method

I used the convolutional feature backbone of VGG-19 with Torchvision's ImageNet-1K pretrained weights. VGG's stacked 3 × 3 convolutions make it straightforward to read several depths of one fixed model ([Simonyan and Zisserman, 2015](https://arxiv.org/abs/1409.1556); [Torchvision VGG-19 documentation](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vgg19)). I did not train a classifier on a few copies of one painting. Instead, VGG was frozen and used as a measurement instrument.

All comparison images were aspect-preservingly letterboxed to 512 × 512 using ImageNet mean color, with the painted content occupying 445 × 512 pixels. This avoided cutting away the canvas or room. Style descriptors came from `relu1_1`, `relu2_1`, `relu3_1`, `relu4_1`, and `relu5_1`. For feature tensor $F_l \in \mathbb{R}^{C_l \times H_lW_l}$, I calculated

\[
G_l = \frac{F_lF_l^\top}{C_lH_lW_l}.
\]

This follows the representation used in neural style transfer, where channel correlations summarize recurring features while weakening their exact positions ([Gatys, Ecker, and Bethge, 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html); [PyTorch neural style tutorial](https://docs.pytorch.org/tutorials/advanced/neural_style_tutorial.html)). I report the mean multi-layer relative Frobenius distance between Gram matrices as **style distance**. It is a CNN-defined texture and co-activation statistic, not complete art-historical style.

Spatial descriptors retained the two-dimensional feature arrangement at `relu3_1`, `relu4_2`, and `relu5_1`, pooled to 32 × 32. I use mean cosine distance across corresponding cells as **spatial distance** and report relative RMS as a complementary magnitude measure. Both remain style-sensitive, so neither is a pure geometry score. The exact working proxy was the reference; every 0% condition therefore measures 0, eliminating an encoding confound.

For neural style transfer, I optimized pixels with frozen VGG-19 features for 500 Adam steps at a 512-pixel long side. The content layer was `relu4_2`; the five style layers matched the Gram analysis; content weight was 1, style weight was $10^6$, total-variation weight was $10^{-4}$, learning rate was 0.02, and the three style-strength multipliers were 0.25, 0.5, and 1.0. The content aspect ratio was preserved, the random seed was 139, and all strengths shared the same initial state. The style donor was an original project-generated abstract image of nested frames, construction lines, and a luminous mirror-like rectangle. Thus the style experiment did not copy a named modern artist.

## 5. Geometry method: HED, Hough, and RANSAC

The geometric pipeline separated learned boundary extraction from classical geometry. HED, a deeply supervised fully convolutional edge network, produced a boundary-probability map ([Xie and Tu, 2015](https://openaccess.thecvf.com/content_iccv_2015/html/Xie_Holistically-Nested_Edge_Detection_ICCV_2015_paper.html)). After thresholding and thinning, probabilistic Hough detection produced candidate line segments; RANSAC selected a dominant line-intersection candidate. A Canny-to-Hough pipeline served as a non-neural control ([OpenCV Hough Line Transform](https://docs.opencv.org/master/d9/db0/tutorial_hough_lines.html)).

The 1,779 × 2,048 proxy was resized to 1,112 × 1,280 for this analysis. The pipeline detected 334 HED-based segments and retained up to 400 Canny-based segments. RANSAC ran 5,000 iterations with a 0.012 image-scale inlier threshold. I compared an unconstrained full-frame run with two explicitly recorded regions of interest (ROIs): a smaller region motivated by an art-historical expectation and a rear-door region. Each ROI is a prior supplied by the analyst; it is not an automatic discovery.

## 6. Results

### 6.1 Style and geometry were separable, but not pure

The controlled trajectories moved in different directions in VGG feature space. The deterministic style trajectory produced style distances of 0.0795, 0.1569, 0.2300, and 0.3025 at 25, 50, 75, and 100%, while spatial distance rose only from 0.0059 to 0.0333. The geometry trajectory produced style distances of 0.0885, 0.1043, 0.1185, and 0.1267, but spatial distance rose from 0.0971 to 0.3525. At 100%, the geometry intervention's spatial distance was 10.6 times the style baseline's, while the style baseline's Gram distance was 2.4 times the geometry intervention's. This is not perfect orthogonality—warping changes local feature statistics, and styling affects spatial feature maps—but it is a meaningful empirical separation.

The neural-style sequence was more aggressive. Its Gram distances were 0.5030, 0.5303, and 0.5549 at strengths 0.25, 0.5, and 1.0; spatial distances were 0.1045, 0.1148, and 0.1298. The optimization itself showed the expected trade-off: final content loss rose from 0.2298 to 0.5764 as strength increased, while the raw style loss fell from (1.437 \times 10^{-6}) to (5.931 \times 10^{-7}). Because style strength multiplies the style term, weighted style loss increased from 0.3593 to 0.5931.

![Neural-style sequence at strengths 0, 0.25, 0.5, and 1.0](../outputs/figures/neural-style-sequence.png)

The neural 2 × 2 endpoints make the separation especially clear:

| Condition | Style distance | Spatial cosine | Spatial relative RMS |
| --- | ---: | ---: | ---: |
| G0S0: working proxy | 0.0000 | 0.0000 | 0.0000 |
| G0S1: neural style, strength 1.0 | 0.5549 | 0.1298 | 0.4822 |
| G1S0: geometry, 100% | 0.1267 | 0.3525 | 0.9189 |
| G1S1: geometry-100 content + neural style 1.0 | 0.5676 | 0.4004 | 0.9083 |

The combined neural condition exceeded either single intervention in cosine distance while retaining strong Gram displacement. The deterministic combined endpoint likewise measured 0.3210 style, 0.3631 spatial cosine, and 0.9807 spatial RMS. The different rankings of cosine and RMS reinforce the need to name the metric rather than report one unexplained “similarity” number.

### 6.2 Breaking a relation barely moved VGG

The manual topology audit gave the original, matched control, geometry-100, neural-style-1, and combined-neural-geometry conditions 5/5 preserved relations. Mirror deletion scored 4/5 because it alone removed the connection between the painted room and the implied outside viewer.

![Original, matched control, mirror deletion, and sham-warp conditions](../outputs/figures/topology-controls.png)

Yet the mirror deletion's VGG distances were tiny: 0.0219 style and 0.0025 spatial. More importantly, the matched control preserved all five relations but had a *larger* style distance of 0.0249. Deletion did produce a 66% larger spatial distance than the matched edit (0.0025 versus 0.0015), but the sham resampling control was larger still at 0.0036. These measurements do not show a CNN encoding the missing reflective relation. They show sensitivity to the local appearance and sampling consequences of the edits. In this controlled case, the manually identified semantic loss is almost orthogonal to the model's distance ranking.

This result changed how I understood the phrase “extract the topology.” A CNN can provide candidate nodes and local features, but the relation *mirror connects depicted room to an unseen position* was not present as an automatically recovered edge. I had to state, alter, and audit that edge explicitly.

### 6.3 The vanishing-point estimate was ROI-dependent

The unconstrained full-frame run produced a HED/RANSAC candidate at normalized (0.720, 0.697), with 57 inlier lines and a weighted inlier ratio of 0.1728. Canny instead produced (0.217, 0.619), placing the two candidates 0.3348 normalized image diagonals apart. The run's medium evidence grade describes the HED candidate's internal line consensus; it does not indicate cross-method agreement. This full-frame disagreement is already evidence that the estimate is unstable.

![HED boundaries, Hough segments, and the unconstrained full-frame RANSAC candidate](../outputs/geometry-analysis/original/cnn-lines-vanishing-point.png)

With the art-history-motivated ROI, HED moved to (0.529, 0.439), had only 19 inliers, and a weighted ratio of 0.0552; Canny estimated (0.518, 0.472), and the run was graded low. In the rear-door ROI, HED moved again to (0.626, 0.612), with a 0.1063 ratio; Canny estimated (0.634, 0.600), again low grade. Thus the two edge methods disagree substantially on the unconstrained frame but happen to approach one another inside each analyst-selected window, while the windows themselves imply substantially different points. Neither internal line consensus nor conditional HED–Canny agreement validates a unique art-historical vanishing point.

I therefore report this pipeline as an instructive failure of automatic discovery. HED extracted useful multi-scale boundaries, but Hough and RANSAC favored whatever straight-line family dominated the ROI. The result depends on painting contours, frames, doorway edges, brush boundaries, and an analyst-defined crop. It does not prove that Velázquez used one detected point, and it does not replace the Prado's broader account of linear perspective, aerial perspective, and light.

## 7. Picasso as a local external comparison

I also analyzed a locally held reproduction corresponding to Pablo Picasso's 1957 *Las Meninas*. The Museu Picasso records that large canvas as 194 × 260 cm, inventory MPB 70.433, and explains that Picasso changed the original portrait format to landscape, enlarged the painter, flattened and reorganized the space, and retained the cast and the figure in the doorway ([Museu Picasso Barcelona](https://museupicassobcn.cat/en/collection/artwork/las-meninas-9)). It is therefore a better conceptual stress test than a random filter: appearance and geometry change dramatically while many role relations remain legible.

Relative to my reference, the local Picasso reproduction produced the largest distances in the set: 1.4196 for Gram style and 0.7139 for spatial features. These numbers are descriptive, not a recognition score. The input also differs in aspect ratio, reproduction conditions, and rights status. I did not collect human recognition judgments, so I cannot claim that people generalize better than VGG from these values. I can only say that the museum's interpretation establishes the historical relationship while this particular VGG representation places the images far apart. Because the local reproduction's authorization chain was not established and the official page reserves reproduction rights, the image was used locally but is not redistributed in the public repository.

## 8. Final artwork: *Entering Las Meninas*

The final artwork recombines what the analysis separated. It preserves the painter and large canvas, the Infanta and attendants, the dog, the rear doorway, the portrait orientation, and the room's broad depth hierarchy. It makes perspective lines and node-like connections visible as charcoal construction marks. Most importantly, it does not delete the mirror connection. It changes the identity inside it from the royal couple to a subtle contemporary viewer silhouette.

![Final AI-assisted artwork, *Entering Las Meninas*](../outputs/artwork/entering-las-meninas-final.png)

The manual audit therefore scores the final work 5/5 for relation preservation, with a disclosed change in mirror identity. Its VGG style distance is 0.4614 and its spatial distance is 0.3638: far from the source in both representations, yet intentionally continuous in the five-relation graph. I do not present this image as a direct output of the quantitative CNN pipeline. It is a separate AI-assisted generative edit informed by the experiments. The complete prompt and input description are recorded beside the file. This distinction matters: measurement helped me decide what to preserve, while the final generative act made an interpretive choice that VGG did not make for me.

## 9. Limitations and evidence boundaries

This is a single-work computational case study, not a general theory of art perception. VGG-19 and HED were trained primarily on natural-image datasets, not on Baroque painting. Gram matrices discard exact position by design, and the spatial descriptor still responds to color and texture. The transformations are controlled but not perfectly factorized: homography introduces crop and interpolation, and neural style optimization can alter boundaries. The 32 × 32 pooled grid also limits fine localization.

The topology score contains only five binary, manually audited relations. It has no second rater and no inter-rater reliability estimate. It clarifies the project's assumptions; it does not convert interpretation into objective ground truth. Likewise, the HED/Hough/RANSAC output has no manually validated vanishing-point target. The severe ROI dependence means the result must remain a candidate, not a discovery.

No human participants were recruited, no survey was administered, and no behavioral or neural data were collected. Consequently, I do not claim that any transformation increased or reduced viewers' feeling of “entering” the painting, that humans and CNNs were experimentally shown to disagree, or that VGG features are a human cognitive map. Tolman's cognitive-map concept concerns organized spatial learning, and O'Keefe and Dostrovsky's evidence concerns hippocampal unit activity; this project borrows the organizing metaphor only ([Tolman, 1948](https://doi.org/10.1037/h0061626); [O'Keefe and Dostrovsky, 1971](https://doi.org/10.1016/0006-8993(71)90358-1)).

## 10. What I learned

The most important result was not that the CNN “solved” *Las Meninas*. It was that the model made its own incompleteness measurable. The style and geometry trajectories showed that a single architecture can yield usefully different summaries when I preserve or discard spatial position. HED showed that a learned boundary map can support geometrical analysis without itself understanding perspective. The mirror control showed the harder boundary: an edit can be semantically decisive for my reading while remaining almost negligible—and ambiguously ranked—in VGG distance.

That process changed the way I look at art because it forced me to stop treating composition as a container for recognizable things. I now look for relations that make a work continue outside its frame: who faces whom, what blocks what, which aperture contains another space, and where the work places my body. In *Las Meninas*, the mirror is small in pixel area but large in the structure of viewing. My final transformation preserves that connection because the experiment convinced me that fidelity to a painting can reside in a relation rather than in a surface.

The CNN was valuable precisely because it was not a substitute viewer. It separated statistics, exposed confounds, and gave me resistance against which to define my interpretation more clearly. “Entering” the painting ultimately did not mean simulating depth alone. It meant recognizing that the painting had already assigned the viewer a position—and deciding, in the final artwork, to make that position visible.

## AI and asset disclosure

The project used frozen VGG-19 features, Gatys-style pixel optimization, a pretrained HED model, OpenCV geometry, and deterministic image transformations. Code, parameters, checksums, CSV metrics, JSON manifests, and generated outputs are included for audit. The style-reference image, installation mockup, and final artwork were created with the Codex image-generation tool from project-authored prompts. The tool did not return a model version or random seed, so those three raster outputs are not guaranteed to be pixel-reproducible; their exact prompts and SHA-256 checksums are recorded in `assets/README.md` and `outputs/artwork/README.md`. The mockup is labeled as generated, and the final work is labeled AI-assisted. Velázquez's source reproduction is public domain. The Picasso reproduction and unsourced auxiliary classroom images remain local and are excluded from public redistribution.

## References

Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). Image style transfer using convolutional neural networks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2414–2423. [CVPR paper](https://openaccess.thecvf.com/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html).

Museo Nacional del Prado. (n.d.). *The Family of Felipe IV, or Las Meninas*, inventory P001174. Accessed July 17, 2026. [Collection record](https://www.museodelprado.es/en/the-collection/art-work/the-family-of-felipe-iv-or-las-meninas/9fdc7800-9ade-48b0-ab8b-edee94ea877f).

Museu Picasso Barcelona. (n.d.). *Las Meninas*, inventory MPB 70.433. Accessed July 17, 2026. [Collection record](https://museupicassobcn.cat/en/collection/artwork/las-meninas-9).

O'Keefe, J., & Dostrovsky, J. (1971). The hippocampus as a spatial map: Preliminary evidence from unit activity in the freely-moving rat. *Brain Research, 34*(1), 171–175. [DOI](https://doi.org/10.1016/0006-8993(71)90358-1).

OpenCV. (n.d.). *Hough Line Transform*. Accessed July 17, 2026. [Official documentation](https://docs.opencv.org/master/d9/db0/tutorial_hough_lines.html).

PyTorch. (n.d.). *Neural Transfer Using PyTorch*. Accessed July 17, 2026. [Official tutorial](https://docs.pytorch.org/tutorials/advanced/neural_style_tutorial.html).

Simonyan, K., & Zisserman, A. (2015). Very deep convolutional networks for large-scale image recognition. *ICLR*. [arXiv](https://arxiv.org/abs/1409.1556).

Tolman, E. C. (1948). Cognitive maps in rats and men. *Psychological Review, 55*(4), 189–208. [DOI](https://doi.org/10.1037/h0061626).

Torchvision Maintainers and Contributors. (n.d.). *VGG19*. Accessed July 17, 2026. [Official documentation](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vgg19).

Wikimedia Commons contributors. (n.d.). *File: Las Meninas, by Diego Velázquez, from Prado in Google Earth.jpg*. Accessed July 17, 2026. [File and licensing page](https://commons.wikimedia.org/wiki/File:Las_Meninas,_by_Diego_Vel%C3%A1zquez,_from_Prado_in_Google_Earth.jpg).

Xie, S., & Tu, Z. (2015). Holistically-nested edge detection. *Proceedings of the IEEE International Conference on Computer Vision*, 1395–1403. [ICCV paper](https://openaccess.thecvf.com/content_iccv_2015/html/Xie_Holistically-Nested_Edge_Detection_ICCV_2015_paper.html).
