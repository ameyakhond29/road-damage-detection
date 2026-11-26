# Related work summary

Short literature context for the Week 2 research question. This is a reading summary for the proposal—not an experimental result.

## Road damage detection datasets and challenges

- **RDD / CRDDC series.** Arya et al. released multi-country road-damage imagery culminating in **RDD2022**, used in the Crowdsensing-based Road Damage Detection Challenge (CRDDC’2022). Images and Pascal VOC annotations cover Japan, India, Czech Republic, Norway, the United States, and China (motor-bike and drone). The four scored damage types are typically **D00, D10, D20, D40**.  
  Sources: [CRDDC data portal](https://crddc2022.sekilab.global/data/), [Figshare RDD2022](https://doi.org/10.6084/m9.figshare.21431547), [sekilab/RoadDamageDetector](https://github.com/sekilab/RoadDamageDetector), Arya et al. geoscience data journal article on RDD2022.

- **Domain shift across countries.** Camera height, resolution, pavement appearance, and labeling practice differ by country. Challenge leaderboards and follow-up papers emphasize that models trained on one region often degrade elsewhere—motivating our **country-held-out** evaluation (train on Japan/India/Czech/China_MotorBike; evaluate on Norway/United States when labels allow).

## Lightweight object detectors

- **YOLO family (Ultralytics).** YOLOv8 and YOLO11 provide nano/small variants with strong speed–accuracy trade-offs and a standard training recipe (mosaic, HSV jitter, AdamW/SGD schedules). We use **YOLOv8n** as a **reference baseline** and **YOLO11n** as the **strong baseline** so architectural and resolution changes are measured against a modern small detector—not an outdated one.

- **Small-object detection.** Thin cracks occupy few pixels at 640². Common remedies in the detection literature include:  
  - **Higher-resolution feature maps / extra P2 (stride-4) heads** (FPN-style multi-scale detection, YOLO-P2 / small-object YOLO variants).  
  - **Attention in the neck** (SE, CBAM, and lighter variants) to emphasize informative channels/locations.  
  - **Larger input sizes**, which improve small-object AP at roughly quadratic cost in compute for dense predictors.

Our Week 2 design treats these as **controlled single-factor changes** and asks which gives the best **accuracy per added GFLOPs/latency**.

## Evaluation practice

- Challenge and survey papers report **mAP@0.5** (and often mAP@0.5:0.95), plus per-class scores for imbalanced damage types.  
- Deployment-oriented work also reports **parameters, FLOPs, latency, and memory**.  
- We follow that dual reporting and add **multi-seed means/std** (seeds 0/1/2) with optional paired bootstrap CIs—without claiming significance from one-seed pilots.

## How this project differs

| Gap in prior practice | Our response |
|----------------------|--------------|
| Many reports change several knobs at once | Single-factor ablations (P2 **or** attention **or** 800 px), then one combined model |
| Country mix often pooled without held-out geography | Explicit country-held-out experiment |
| Test set used for tuning | Fixed split; **val-only** checkpoint selection |
| Weak or missing compute accounting | Mandatory param/GFLOPs/latency/memory/time log |

## Key references (for citation in later write-ups)

1. Arya et al., RDD2022 / CRDDC’2022 dataset and challenge materials.  
2. Ultralytics YOLOv8 and YOLO11 documentation and model cards.  
3. Lin et al., Feature Pyramid Networks (multi-scale detection motivation).  
4. Woo et al., CBAM; Hu et al., SENet (attention blocks).  
5. Small-object detection surveys (extra high-res heads and input scaling).

Exact bibliographic entries will be finalized in the project report; this file is the Week 2 conceptual map only.
