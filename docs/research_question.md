# Research question

## Final question

Starting from a small detector, which single change provides the best accuracy per added computational cost on RDD2022: a **P2 stride-4 detection head**, a **lightweight neck-attention block**, or a **larger input size**? Do these improvements remain effective on countries not seen during training?

## Motivation

Road damage objects—especially thin longitudinal (D00) and transverse (D10) cracks—are often **small and elongated**. Lightweight detectors (YOLO-nano class) are attractive for edge deployment but can miss fine cracks at standard 640² resolution. Three common remedies trade compute for accuracy in different ways:

1. **Higher-resolution features (P2 head)** — detect on stride-4 maps without upscaling every image.  
2. **Neck attention** — reweight channels/spatial locations to emphasize damage cues.  
3. **Larger input (800 px)** — feed the same architecture more pixels.

This project isolates those changes on a **shared RDD2022 split** and measures **accuracy per added cost**, then tests whether gains survive **country shift**.

## Hypotheses

**H1.** The P2 detection head will improve AP50 for thin longitudinal and transverse cracks (D00, D10) more than the attention block at a similar computational cost.

**H2.** Improvements will be smaller on held-out countries than on in-distribution evaluation data, and at least one modification may lose its advantage under country shift.

## Scope boundaries

- Classes: D00, D10, D20, D40 only.  
- China_Drone excluded (different aerial setup).  
- Fixed Week 1 leakage-safe splits; no re-splitting for model shopping.  
- **Validation** for selection; **test / held-out** only for final reporting.  
- No fabricated metrics; one-seed runs labeled as such.

## Alignment with experiment protocol

See `docs/week2_experiment_protocol.md` for roles, experiments 1–5, country-held-out design, metrics, compute logging, and variability plan. The protocol and this question use the same hypotheses, class mapping, and selection rules.
