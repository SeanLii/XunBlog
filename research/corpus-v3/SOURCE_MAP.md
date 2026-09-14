# Source Map

模型、算法与架构页面以原论文为主要依据；涉及 released behavior / tensor flow / fixed defaults 时，以官方实现作为主要实现依据。论文与当前代码不一致时分别记录，不静默统一。

## ACT / Existing v2 Sources

- **ACT / ALOHA** — Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (2023): https://arxiv.org/abs/2304.13705
- **ACT official repository**: https://github.com/tonyzhaozh/act
- **Transformer** — Vaswani et al., *Attention Is All You Need* (2017): https://arxiv.org/abs/1706.03762
- **DETR / learned queries** — Carion et al., *End-to-End Object Detection with Transformers* (2020): https://arxiv.org/abs/2005.12872
- **VAE** — Kingma & Welling, *Auto-Encoding Variational Bayes*: https://arxiv.org/abs/1312.6114
- **CVAE** — Sohn, Lee & Yan, *Learning Structured Output Representation using Deep Conditional Generative Models* (2015)
- **DAgger** — Ross, Gordon & Bagnell (2011): https://proceedings.mlr.press/v15/ross11a.html
- **ResNet** — He et al., *Deep Residual Learning for Image Recognition*: https://arxiv.org/abs/1512.03385

## π0 Primary Sources

- **π0** — Black et al., *π0: A Vision-Language-Action Flow Model for General Robot Control* (2024): https://arxiv.org/abs/2410.24164
- **Physical Intelligence π0 research page** — *π0: Our First Generalist Policy*: https://www.pi.website/blog/pi0
- **Official openpi repository**: https://github.com/Physical-Intelligence/openpi
- **Current π0 model implementation**: https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py
- **Current π0 configuration**: https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0_config.py

## π0 Prerequisite Sources

- **Flow Matching** — Lipman et al., *Flow Matching for Generative Modeling* (2022): https://arxiv.org/abs/2210.02747
- **Rectified Flow** — Qiang Liu, *Rectified Flow: A Marginal Preserving Approach to Optimal Transport* (2022): https://arxiv.org/abs/2209.14577
- **PaliGemma** — Beyer et al., *PaliGemma: A versatile 3B VLM for transfer* (2024): https://arxiv.org/abs/2407.07726
- **VLA / RT-2** — Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control* (2023): https://arxiv.org/abs/2307.15818
- **OpenVLA** — Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model* (2024): https://arxiv.org/abs/2406.09246
- **Open X-Embodiment** — Open X-Embodiment Collaboration et al., *Open X-Embodiment: Robotic Learning Datasets and RT-X Models* (2023): https://arxiv.org/abs/2310.08864

## Page-to-Source Mapping for π0

| Page | Primary grounding |
|---|---|
| `π0` | π0 paper Sections I, III, IV + official research page |
| `Architecture` | π0 paper Section IV + Appendix B + `openpi/pi0.py` |
| `Action Expert` | π0 paper Appendix B + `pi0_config.py` + `pi0.py` |
| `Blockwise Causal Attention Mask` | π0 paper Appendix B + `pi0.py` |
| `Flow Matching in π0` | π0 paper Section IV + Flow Matching paper + `pi0.py` |
| `Pre-training and Post-training in π0` | π0 paper Sections III, V + official research page |
| `Training` | π0 paper Section IV + `Pi0.compute_loss` |
| `Inference` | π0 paper Appendix D + `Pi0.sample_actions` |
| `Complete Data Flow` | π0 paper Section IV + Appendix B/D + current openpi implementation |
| `Paper and Released Implementation` | side-by-side comparison of π0 paper and current openpi `main` |

## Important Recorded Discrepancy

π0 paper uses flow time:

```text
τ = 0 → noise
τ = 1 → data
```

Current official `openpi` implementation explicitly uses the reverse convention:

```text
t = 1 → noise
t = 0 → data
```

The corpus keeps both versions separate and explains that they parameterize the same path in opposite directions.

## Grounding Policy

- 原论文负责研究方法、architecture、training recipe 与实验事实。
- 官方代码负责确认 released tensor flow、default dimensions、current inference loop、masks 与 API behavior。
- 当前 repository 同时支持 π0-FAST / π0.5；这些后续兼容代码不能反过来定义 2024 π0 paper。
- 通用理论（Flow Matching、ODE、Euler Method、VLM、VLA）拥有独立 canonical pages，不由 π0 页面重新定义。
