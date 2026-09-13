# ACT primary-source notes

Source under review:

Tony Z. Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. “Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.” *Robotics: Science and Systems*, 2023.

- Proceedings PDF: <https://www.roboticsproceedings.org/rss19/p016.pdf>
- arXiv record: <https://arxiv.org/abs/2304.13705>
- Project: <https://tonyzhaozh.github.io/aloha/>

This is a claim ledger, not publishable prose. It records where claims used by the ACT knowledge chain come from.

## Scope

- The paper contributes both the ALOHA teleoperation hardware system and ACT, its imitation-learning algorithm. Do not describe ACT as the entirety of the paper. (Abstract, Introduction, Section VII)
- ACT is evaluated on two simulated and six real-world fine-grained bimanual tasks. (Section V)

## Problem claims

- Fine manipulation requires precision, coordination of contact forces, and closed-loop visual feedback; millimeters of error can cause failure. (Abstract, Section I)
- End-to-end imitation learning is sensitive to the training distribution. Small policy errors can change future states and worsen compounding error. (Section I; Section II, “Addressing compounding errors”; Section IV)
- Human demonstrations are non-stationary/noisy: the same observation may admit different trajectories, humans are more stochastic where precision matters less, and pauses can act as temporally correlated confounders. (Section I; Section IV-A/B; Section V-B)

## Method claims

- The action is the target joint position for both arms at the next timestep; a low-level PID controller tracks it. Observations contain follower joint positions and four camera feeds. (Section IV)
- Action chunking models `pi(a[t:t+k] | s[t])` rather than `pi(a[t] | s[t])`. The paper describes this as a `k`-fold reduction in effective horizon. (Section IV-A, Figure 6)
- The actual ACT inference loop queries the policy every timestep. Overlapping chunks provide multiple predictions for the same timestep, which temporal ensembling combines with exponential weights. This is not ordinary smoothing across adjacent executed timesteps. (Section IV-A, Figure 6, Algorithm 2)
- ACT trains the action-chunk policy as a CVAE. The encoder is used only for training; it receives proprioception and the demonstration action sequence. The decoder/policy receives images, joints, and `z`. At inference, `z` is set to the prior mean, zero. (Section IV-B, Figure 5, Algorithms 1–2)
- Transformers implement the CVAE encoder and decoder/policy because they synthesize and generate sequences. The policy combines ResNet image features, joint positions, and `z`, then predicts a `k x 14` action tensor. (Section IV-C, Figure 5, Appendix C)
- Prose in Section IV-C states that the implementation uses L1 reconstruction loss, while Algorithm 1 labels the reconstruction term as MSE. Any future loss page must acknowledge or resolve this internal inconsistency instead of silently choosing one statement.

## Evidence and limits

- Without temporal ensembling, aggregate simulated success rises from 1% at `k=1` to 44% at `k=100`, then tapers for larger chunks near open-loop control. This supports an intermediate chunk size, not “larger is always better.” (Section VI-A, Figure 9a)
- Temporal ensembling adds 3.3 percentage points for ACT and 4 points for BC-ConvMLP in the reported aggregate, but hurts VINN by 20 points. Do not claim it universally improves every policy. (Section VI-A, Figure 9b)
- Removing the CVAE objective barely changes scripted-data results but drops the human-data aggregate from 35.3% to 2%. (Section VI-B, Figure 9c)
- Final real-world success rates reported for ACT are 88% Slide Ziploc, 96% Slot Battery, 84% Open Cup, 20% Thread Velcro, 64% Prep Tape, and 92% Put On Shoe. Preserve the low-performing tasks when discussing results. (Tables II–III, Section V-C)
- Real-world datasets use 50 demonstrations per task except Thread Velcro with 100, totaling roughly 10–20 minutes of demonstration data per task. (Section V-B)
- The paper explicitly notes tasks beyond the system, such as buttoning a dress shirt. (Section VII; Appendix F)

