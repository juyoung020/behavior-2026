# arXiv 2512.10071 본문(HTML판) — Openpi Comet: Competition Solution For 2025 BEHAVIOR Challenge

> 원본: https://arxiv.org/html/2512.10071
> 받은 시각: 2026-09-29 18:22 KST (2026-09-29 09:22 UTC)
> 비고: 수식·그림은 변환 중 깨질 수 있음. 정확한 수치는 PDF 로 확인
> 이 파일은 `_scrape.py` 가 자동으로 받은 원문 보관본이다. HTML→Markdown 변환 중 서식은 달라질 수 있으나 문장은 원문 그대로다.

---
# Openpi Comet: Competition Solution For 2025 BEHAVIOR Challenge

  Team Comet

###### Abstract

The 2025 BEHAVIOR Challenge is designed to rigorously track progress toward solving long-horizon tasks by physical agents in simulated environments. BEHAVIOR-1K focuses on everyday household tasks that people most want robots to assist with and these tasks introduce long-horizon mobile manipulation challenges in realistic settings, bridging the gap between current research and real-world, human-centric applications. This report presents our solution to the 2025 BEHAVIOR Challenge in a very close 2nd place and substantially outperforms the rest of the submissions. Building on π0.5\pi_{0.5}, we focus on systematically building our solution by studying the effects of training techniques and data. Through careful ablation studies, we reveal the scaling benefits in both the pre-training and post-training phases, leading to a validation Q-score of 0.345, significantly surpassing previous state-of-the-art performance. We summarize our practical lessons and design recommendations that we hope will provide actionable insights for the broader embodied AI community when adapting powerful foundation models to complex embodied scenarios.

| [https://github.com/mli0603/openpi-comet](https://github.com/mli0603/openpi-comet) |
|---|

## 1 Introduction

Vision-Language-Action (VLA) models (Brohan et al., 2022; Brohan et al., 2023; Octo Model Team et al., 2024; Kim et al., 2024; Qu et al., 2025; Black et al., 2024) have recently emerged as a unifying paradigm for robotic policy learning, leveraging large-scale robot datasets to acquire robust and generalizable manipulation and navigation capabilities. By integrating perception, language understanding, and control within a single end-to-end framework, VLAs bypass the need for hand-engineered modules and have demonstrated strong performance across a variety of embodied AI benchmarks. Despite this progress, most existing VLA systems are primarily optimized for short-horizon tasks, and their ability to scale to complex, temporally extended activities remains limited.

Long-horizon manipulation (Zhao et al., 2025; Zawalski et al., 2024) introduces additional difficulties that fundamentally challenge current VLA designs. Such tasks require orchestrated sequences of interdependent behaviors, where compounding errors and shifting state distributions can degrade performance over time. A common approach is to decompose tasks into subtasks (Lin et al., 2022; Shi et al., 2023; Tie et al., 2025) and train separate local policies. However, this strategy does not resolve the skill chaining problem (Chen et al., 2024; Konidaris and Barto, 2009), which involves modeling and executing reliable transitions between subtasks while mitigating error accumulation. In addition, many solutions proposed for skill chaining rely on online adaptation or modular architectures, and these methods are often incompatible with the large-scale, offline, end-to-end training paradigm that underpins modern VLA models. Consequently, achieving reliable long-horizon performance while preserving scalability and generality remains an open challenge.

The BEHAVIOR Challenge, built upon the BEHAVIOR-1K (Li et al., 2024) benchmark, provides an rigorous benchmark for this problem. It features realistic household environments containing complex object interactions, and evaluates agents on 50 long-horizon tasks that reflect human-centered daily activities. Each task requires multi-step reasoning, precise manipulation, and coordinated navigation, making success highly dependent on robust long-horizon policy execution. With 10,000 expert demonstrations and a standardized evaluation protocol, the challenge places strong emphasis on generalization, control robustness, and error tolerance. These capabilities remain difficult for current VLA models to achieve consistently.

In this report, we examine how far a strong publicly available VLA backbone can be pushed on long-horizon tasks using careful data, training, and inference design within a simple end-to-end training pipeline. We treat the BEHAVIOR Challenge as a case study in adapting powerful but generic foundation policies to a complex embodied benchmark. Through systematic exploration of training configurations, pre-training choices, and inference strategies, we show that our solution completes 22 tasks out of the 50 household tasks, achieving a Q-score of 0.2514 in the competition Table 1. After the challenge, we have refined our post-training algorithm and achieved a validation Q-score of 0.3453, substantially outperforming all previous results.

## 2 Architecture

As shown in Figure 1(a), we adopt the π0.5\pi_{0.5} as the base policy of our system. π0.5\pi_{0.5} follows the standard VLA design paradigm, combining a visual encoder for multi-view robot observations with a language encoder for task instructions, and fusing these modalities into a shared representation that conditions the action expert. The action expert is implemented as a transformer-style network that ingests features and denoises the low-level continuous control actions at each timestep. This end-to-end architecture allows perception, language understanding, and control to be trained jointly from large-scale robot datasets, and π0.5\pi_{0.5} further enhances generalization by being pretrained on heterogeneous data spanning multiple embodiments, environments, and tasks. The dataset includes the 1k hours human demonstrations officially provided by the BEHAVIOR Challenge, as well as our additional motion-planner trajectories and offline RL rollouts, which together supply rich long-horizon behaviors and diverse manipulation strategies crucial for robust policy learning.

For post-training, we adopt an iterative RFT procedure as illustrated in Figure 1(b). Starting from the official human demonstrations, we introduce random pose perturbations and roll out the pretrained policy under these disturbed initial conditions. Successful episodes are retained as additional training data, progressively forming an offline data flywheel that continually improves the robustness and coverage of the policy.

![Refer to caption](https://arxiv.org/html/2512.10071v3/pipeline.png) Figure 1: (a) Pre-training on large-scale heterogeneous data, including 1.1K hours of human demonstrations and ∼\sim0.4K hours of additional planner and offline RL trajectories. (b) RFT post-training: perturb initial poses, roll out the policy, and retain successful episodes to iteratively augment the dataset.

 Table 1: Results of 2025 BEHAVIOR Challenge for standard track and our post-challenge solution. Q-score for the test set is used for final ranking. Best viewd in color.

| Rank | Team | Full Task Success Rate | Q-Score |   |   |
|---|---|---|---|---|---|
| Validation | Test | Validation | Test |   |   |
|   | Comet (ours, post-challenge) | 0.1500 |   | 0.3453 |   |
| 1 | Robot Learning Collective | 0.1120 | 0.1240 | 0.2605 | 0.2599 |
| 2 | Comet (ours) | 0.1440 | 0.1140 | 0.183011 1 Due to limited time by the submission deadline, we could not finish evaluation across all tasks. | 0.2514 |
| 3 | SimpleAI Robot | 0.1400 | 0.1080 | 0.1943 | 0.1591 |
| 4 | The North Star | 0.1280 | 0.0760 | 0.1702 | 0.1204 |

## 3 Dataset

 (a) Skill distribution.

 (b) Dataset statistic.

 Figure 2: Dataset statistics and skill distribution for BEHAVIOR Challenge @ NeurIPS 2025. (a) Proportion of video frames occupied by each skill across the entire dataset. (b) Per-task distribution, showing the average trajectory length (in frames) and the average number of unique skills per trajectory.

The BEHAVIOR-1K benchmark provides 1,000 realistic household activities instantiated in 50 fully interactive 3D scenes with over 10k objects, designed to stress long-horizon mobile manipulation and high-level reasoning in human-centric environments. The NeurIPS 2025 BEHAVIOR Challenge selects 50 representative tasks from BEHAVIOR-1K and supplies 10,000 teleoperated expert demonstrations (200 per task, over 1,200 hours) with multi-modal observations and fine-grained skill annotations. Policies are evaluated in simulation by the task success rate defined via BDDL goal predicates, emphasizing completion of entire activities rather than short-horizon subroutines.

Beyond the official teleoperated dataset, we further incorporate ∼\sim3.6K trajectories composed of motion-planner demonstrations and offline RL rollouts. The planner data provides precise low-noise manipulation sequences, while the RL rollouts introduce broader behavioral variability. Together, these additional trajectories substantially enrich the state–action coverage beyond human demonstrations.

To better understand the learning landscape, we analyze the demonstration dataset at the level of semantic skills and per-task complexity. As shown in Figure 2(a), the distribution is highly imbalanced: move to and pick up from dominate with roughly 33.3% and 24.4% of frames, followed by place in (8.8%) and a long tail of infrequent skills (11.5%). Figure 2(b) reports per-task statistics: many tasks require trajectories of several hundred frames and typically involve 5–10 distinct skills, with some exceeding 12. We treat tasks with an average length below 250 frames as relatively simple, since they involve shorter horizons and fewer skill compositions, and they are particularly useful for quickly bringing up our system and validating basic policy behavior in the simulator. In contrast, several tasks, including Task 48 and Task 49, fall into a clearly harder regime, characterized by long-horizon execution and rich mixtures of navigation and manipulation skills (e.g., moving between rooms, opening/closing containers, and rearranging multiple objects). These high-complexity tasks provide a stringent testbed for evaluating policies’ ability to handle extended temporal credit assignment and frequent skill switching.

## 4 Experiment

In this section, we present details of our implementations (Section 4.1), pre-training (Section 4.2) and post-training (Section 4.3). We report the results across different stages in section 4 and present detailed ablations in Section 4.4.

 Table 2: Validation Q-scores across training stages.

| Pre-training | Post-training | Theoretical best |
|---|---|---|
| 0.192 | 0.345 | 0.611 |

### 4.1 Implementation details

We adopt π0.5\pi_{0.5} implemented in JAX as our policy, and all experiments are conducted on NVIDIA H200 GPUs with a per-device batch size of 64, using cosine decay scheduler. For single task SFT, we train for 15k–20k steps. For all multi-task pre-training, we use 50k training steps, with a learning rate of 2.5×10−52.5\times 10^{-5}. During both SFT and the subsequent RFT adaptation, we use a reduced learning rate of 2.5×10−62.5\times 10^{-6} on 8 GPUs to ensure stable fine-tuning.

To accelerate offline rollout generation, we parallelize evaluation across the 10 test instances by distributing them over multiple GPUs, which mitigates the extremely slow simulation rate of BEHAVIOR—where evaluating a single task can otherwise take from one hour to nearly a full day, making online RL potentially inefficient under this simulator Yu et al. (2025).

### 4.2 Pre-training

To study how pre-training task coverage and diversity affect long-horizon performance, we compare single-task pre-training pt1 with three multi-task pre-training settings on the BEHAVIOR Challenge: #pt7, #pt10, and #pt50. Each setting trains a single VLA policy on demonstrations from 7, 10, or all 50 challenge tasks, For each configuration, we train the model end-to-end on the corresponding task subset and then evaluate on the same 50-task challenge protocol. This design allows us to isolate how increasing the number and heterogeneity of pre-training tasks influences task success, while keeping the overall training recipe fixed.

As shown in Figure 3, pt1 trains the policy only on demonstrations from a single target task; this single-task finetuning regime yields the lowest average success and only produces successful rollouts on 2 tasks, indicating that purely task-specific adaptation is insufficient for robust long-horizon control. Building on this baseline, pt7 pre-trains on a small subset of relatively short-horizon BEHAVIOR Challenge tasks such as bringing water, cook hot dogs, and make microwave popcorn, which share similar interactions with common household objects. As the pre-training set expands from pt7 to pt10 and finally pt50, more tasks begin to exhibit successful rollouts, showing that broader task coverage improves the model’s ability to generalize. In particular, pt10 augments pt7 with additional, slightly more complex tasks (e.g., moving boxes to storage, hanging pictures), while pt50 uses demonstrations from all 50 challenge tasks, including rare and highly compositional activities such as rearranging kitchen furniture and setting the fire, thus exposing the policy to the full long-horizon distribution during pre-training.

 Figure 3: Training subset coverage and the number of successful tasks.

After pre-training, we reach a validation Q-score of 0.192 on the validation set, as shown in section 4.

### 4.3 Post-training

Despite the recent progress in online RL (Chen et al., 2025), its low sample efficiency makes it impractical for the BEHAVIOR challenge. Moreover, online RL requires a heterogeneous compute setup: GPUs with RT Cores are needed for simulation, while GPUs with Tensor Cores are needed for model training and rollouts. Given these constraints, we instead adopt rejection sampling fine-tuning (RFT), a technique shown to be effective in both LLMs and VLMs (Ahn et al., 2024; Touvron et al., 2023; Azzolini et al., 2025). An overview of our RFT setup is provided in Algorithm 1. Starting from all the provided train and validation human demonstrations for each scene, we randomly perturb the robot’s initial pose and use our pre-trained policy to perform rollouts under these perturbed configurations. Using both train and validation set helps avoid overfitting to the validation set and provides better signals of the model quality. Successful rollouts are retained as additional demonstrations. We perform N=3N=3 rounds of RFT in total, collecting on average T=8500T=8500 trajectories per round, and eventually selected 1469 trajectories for training after de-duplication and task balancing.

 Algorithm 1  The RFT Algorithm

 1: Initialize 𝒟←\mathcal{D}\leftarrow human demos.

 2: Initialize π1\pi_{1} to pre-trained πp​t\pi_{pt}.

 3: for i=1i=1 to NN do

 4:   𝒟i←∅\mathcal{D}_{i}\leftarrow\emptyset

 5:   for t=1t=1 to TT do

 6:    Sample initial state s0s_{0} from 𝒟\mathcal{D}.

 7:    s0′←s0+ϵs^{\prime}_{0}\leftarrow s_{0}+\epsilon.

 8:    Rollout τ\tau from s0′s^{\prime}_{0} using πi\pi_{i}.

 9:    𝒟i←𝒟i∪𝒟τ\mathcal{D}_{i}\leftarrow\mathcal{D}_{i}\cup\mathcal{D}_{\tau} if successful.

 10:   end for

 11:   𝒟←𝒟∪𝒟i\mathcal{D}\leftarrow\mathcal{D}\cup\mathcal{D}_{i}.

 12:   Train πi+1\pi_{i+1} on 𝒟\mathcal{D}.

 13: end for

 14: return best πi\pi_{i} on validation.

During the challenge, our suite of models achieves a validation Q-score of 0.224 after post-training. After the challenge, we further refine the task balancing strategy and achieve a significantly higher Q-score of 0.345 on the validation set, using only two model checkpoints. Representative policy execution visualizations are provided in Figure 5. Beyond improving overall performance, Rejection Sampling Fine-Tuning (RFT) also serves as a diagnostic mechanism by aggregating successful instances across historical checkpoints. As shown in section 4, the theoretical best results we could obtain is a validation Q-score of 0.611 and a success rate of 0.35, revealing substantial headroom for further optimization. Further scaling up the model size could be helpful to realize the full potential of our post-training strategy.

 Figure 4: We can achieve an aggregated validation Q-score of 0.345 and a validation success rate of 15% across 50 tasks. Our models can perform easy tasks such as turnining_on_radio robustly. Zero-success rate tasks are not shown.

### 4.4 Ablations

Beyond pre-training and post-training, we find that low-level design decisions in training and inference have a substantial impact on the performance. Table 3 reports a set of controlled ablations along four such axes: Action Horizon and Input Modality for the training procedure, and Control Mode and Image Resolution for the inference strategy.

Control mode (#1). Temporal Ensemble and Receding Temporal fail to produce stable closed-loop behavior and result in near-zero success rates. In contrast, the Receding Horizon scheme, which executes all the predicted action segment and performs re-planning after finishing manipulation, significantly improves performance. This result highlights the necessity of continuous feedback for long-horizon manipulation and shows that smoothing or averaging open-loop predictions quickly accumulates error.

Action horizon (#2). Varying the action horizon reveals a non-monotonic relationship between prediction length and downstream control performance. Moderate horizons strengthen long-horizon manipulation by allowing the model to anticipate multi-stage behaviors, whereas excessively long horizons introduce conflicting temporal dependencies that compromise stable control. These findings indicate that the horizon length must be chosen to balance future awareness with the reliability of short-term control. We empirically find that setting the receding horizon to 32 gives us the best results.

 Table 3: Ablation study of control mode, action horizon, input modality, and image resolution on the turning on radio task.

| Settings | Control Mode | Action Horizon | Input Modality | Image Resolution | Success Rate |
|---|---|---|---|---|---|
| #1 | Temporal Ensemble | 50 | RGB | Head:224 Wrist:224 | 0.00 |
| Receding Temporal | 0.00 |   |   |   |   |
| Receding Horizon | 0.25 |   |   |   |   |
| #2 | Receding Horizon | 8 | RGB | Head:224 Wrist:224 | 0.00 |
| 16 | 0.10 |   |   |   |   |
| 50 | 0.25 |   |   |   |   |
| 32 | 0.30 |   |   |   |   |
| #3 | Receding Horizon | 32 | RGB+Depth Image | Head:224 Wrist:224 | 0.20 |
| RGB+Point Cloud | 0.30 |   |   |   |   |
| RGB | 0.30 |   |   |   |   |
| #4 | Receding Horizon | 32 | RGB | Head:224 Wrist:224 | 0.30 |
| Head:720 Wrist:480 | 0.60 |   |   |   |   |

![Refer to caption](https://arxiv.org/html/2512.10071v3/example.png) Figure 5: Rollout Examples on BEHAVIOR-1K. Our policy successfully handles long-horizon, multi-stage household tasks involving navigation, fine manipulation, and tool use, demonstrating robust execution across diverse household activities.

Input modality (#3). We find that reconstructed point clouds improve performance compared to depth map as an input modality, which suggests that explicit geometric structure provides more informative cues for object-centric manipulation. However, the improvement over RGB-only inputs is limited, while the computational and latency overhead is considerable. Overall, explicit 3D geometry offers benefits in cluttered scenes, but the gains are not consistently large enough to justify the increased system complexity.

Image resolution (#4). Increasing the spatial resolution of both head and wrist views leads to substantial gains in performance. Low-resolution inputs of 224 by 224 pixels provide only coarse visual information, whereas high-resolution inputs more than double the success rate. These results indicate that precise visual cues are critical for reliable manipulation in visually complex environments, and that high-fidelity perception plays an essential role in long-horizon tasks.

Data process (#5). We further evaluate several data-processing strategies, including action representation, action subsampling, removal of proprioceptive state inputs, and skill weighting about Manipulation and Navigation in Table 4. We use default training setting as the folling: Receding Horizon, 32 action horizon, RGB input, and 224x224 image resolution. Results shows that relative-action parameterization and the removal of state both lead to worse results, suggesting that absolute action anchoring and explicit system observability are essential for stable closed-loop optimization. Action subsampling at 15 Hz accelerates execution but reduces temporal precision, leading to degraded manipulation accuracy. Finally, although reweighting the dataset toward manipulation segments shifts the empirical distribution, it yields no measurable improvement, indicating that simple resampling is insufficient to compensate for the structural difficulty and heterogeneous dynamics of long-horizon manipulation.

 Table 4: Ablation study of action representation, sampling frequency, proprioceptive state inputs, and skill weighting under default training settings in Table 3.

| Settings | Action Representation | Action Sampling | State Input | Skill weighting | Success Rate |
|---|---|---|---|---|---|
| #5.1 | Delta Joint | 30 Hz | ✓ | No Weighting | 0.00 |
| Absolute Joint | 0.30 |   |   |   |   |
| #5.2 | Absolute Joint | 15 Hz | ✓ | No Weighting | 0.00 |
| 30 Hz | 0.30 |   |   |   |   |
| #5.3 | Absolute Joint | 30 Hz | ✗ | No Weighting | 0.00 |
| ✓ | 0.30 |   |   |   |   |
| #5.4 | Absolute Joint | 30 Hz | ✓ | No Weighting | 0.30 |
| Manip:Nav=2:1 | 0.30 |   |   |   |   |

## 5 Conclusion

In this report, we presented our solution to the 2025 BEHAVIOR Challenge, adapting the publicly available π0.5\pi_{0.5} backbone to a demanding long-horizon household benchmark and systematically studying how pre-training task coverage, post-training, and inference-time design choices affect performance on all tasks. Our experiments reveal that scaling pre-training over more numerous and diverse BEHAVIOR tasks significantly generates and unlocks success on rare, compositional activities. Additionally, We show that RFT as a post-training technique avoids the infrastructure issues from online RL yet yield trackable targets as well as significant performance boost.

Despite the promising results, we acknowledge that our Q-score is still far from perfect. We find that despite the effectiveness of RFT, the sampling efficiency is still too low. Paradigms such as DAgger or on-policy distillation where an expert policy potentially using privileged data could greatly improve the sampling efficiency. In addition, RL approaches that provide both positive and negative rewards could balance the learning.

We believe that combining strong VLA backbones with more structured long-horizon reasoning, richer post-training objectives, and better curriculum design will further close the gap between synthetic benchmarks and real-world deployment, and we hope our empirical findings provide practical guidance for scaling foundation policies to complex, human-centric environments.

## 6 Authors

Core Contributors: Delin Qu11 1 First authors with random order, Qizhi Chen11 1 First authors with random order, Shangkun Sun11 1 First authors with random order, Zhaoshuo Li, Yu-Wei Chao, Xiaohui Zeng, Xuan Li, Junjie Bai, Tsung-Yi Lin, Ming-Yu Liu

Contributors: Kaichun Mo, Jinwei Gu, Moo Jin Kim, Fangyin Wei, Hongchi Xia, Nic Ma

## References

- Ahn et al. (2024) J. Ahn, R. Verma, R. Lou, D. Liu, R. Zhang, and W. Yin  Large language models for mathematical reasoning: progresses and challenges.  arXiv preprint arXiv:2402.00157.  Cited by: §4.3.
- Azzolini et al. (2025) A. Azzolini, J. Bai, H. Brandon, J. Cao, P. Chattopadhyay, H. Chen, J. Chu, Y. Cui, J. Diamond, Y. Ding, et al.  Cosmos-reason1: from physical common sense to embodied reasoning.  arXiv preprint arXiv:2503.15558.  Cited by: §4.3.
- Black et al. (2024) K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, et al.  π0\pi_{0}: A vision-language-action flow model for general robot control.  arXiv preprint arXiv:2410.24164.  Cited by: §1.
- Brohan et al. (2023) A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choromanski, T. Ding, D. Driess, A. Dubey, C. Finn, et al.  Rt-2: vision-language-action models transfer web knowledge to robotic control.  arXiv preprint arXiv:2307.15818.  Cited by: §1.
- Brohan et al. (2022) A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu, et al.  Rt-1: robotics transformer for real-world control at scale.  arXiv preprint arXiv:2212.06817.  Cited by: §1.
- Chen et al. (2025) K. Chen, Z. Liu, T. Zhang, Z. Guo, S. Xu, H. Lin, H. Zang, Q. Zhang, Z. Yu, G. Fan, et al.  PiRL: online rl fine-tuning for flow-based vision-language-action models.  arXiv preprint arXiv:2510.25889.  Cited by: §4.3.
- Chen et al. (2024) Z. Chen, Z. Ji, J. Huo, and Y. Gao  Scar: refining skill chaining for long-horizon robotic manipulation via dual regularization.  Advances in Neural Information Processing Systems 37, pp. 111679–111714.  Cited by: §1.
- Kim et al. (2024) M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, et al.  OpenVLA: an open-source vision-language-action model.  arXiv preprint arXiv:2406.09246.  Cited by: §1.
- Konidaris and Barto (2009) G. Konidaris and A. Barto  Skill discovery in continuous reinforcement learning domains using skill chaining.  Advances in neural information processing systems 22.  Cited by: §1.
- Li et al. (2024) C. Li, R. Zhang, J. Wong, C. Gokmen, S. Srivastava, R. Martín-Martín, C. Wang, G. Levine, W. Ai, B. Martinez, et al.  Behavior-1k: a human-centered, embodied ai benchmark with 1,000 everyday activities and realistic simulation.  arXiv preprint arXiv:2403.09227.  Cited by: §1.
- Lin et al. (2022) X. Lin, Z. Huang, Y. Li, J. B. Tenenbaum, D. Held, and C. Gan  Diffskill: skill abstraction from differentiable physics for deformable object manipulations with tools.  arXiv preprint arXiv:2203.17275.  Cited by: §1.
- Octo Model Team et al. (2024) Octo Model Team, D. Ghosh, H. Walke, K. Pertsch, K. Black, O. Mees, S. Dasari, J. Hejna, C. Xu, J. Luo, T. Kreiman, Y. L. Tan, L. Y. Chen, P. Sanketi, Q. Vuong, T. Xiao, D. Sadigh, C. Finn, and S. Levine  Octo: an open-source generalist robot policy.  In Proceedings of Robotics: Science and Systems (RSS),  Cited by: §1.
- Qu et al. (2025) D. Qu, H. Song, Q. Chen, Y. Yao, X. Ye, Y. Ding, Z. Wang, J. Gu, B. Zhao, D. Wang, et al.  Spatialvla: exploring spatial representations for visual-language-action model.  arXiv preprint arXiv:2501.15830.  Cited by: §1.
- Shi et al. (2023) H. Shi, H. Xu, S. Clarke, Y. Li, and J. Wu  Robocook: long-horizon elasto-plastic object manipulation with diverse tools.  arXiv preprint arXiv:2306.14447.  Cited by: §1.
- Tie et al. (2025) C. Tie, S. Sun, J. Zhu, Y. Liu, J. Guo, Y. Hu, H. Chen, J. Chen, R. Wu, and L. Shao  Manual2skill: learning to read manuals and acquire robotic skills for furniture assembly using vision-language models.  arXiv preprint arXiv:2502.10090.  Cited by: §1.
- Touvron et al. (2023) H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, et al.  Llama 2: open foundation and fine-tuned chat models.  arXiv preprint arXiv:2307.09288.  Cited by: §4.3.
- Yu et al. (2025) C. Yu, Y. Wang, Z. Guo, H. Lin, S. Xu, H. Zang, Q. Zhang, Y. Wu, C. Zhu, J. Hu, et al.  RLinf: flexible and efficient large-scale reinforcement learning via macro-to-micro flow transformation.  arXiv preprint arXiv:2509.15965.  Cited by: §4.1.
- Zawalski et al. (2024) M. Zawalski, W. Chen, K. Pertsch, O. Mees, C. Finn, and S. Levine  Robotic control via embodied chain-of-thought reasoning.  arXiv preprint arXiv:2407.08693.  Cited by: §1.
- Zhao et al. (2025) Q. Zhao, Y. Lu, M. J. Kim, Z. Fu, Z. Zhang, Y. Wu, Z. Li, Q. Ma, S. Han, C. Finn, et al.  Cot-vla: visual chain-of-thought reasoning for vision-language-action models.  In Proceedings of the Computer Vision and Pattern Recognition Conference,  pp. 1702–1713.  Cited by: §1.
