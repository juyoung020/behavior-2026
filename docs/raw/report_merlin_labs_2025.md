# Removing Bottlenecks: Training Generalist VLAs for the BEHAVIOR-1K Challenge | Merlyn Labs

> 원본: https://merlyn-labs.com/behavior-report
> 받은 시각: 2026-09-29 18:22 KST (2026-09-29 09:22 UTC)
> 비고: 2025 11위 Merlin Labs 보고서 (리더보드 링크)
> 이 파일은 `_scrape.py` 가 자동으로 받은 원문 보관본이다. HTML→Markdown 변환 중 서식은 달라질 수 있으나 문장은 원문 그대로다.

---
---

Open Source Code:[Simulator](https://github.com/Merlyn-Labs/BEHAVIOR-1K)•[Training Code](https://github.com/Merlyn-Labs/openpi)

For the past six weeks, we've been participating in the Stanford BEHAVIOR Challenge, training a generalist policy model to control a Galaxea R1 Pro Humanoid–we call him Arthur–across 50 household tasks in simulation. Our final leaderboard score was a modest 1.78%. Most of these gains occurred almost overnight, jumping from 0 to 0.9% to 1.78% within 48 hours. Several key insights and the removal of serious learning bottlenecks lead to a very effective recipe that we are confident can be scaled further, with no ceiling in sight.

This post shares what we learned along the way: the bottlenecks we discovered, the surprising insights we stumbled into, and a glimpse of something we think points toward a deeper principle in robot intelligence.

Tidying Up the Room(1.5x)

Moving Boxes to Storage(5x)

Arthur successfully completing household tasks

---

## The Challenge

The BEHAVIOR Challenge provides 10,000 teleoperated demonstrations across 50 household tasks in the OmniGibson simulator from the Stanford Vision & Learning Lab. This equates to 200 episodes per task, not a whole lot, but a great start. From there, we took Physical Intelligence's π0.5 VLA as our base, and got started.

Running into Table

Falling Over

Early failures during training

---

## The First Bottleneck: Over-Indexing on Proprioception

Our initial runs showed validation loss dropping monotonically in every situation. By every metric on our training curves, the model was learning. But when we deployed it in simulation, Arthur would barely move. At most, he would tremble in place.

We eventually found the culprit to be the proprioceptive state. At every time step during training, Arthur was fed the current world state, which consisted of images from his cameras (head, left wrist, right wrist) and the proprioceptive state vector. We realized the model had learned to over-index its joint positions—or proprio state—to predict actions that perfectly matched the current trajectory. He had learned that the best way to succeed was to just continue whatever trajectory he was currently on. This worked well to game the loss function, especially for navigation-heavy sub-tasks—which can make up over 60% of individual demonstration—where it's actually advantageous to continue a current trajectory and change very little. But during inference, Arthur had no idea how to generate momentum from rest. It had never learned to move without already being in motion.

Our solution was aggressive proprioceptive dropout during training. By purposefully hiding Arthur's proprioceptive state from him 60% of the time, we forced the model to rely on visual information and language instructions to plan actions rather than simply continuing whatever motion was already happening.

The effect was immediate. Arthur went from a jittering mess to taking meaningful actions for the first time.

Before Proprio Dropout(2x)

After Proprio Dropout(2x)

The effect of proprioceptive dropout on movement initiation

Despite the initial success of proprio dropout, we noticed that *constant* dropout at this rate was hurting Arthur's fine manipulation ability. It was very difficult for him to learn precise grasping without some level of proprioceptive feedback. Our final recipe uses a decaying dropout schedule: 60% early in training to break the proprio over-indexing habit, gradually reducing to 10% to allow proprioceptive awareness back in for fine motor control.

---

## Making Actions Count

With the model finally moving, a new problem emerged: Arthur was slow to move in the world. His movements were timid and lethargic, almost five times slower than the human teleoperated counterparts.

The issue seemed to trace to how the model weighted action loss. Arthur had 23 different joint vectors to control, and actions with smaller magnitudes and movements like those with the arms were being penalized more heavily than large adjustments.

We addressed this directly by implementing action delta magnitude weighting. Actions with a high delta between time steps were proportionally weighted, up to a maximum to ensure the model learned to generate momentum quickly and complete tasks much faster.

---

## Rare Skills and Critical Boundaries

Data scarcity was a meaningful bottleneck. Each task had only 200 demonstrations, but a more granular problem occurred when analyzing particular sub-skills within the demonstrations.

### Skill-based resampling

Some subtasks appear rarely in the demonstrations—opening a drawer might happen once per episode while navigation happens 60-70% of the episode. We realized that Arthur was getting really good at those dense skills, while performing poorly on others. We then began oversampling rare subtask types to ensure the model got enough exposure to learn these infrequent but critical skills. Credit to Wensi Ai (one of the challenge organizers), for suggesting we explore data filtering along these lines.

### Boundary resampling

Further down this avenue, we noticed that more than just underrepresented sub-skills, Arthur was struggling at transition states. He could navigate toward an object, but fumbled the switch to grasping. He could approach a door, but couldn't smoothly transition to opening and walking through it. He was unable to operate effectively at the boundary between two sub-skills.

The solution was exactly the same, by explicitly oversampling boundary regions, we saw substantial improvements in these critical transitions.

To illustrate the imbalance, we analyzed sub-skill distributions across four representative tasks from the BEHAVIOR dataset. The breakdown reveals a consistent pattern: navigation and manipulation primitives dominate, while task-specific skills like chopping, pouring, or operating appliances appear in less than 10% of frames.

#### SKILL DISTRIBUTION BY TASK

| Task | Skills | Frames |
|---|---|---|
| Picking Up Trash | pick up from42.2%move to35.4%place in14.6%place on7.9% | 0.93M |
| Moving Boxes to Storage | move to63.2%pick up from17.7%place on14.0%open door5.1% | 2.85M |
| Make Pizza | pick up from27.7%open door15.0%move to13.6%close door7.8%pour7.5%place on next to7.4%place on6.9%pull/push tray5.4%place in4.1%chop2.9%turn on switch1.1% | 3.83M |

■ Skills appearing in <10% of task frames — candidates for oversampling

#### AGGREGATE SKILL DISTRIBUTION

Combined distribution across 7.6M frames from 4 tasks. Navigation and grasping account for over 60% of all training data, while critical manipulation skills like chopping and operating appliances comprise less than 2%.

move to

34.8%

pick up from

25.8%

place on

9.7%

open door

9.5%

close door

3.9%

place in

3.8%

pour

3.8%

place on next to

3.7%

pull tray

2.2%

chop

1.5%

push tray

0.5%

turn on switch

0.5%

 Dense skills (>10%) Rare skills (<10%) — require oversampling

---

## Two-Phase Training: Generalist Foundation, Specialist Refinement

With the above bottlenecks removed, the model was now able to learn effectively.

We then kicked off a large training run on several tasks, but noticed that at rollout, different tasks were getting jumbled together. When Arthur was supposed to pick up a box, he'd go to the fridge, or search for trash to pick up. At the same time, there was not enough per-task data, to train a strong policy off of a single task.

We resolved this with a two-phase approach:

### Phase 1: Generalist Pre-Training

We trained on all available demonstrations across 22 tasks in a single house environment. To reduce interference between tasks, we added task-specific learned embeddings. This phase used heavy proprioceptive dropout (60%) as well as an increased LoRA rank from 16 to 32 to increase expressivity.

Surprisingly, on many individual tasks, the generalist pretraining achieved lower validation loss than individual models trained specifically for that task. The diversity of the pre-training data provided a better foundation than narrow specialization. We also observed novel error-correction behavior. For instance, when struggling to pick up a box from an awkward angle, Arthur would reverse and reorient himself. This kind of behavior was not present in the training data and non-existent before attempting generalist pre-training.

### Phase 2: Per-Task Fine-Tuning

We then fine-tuned separate checkpoints for each task, reducing proprioceptive dropout to ~10% to enable precise manipulation. This gave us task-specific experts built on a general foundation.

The learned task embeddings and per-task fine-tuning were pragmatic solutions given tight time constraints, to quickly mitigate inter-task mixing. For actual real-world deployment, we would use more flexible approaches like task-contrastive training, which can generalize to novel tasks without pre-defined embeddings.

---

## The Temporal Awareness Discovery

Our next insight, we stumbled upon by chance.

The default control strategy we were working with is *temporal ensembling*: at every timestep, the model predicts a trajectory of future actions. Current predictions are then averaged with past ones to produce a smooth and dynamic trajectory. This seemed to obviously be the superior strategy, able to react dynamically to changing situations and make the best prediction at every time step.

On a whim, we tried switching to *receding horizon* control: predict a chunk of 50 actions, execute all of them, then replan. Intuitively, this seemed worse. Arthur essentially flies blind for a few seconds before readjusting.

In practice, however, on the subtask of getting through the door within our "moving boxes to storage" task, the success rate jumped from 30% to 100%.

Failed at Door (Temporal Ensemble)(2x)

Succeeded at Door (Receding Horizon)

Door navigation comparison

Failed to Grab (Temporal Ensemble)

Succeeded to Grab (Receding Horizon)

Object grasping comparison

We were stunned.

Here's our interpretation: temporal ensembling is like falling asleep and waking up every 0.3 seconds. The model never exercises a sense of agency. It plans, starts to act, then replans again at every timestep, second guessing itself. Each action, plagued by past inaccuracies. Whatever implicit knowledge the model has about trajectory coherence gets discarded with every replan.

In simple terms, Arthur had no sense of what it had done in the past, and no confidence in what it should do in the future.

Receding horizon control gives the model *agency*. It commits to a plan and executes it. This allows the model to leverage whatever understanding it has of action dynamics and sequential structure.

This goes beyond a hyperparameter finding. We think it points toward something deeper about what's missing in current VLA architectures: temporal awareness. The model has no explicit representation of where it is in a trajectory, no sense of what it was doing a moment ago beyond whatever is implicit in the observation. This sometimes manifests in behavior like carrying a box to the garage, forgetting why it's there in the first place, and leaving without putting it down.

We suspect this control mode switch is a bandaid fix that revealed a small glimpse of the deeper problems inherent in current control strategies. Arthur lacks lucidity in the way a human does when taking actions. Existing in a constant state of limbo without the context necessary to effectively execute long-range complex tasks. We're exploring techniques to better integrate a sense of lucidity and temporal awareness into the policy: control-mode aware training and learned re-planning triggers.

---

## Emergent Behaviors

As mentioned in the “Two-Phase Training” section, we observed emergent behaviors we never explicitly trained, from the generalist pre-training phase.

In earlier checkpoints, when Arthur failed to pick up an object, he would keep trying the same failed motion or give up entirely. Later checkpoints showed something different: failed grasps followed by backing up, repositioning, and retrying. Error correction emerged from scale.

Your browser does not support the video tag.Arthur failing a pickup, backing up, repositioning, and successfully grasping on the second attempt

This is the kind of capability that gives us optimism about the foundation model approach. We didn't engineer recovery behaviors and they weren't present in the dataset. They emerged from training on enough diverse data that the model learned to handle its own failures.

---

## High-Level Takeaway

What we eventually came to learn was very simple:

Our primary job is to remove the bottlenecks preventing the model from learning, then let it benefit from more data and more compute.

Proprioceptive dropout, action magnitude weighting, and intelligent resampling were our most impactful interventions—not because they made the model smarter, but because they removed obstacles that were preventing it from learning what was already in the data.

The temporal awareness finding points in a different direction: there are aspects of intelligent behavior that our current architectures may not be capturing. The model doesn't know where it is in time. It doesn't maintain a sense of trajectory or commitment to action. Receding horizon control gave us a clue, but further research is necessary.

---

## Future Directions

We're excited to continue this research along several fronts:

### Learned replanning

Rather than using a fixed control strategy, we'd like to include learned triggers during training so the model can learn when to stick to a trajectory, and when to replan. This would combine the benefits of temporal ensembling (continuous correction) and receding horizon (trajectory coherence).

### Longer temporal context

Instead of only supplying the current proprioceptive state, we want to provide a history of past actions and states. This would give the policy a sense of the trajectory it's executing, enable longer-horizon planning, and incorporate feedback from the environment over time.

### Joint world-model and policy training

Inspired by approaches like [UnifoLM-WMA](https://unigen-x.github.io/unifolm-world-model-action.github.io), training a world model alongside the policy could improve robustness, correction behavior, and consistency over long horizons. When we experimented with RL (shoutout to the [RLInf](https://github.com/RLinf/RLinf) team), simulator speed was a major bottleneck. Generative world models could help here by enabling efficient rollouts on standard GPU hardware without ray-tracing.

### Language adherence

Our current model sometimes drifts from the instructed task. Improving prompt following is critical for practical deployment. Approaches like task-contrastive training (contrastive loss on predicted actions across different task prompts for identical inputs) could encourage genuine prompt adherence.

---

## Acknowledgments

Thank you to the [Stanford BEHAVIOR Challenge](https://behavior.stanford.edu/challenge/index.html) organizers, who created this benchmark and guided us throughout the competition—especially Wensi Ai and Hang Yin, whose insights directly influenced some of our breakthroughs. Thank you also to [Physical Intelligence](https://www.physicalintelligence.company) for open-sourcing π0.5, this work builds directly on their foundation. Finally, thank you to [lium.io](https://lium.io) for the compute grant that helped kickstart our research.

Our team is excited to be attending NeurIPS in San Diego and the Humanoids Summit in SF. Looking forward to seeing you all there!

Questions or want to collaborate? Reach out at [[email protected]](https://merlyn-labs.com/cdn-cgi/l/email-protection#33505c5d47525047735e56415f4a5d1e5f5251401d505c5e)
