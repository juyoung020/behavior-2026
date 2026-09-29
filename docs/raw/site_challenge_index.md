# 🏆 2026 BEHAVIOR Challenge - BEHAVIOR

> 원본: https://behavior.stanford.edu/challenge/index.html
> 받은 시각: 2026-09-29 18:22 KST (2026-09-29 09:22 UTC)
> 이 파일은 `_scrape.py` 가 자동으로 받은 원문 보관본이다. HTML→Markdown 변환 중 서식은 달라질 수 있으나 문장은 원문 그대로다.

---
# 🏆 **2026 BEHAVIOR Challenge**

Join us for the second year of the BEHAVIOR Challenge: solve **100 full-length household tasks** in the realistic BEHAVIOR-1K environment. BEHAVIOR tests whether embodied agents can combine high-level reasoning, long-horizon navigation, and dexterous bimanual manipulation in house-scale scenes.

## Important Dates

- **Challenge Launch:** 07/02/2026
- **Submission Deadline:** 10/16/2026
- **Winners Announcement:** 11/04/2026

 [Submission Portal](https://behavior-1k-2026-challenge-leaderboard.hf.space/submit) [Submission Guidelines](https://behavior.stanford.edu/challenge/submission.html)

## Event Details

- **Event:** To be announced
- **Time:** To be announced
- **Location:** To be announced

 [Leaderboard](https://huggingface.co/spaces/behavior-1k/2026-challenge-leaderboard) [Evaluation & Rules](https://behavior.stanford.edu/challenge/evaluation.html)

## **Prize Pool**

 **🏆** Total Prize Pool **$11,000**

 **🥇** 1st Place **$5,000**

 **🥈** 2nd Place **$3,000**

 **🥉** 3rd Place **$2,000**

 **🌐** Outstanding Open Source **$1,000**

## **Challenge at a Glance**

| Tasks | 100 full-length household tasks |
|---|---|
| Environments | 7 scenes, including 4 new scenes |
| Evaluation track | One track using RGB + depth + proprioception |
| Demonstrations | 20,000 human teleoperation demos, 1,950 hours in total |
| Baselines | π0.5 (pi0.5) and GR00T N1.7 |
| Ranking metric | Average task success score with BDDL partial credit |

Detailed specifications live on the canonical challenge pages: [Dataset](https://behavior.stanford.edu/challenge/dataset.html), [Baselines](https://behavior.stanford.edu/challenge/baselines.html), [Evaluation and Rules](https://behavior.stanford.edu/challenge/evaluation.html), and [Submission Guidelines](https://behavior.stanford.edu/challenge/submission.html). Browse the full task list in the [Demo Gallery](https://behavior.stanford.edu/challenge/tasks/index.html).

## **Demonstration Data**

The challenge provides large-scale human teleoperation demonstrations for learning long-horizon household behaviors. The release includes RGB and depth observations, robot proprioception and actions, and skill/subtask annotations; the full dataset format and statistics are documented on the [Dataset](https://behavior.stanford.edu/challenge/dataset.html) page.

Demonstrations were collected with **JoyLo**, a whole-body teleoperation interface for controlling the robot base, torso, arms, and grippers. We thank [Simovation](https://www.linkedin.com/company/simovationinc/) for providing high-quality JoyLo teleoperation data in simulation.

## **Why Participate**

BEHAVIOR tasks go beyond short pick-and-place or navigation benchmarks. Agents must search across rooms, manipulate many objects, handle object state changes, and satisfy symbolic BDDL goal conditions after several minutes of autonomous execution.

The 2026 challenge is intended as a shared benchmark for testing robot foundation models, imitation learning, reinforcement learning, task and motion planning, memory systems, SLAM, and LLM-assisted policies under the same realistic evaluation protocol.

The tasks also exercise diverse object state changes and low-level skills, including opening, closing, pouring, wiping, spraying, attaching, toggling, cooking, and slicing.

## **Getting Started**

1. Join the [Discord community](https://discord.gg/bccR5vGFEx) for announcements and participant discussion.
2. Attend office hours every Monday, 5-6pm Pacific Time, over [Zoom](https://stanford.zoom.us/j/98056621630?pwd=G3JqHl6lWB0qnMAV3BZMJaWgoiXeqT.1).
3. Download the dataset and review the [dataset documentation](https://behavior.stanford.edu/challenge/dataset.html).
4. Start from the [π0.5 and GR00T N1.7 baseline pipelines](https://behavior.stanford.edu/challenge/baselines.html).
5. Run evaluation and prepare your submission using the [submission guidelines](https://behavior.stanford.edu/challenge/submission.html).

Whether you're a robotics veteran or just entering the field, we're here to support you.

## **BibTeX**

To cite BEHAVIOR-1K, please use:

```
@article{li2024behavior,
  title={Behavior-1k: A human-centered, embodied ai benchmark with 1,000 everyday activities and realistic simulation},
  author={Li, Chengshu and Zhang, Ruohan and Wong, Josiah and Gokmen, Cem and Srivastava, Sanjana and Mart{\'i}n-Mart{\'i}n, Roberto and Wang, Chen and Levine, Gabrael and Ai, Wensi and Martinez, Benjamin and Yin, Hang and Lingelbach, Michael and Hwang, Minjune and Hiranaka, Ayano and Garlanka, Sujay and Aydin, Arman and Lee, Sharon and Sun, Jiankai and Anvari, Mona and Sharma, Manasi and Bansal, Dhruva and Hunter, Samuel and Kim, Kyu-Young and Lou, Alan and Matthews, Caleb R. and Villa-Renteria, Ivan and Tang, Jerry Huayang and Tang, Claire and Xia, Fei and Li, Yunzhu and Savarese, Silvio and Gweon, Hyowon and Liu, C. Karen and Wu, Jiajun and Fei-Fei, Li},
  journal={arXiv preprint arXiv:2403.09227},
  year={2024}
}
```

## **Sponsors**

High-quality simulation data provided by Simovation.

We gratefully acknowledge the support of our sponsors who make this challenge possible:

 [![Simovation](https://behavior.stanford.edu/assets/challenge_2025/simovation_logo.png)](https://www.linkedin.com/company/simovationinc/) [![IMDA](https://behavior.stanford.edu/assets/challenge_2025/imda_logo.png)](https://www.imda.gov.sg/) [![Stanford HAI](https://behavior.stanford.edu/assets/challenge_2025/hai_logo.png)](https://hai.stanford.edu/) [![Schmidt Family Foundation](https://behavior.stanford.edu/assets/challenge_2025/schmidt_family_foundation_logo.png)](https://tsffoundation.org/)
