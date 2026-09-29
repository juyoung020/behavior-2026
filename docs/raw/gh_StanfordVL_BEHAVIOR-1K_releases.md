# StanfordVL/BEHAVIOR-1K 태그·릴리스 노트

> 원본: https://github.com/StanfordVL/BEHAVIOR-1K/releases
> 받은 시각: 2026-09-29 18:22 KST (2026-09-29 09:22 UTC)
> 이 파일은 `_scrape.py` 가 자동으로 받은 원문 보관본이다. HTML→Markdown 변환 중 서식은 달라질 수 있으나 문장은 원문 그대로다.

---
## 태그 목록 (GitHub API)

- `v3.9.3-post1` (bd049de)
- `v3.9.2` (b197991)
- `v3.9.1` (26f2c7e)
- `v3.9.0` (6559858)
- `v3.7.2` (88454bd)
- `v3.7.1` (79f0f5f)
- `v3.7.0` (44186f1)
- `v1.1.1` (d15de67)
- `v1.1.0` (032c9b2)
- `v1.0.0` (282adda)
- `v0.2.1` (aa8b870)
- `v0.2.0` (452b28f)
- `v0.1.0` (c38cfdd)
- `v0.0.6` (c77de3e)
- `v0.0.5` (915add3)
- `v0.0.4` (cb70ec4)
- `v0.0.3` (8691b02)
- `v0.0.2` (c2f3eeb)
- `v0.0.1` (61323e4)

## 릴리스 노트 (최근 10개)

### BEHAVIOR-1K v3.9.3 post 1 — 태그 `v3.9.3-post1`

게시: 2026-09-29T03:39:11Z · https://github.com/StanfordVL/BEHAVIOR-1K/releases/tag/v3.9.3-post1

## What's Changed
* Fix challenge update list formatting by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2338
* Retry Isaac Sim wheel downloads and fail loudly on a bad one by @rakhimovv in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2336
* update handles after clearing particles by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2342
* Move LICENSE to repo root for visibility. by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2350
* Merge vector into main by @Andorexad in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2329
* Add type checking to CI by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2330
* Add mirrored attachment point for poster  by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2349
* Bump version to 3.9.3 by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2353
* Fix ToggledOn state access in evaluation light synchronization by @mengxyokok in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2357

## New Contributors
* @rakhimovv made their first contribution in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2336
* @mengxyokok made their first contribution in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2357

**Full Changelog**: https://github.com/StanfordVL/BEHAVIOR-1K/compare/v3.9.2...v3.9.3-post1

### BEHAVIOR-1K v3.9.2 — 태그 `v3.9.2`

게시: 2026-08-24T23:54:06Z · https://github.com/StanfordVL/BEHAVIOR-1K/releases/tag/v3.9.2

## What's Changed
* Add workflow to manually dispatch docker build from non-main branches by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2315
* Update eval script to use rooms specified in B100_task metadta by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2331
* Refresh handles before loading observations in RGBDFullResWrapper by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2332
* Add August 2026 challenge update by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2337


**Full Changelog**: https://github.com/StanfordVL/BEHAVIOR-1K/compare/v3.9.1...v3.9.2

### BEHAVIOR-1K v3.9.1 — 태그 `v3.9.1`

게시: 2026-07-28T23:18:43Z · https://github.com/StanfordVL/BEHAVIOR-1K/releases/tag/v3.9.1

## What's Changed
* hf download update by @13RENDA in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2309
* Add constant options for torch threads nums by @hirobon1690 in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2303
* edit index.md sponsors by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2314
* Fix R1Pro base qvel frame and document demo updates by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2320

## New Contributors
* @hirobon1690 made their first contribution in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2303

**Full Changelog**: https://github.com/StanfordVL/BEHAVIOR-1K/compare/v3.9.0...v3.9.1

### BEHAVIOR-1K v3.9.0 — 태그 `v3.9.0`

게시: 2026-07-03T06:44:27Z · https://github.com/StanfordVL/BEHAVIOR-1K/releases/tag/v3.9.0

## What's Changed
* Add code/report release column to leaderboard by @hang-yin in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1938
* Feat/new robots by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1890
* Fix/misc by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1944
* Update Docker setup to create image from scratch, use setup script by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1945
* Include conda activation on gh actions entrypoint by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1961
* Use default environment inside actions runner by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1960
* Refactor robot subclasses into single robot class + YAML definition files by @Andorexad in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1905
* Hang's freedom by @hang-yin in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1967
* Set fixed base for non mobile robots to be true by @Andorexad in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1970
* Fix example demos: tensor handling, keyboard teleop   by @kmy17518 in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1952
* Fix curobo holonomic joint limit by @hang-yin in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1937
* Convert visual and collision geom prims to plain geom prim by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1973
* Fix profiling by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1980
* Refactor assisted grasping and state restoration flow by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1982
* Fix test failures: velocity returns, robot YAML defs, scene pose cache by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1981
* Small fix for robot_control_example quickstart behavior by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1985
* Small update to robot definition schema to make controllers optional by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1986
* Small docs nits by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1987
* Revert "Small update to robot definition schema to make controllers optional" by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1988
* Fix robot get_position_orientation passthrough params by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1989
* disable test_primitives for tiago for 50 series GPU by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1993
* Update test_robot_states_flatcache by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1991
* Hygiene: update ui_utils omnigibson logo to avoid deprecationwarning by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1995
* Update CI tests by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1994
* Refactor: merge BaseObject with StatefulObject by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1996
* Fix kinematic mixin regression from object state refactor by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2003
* Add lenient timeout per-step for each test by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2002
* test different test report by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1997
* Move light object and primitive object to inherit from USDObject by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2000
* Remove fixed_base from LightObject by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2008
* Fix / disable a few tests from the examples as tests by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2004
* Fix issues in robot tests and examples by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2011
* Persistent-cached, step-updated, robust-to-sleep RigidContactAPI for contact checking by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1977
* Make tests blocking by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2015
* Temporarily disable failing tests on main and add a few more test fixes by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2022
* Multi-step contact caching by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2024
* Merge USDObject with BaseObject by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2014
* Controller refactor for batched compute by @Andorexad in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1990
* Move articulation root computing to preload by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2027
* Use jparse for ik. Remove unused non-batch compute by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2033
* Fix try/catch statements in windows setup script by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2031
* Skip contact checking when objects in scene are kinematic-only or fixed by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2034
* Update to Isaac Sim 5.1 by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1992
* Change og_test to only load objects as needed by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2038
* Update Dockerfile to Isaac 5.1 by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2043
* CuRobo fix by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2046
* Fix RigidContactAPI: don't create None views if scene contains no objects by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2051
* Fix curobo and floor plane collsion by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2052
* Fix PoseAPI invalidation issues by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2055
* Small fix for draw_bounding_box test by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2058
* Refactor simulator simcontext as instance instead of subclass by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2059
* Fix scale for dataset objects with bounding boxes by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2064
* add play/stop in USD use case by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2062
* Fix/sampling clean up by @hang-yin in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1886
* Fix physics context callback by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2069
* fixed left, right typo on _update_hand_tracking_data in omnigibson/utils/teleop_utils.py by @asattiraju1 in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2071
* Fix articulation root and post_load operations by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2070
* Fix test_object_states CI failures and disable broken particle tests with tracking issues by @yalcintur in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2067
* Refactor data collection to support LeRobot playback and HDF5 formats by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2074
* Fix primitives scale by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2076
* Fix lerobot commit hash by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2077
* Fix/lerobot by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2078
* Clean up and reorganize joylo scripts by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2079
* Use mesh instead of sphere for particles by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2075
* Update joylo documentation by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2086
* Refactor sampling by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2087
* Refactor metrics to use MetricBase and add streaming replay support by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2084
* Enforce Fabric enabled, remove option to disable by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2091
* Add agent md file by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2092
* FIx Assisted Grasp Window from 3.0 to 0.7 and use physics _dt instead of sim_step_dt by @yalcintur in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2072
* Add documentation links to claude/agents.md by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2095
* Fix depth writing by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2096
* fix ag by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2098
* BDDL refactor: unified KnowledgeBase API, explicit predicates, and clean task interface by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2040
* Enforce USD-Fabric sync via editing_usd() context by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2094
* bump stepping in one transition_rules test to reduce flakiness by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2107
* Add snapshotting tests by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2105
* add trunk integration by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2108
* Add cache=true options to jit compile by @yalcintur in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2101
* Include pycache in gitignore by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2112
* Fix Trunk integration by checking out source first  by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2113
* Refactor sampling scripts to use outputs/ folder and remove online mode by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2097
* Fix: clean PhysX step/event subscriptions during og.clear to prevent stale callbacks by @Andorexad in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2114
* Fix bugs by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2118
* Fix Test Report to fail if jobs are cancelled by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2123
* Serialize assisted grasping state by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2119
* Skip viewport creation in headless mode for VisionSensor by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2117
* Update robot import, add kinova and ur5e yaml by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2019
* Fix synsets index.html page on knowledgebase website by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2128
* Use appropriate pose setters for articulations, force physx-to-Fabric sync, and remove PoseAPI by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2127
* reinstate test_snapshots with 99% accuracy by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2129
* Fix unresponsive goal-condition text color updates by @yalcintur in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2132
* Fix rollback and checkpointing during teleop by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2135
* Add floor plan visualizer by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2136
* Update inside setter by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2131
* Add utils for sampling by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2134
* Minor changes to sampling pipeline by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2137
* Sampling Fixes by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2141
* Parameterize test_grasping_mode over grasping modes by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2120
* cache link.volume once and reuse after by @yalcintur in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2099
* Per-robot RigidContactView for assisted-grasp contact positions by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2116
* Add collision and joint state checks into Inside sampling by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2143
* Fix validate task in multiply script by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2145
* Fix qa failed grasp by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2147
* Update cook brisket and freeze fruit bddl by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2148
* Additional fixes for autogenerate script by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2150
* Fix replay transition ordering by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2151
* Update tidying_living_room, sorting_bottles_cans_and_paper bddl  by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2149
* Update cook brisket bddl by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2157
* sample ground particles to be reachable by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2160
* Fix robot height in sample_robot_pose by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2158
* Modify bddl for sweeping garage and stacking wood by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2159
* fix robot sampling pose script by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2161
* Add interactive prompting for choosing the episode to replay by @yalcintur in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2165
* Fix robot keep still by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2164
* Update store produce bddl by @yalcintur in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2171
* Update make rose centerpieces bddl by @yalcintur in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2168
* Update organizing art supplies bddl by @yalcintur in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2170
* Update installing smoke detectors bddl by @yalcintur in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2167
* Update halve an egg bddl by @yalcintur in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2169
* Update re_shelving_library_books bddl by @kmy17518 in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2172
* Fix JoyLo Home reset after slicing objects by @yalcintur in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2173
* Update bddl and teleop cfg for bringing paper to recycling by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2175
* Add attachment visual guides for teleop by @yalcintur in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2176
* bddl for turning_out light and unloading the car by @13RENDA in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2180
* Remove duplicate logging from OmniGibson by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2182
* Enable macro physical particle test by @yalcintur in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2184
* Fix asset export pipeline to run on 5.1, implement first round of requested fixes from sampling team by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2178
* Enable RT2 and fractional opacity by @yalcintur in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2183
* Add helper script to scan and reboot joylo motors by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2181
* New dataset RC without visual-only error, added version checking by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2189
* Bump pytest from 7.2.0 to 9.0.3 in /asset_pipeline by @dependabot[bot] in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2185
* Replace context managers for play/stop/pause by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2162
* Remove more og.sim.stopped context managers by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2191
* Update bddl for tidying living room by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2192
* update bddl for boxing food, cleaning branches, thawing frozen food, … by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2193
* update vacuum mass by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2201
* Small change to replay_data, auto-select when only one episode by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2207
* fix teleop cfg for new scenes by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2204
* Format bounding box sizes for super-thin objects by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2203
* Fix deleting particle group by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2205
* Change USD hash warnings from warning to debug by @yalcintur in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2208
* Use th.as_tensor for bounding box construction by @yalcintur in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2210
* Fix sampling to use reachability map by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2206
* update trash can mass by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2209
* Clear stale assisted-grasp state when removing objects by @yalcintur in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2214
* Refresh PhysX-Fabric bindings when restoring re-added scene objects by @yalcintur in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2212
* Remove visual only objects from ignore set for is_in_contact by @yalcintur in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2216
* Update bddl for clean up broken glass, clean rusty tools, packing meal by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2213
* Fix syncing BDDL github action by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2218
* Fix sampling scripts for scenes that have capital letters by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2219
* Release new asset RC containing scene, texture fixes by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2221
* Revert "update trash can mass" by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2222
* Lower mass on potted plant by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2223
* update bringing paper to recycling, cleaning up branches bddl by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2225
* Add overlap based failed grasping check  by @yalcintur in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2226
* Update dispose_of_glass bddl by @yalcintur in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2227
* Add rs_int and restaurant diner to teleop config by @yalcintur in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2228
* [tmp] remove objects from combined room object list by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2230
* update polishing_shoes bddl by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2231
* update clean rusty tools bddl by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2229
* Run combined room object list and some file manifests by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2234
* Add floor plan room picker for task sampling by @yalcintur in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2177
* Actually build knowledgebase in pull-sheets workflow check by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2235
* Visual QA for task sampling by @yalcintur in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2196
* Update put_together_a_basic_pruning_kit task by @kmy17518 in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2239
* Brenda/sampling bddl by @13RENDA in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2241
* New dataset RC with scene fixes by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2242
* Installing a modem bddl update by @yalcintur in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2243
* Update bddl for tidying bathroom and carrying out garden furniture by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2244
* Update BDDL for store_batteries and store_honey by @yalcintur in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2246
* update dirty dishes bddl by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2248
* update several bddl files by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2249
* Update bddl by @kmy17518 in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2250
* Update laying_tile_floors bddl by @kmy17518 in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2253
* Fix task instruction by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2256
* Brenda/sampling bddl by @13RENDA in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2254
* Brenda/sampling bddl by @13RENDA in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2263
* Update sorting_books_on_shelf bddl by @kmy17518 in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2266
* Bddl change for tasks store_produce and cook_broccolini by @yalcintur in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2269
* Update garden furniture bddl by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2273
* update cocktail party bddl to spawn on cabinet by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2271
* update bddl for packing meal task by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2275
* adjust packing meal bddl to only have 3 burgers by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2276
* Correctly compute updated_state_objects when multiple objects change temperature by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2272
* Do not mark object as sleeping if it has moved by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2274
* Brenda/sampling bddl by @13RENDA in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2278
* Fix installing a fax machine bddl by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2277
* Handle multiple types of line breaks in the task csv file by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2279
* Update cook pie bddl by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2281
* fix docker build by @stefren in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2282
* Replay & Eval fix by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2260
* Add use last demo functionality by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2283
* Fix lightning for turning off all lights before sleep by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2286
* Disable crash reporter by @yalcintur in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2290
* Release asset version 3.9.0 by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2293
* Bump version to 3.9.0 by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2294
* Website Update by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2291
* Create work in progress (WIP) folder for examples by @yalcintur in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2289
* Update Eval by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2285
* Fix installation by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2297
* Fix custom robot import pipeline for Isaac Sim 5.1 by @yalcintur in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2292
* fix warp version for curobo install by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2298
* Decrese port requirement to 50. Decrease timeout to 1.5x by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2299
* Update baselines.md by @kmy17518 in https://github.com/StanfordVL/BEHAVIOR-1K/pull/2300

**Full Changelog**: https://github.com/StanfordVL/BEHAVIOR-1K/compare/v3.7.2...v3.9.0

### BEHAVIOR-1K v3.7.2 — 태그 `v3.7.2`

게시: 2025-12-15T19:07:59Z · https://github.com/StanfordVL/BEHAVIOR-1K/releases/tag/v3.7.2

This is the post release of the 2025 BEHAVIOR Challenge at NeurIPS. For more information about the challenge, check out https://behavior.stanford.edu/challenge

## What's Changed
* Feat/doc update by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1833
* Fix/0923 by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1840
* Switch back from USDZ to USD due to load time and caching issues by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1843
* No credit for just achieving initial state by @hang-yin in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1842
* Update challenge submission docker by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1849
* Fix rigid prim volume, fix typo in doc by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1853
* Fix/doc by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1856
* Add the announcements on the challenge front page, and fix some typos. by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1857
* Add clarification by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1860
* Add unique skill list to doc by @hang-yin in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1866
* Move appdata directory into OmniGibson and make it configurable by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1858
* Update heavy robot wrapper. Fix eval bug by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1867
* Submission Update by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1880
* Fix task_to_task_index by @Tavish9 in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1885
* Update hidden test instance handling by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1887
* Update joylo doc and momagen link by @hang-yin in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1889
* Update by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1894
* Update doc by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1899
* BDDL -> BDDL3 by @hang-yin in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1906
* Update announcements for 2025 BEHAVIOR Challenge by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1909
* Fix torch by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1907
* Specify version for PyQt6 dependency by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1914
* Make action receiving more robust for ws by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1915
* Update leaderboard by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1908
* Fix/ag by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1919
* Fix pcd downsample = None by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1922
* Feat/leaderboard by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1921
* update leaderboard title by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1924
* Run ruff format on whole repo by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1926
* Update venue by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1928
* fix submission typo by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1929
* Commit evaluation system used for challenge by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1927
* Update challenge zoom link by @hang-yin in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1930
* Fix/doc by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1931
* 3.7.2 version bump by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1933
* Fix/installation by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1934

**Full Changelog**: https://github.com/StanfordVL/BEHAVIOR-1K/compare/v3.7.1...v3.7.2

### BEHAVIOR-1K v3.7.1 — 태그 `v3.7.1`

게시: 2025-09-12T23:38:25Z · https://github.com/StanfordVL/BEHAVIOR-1K/releases/tag/v3.7.1

In this release, we fixed various bugs and added more features/supports for the 1st BEHAVIOR challenge. Visit https://behavior.stanford.edu/ for more information!

## Updates
* Removed `grasp_left` and `grasp_right` from robot proprioception and updated the [demo dataset](https://huggingface.co/datasets/behavior-1k/2025-challenge-demos) accordingly. 
  * Now, the dimension of `observation.state` in the dataset is 256 instead of 258
* Updated [task-instances](https://huggingface.co/datasets/behavior-1k/2025-challenge-task-instances) dataset, included 2 more metadata files for data replay
* Fixed library errors (`av`, `gspread`, `pandas`, `cffi`) during installation
* Added `RGBLowResWrapper` and set it to be the default wrapper during evaluation rollout
* Added GOP-level randomization to `BehaviorLeRobotDataset` for faster data access during training
* Added point cloud modality to vision sensor
* Updated camera relative pose to base implementation, changed `n_render_iterations` to 1 during evaluation rollout
* Fixed bugs in demo replay script
* Added `NO_OMNI_LOG` macro
* Updated challenge rules and documentations:
   * Changed evaluation rollout number from 20 * 5 to 10 * 1
   * Specified robot base global pose is not allowed in standard track
   * Added video recordings as part of the submission

## To migrate from `v3.7.0` to `v3.7.1`

* If you already downloaded the [demo dataset](https://huggingface.co/datasets/behavior-1k/2025-challenge-demos), please update this by syncing local dataset with HF
  ```
  ds = BehaviorLeRobotDataset(
      repo_id="behavior-1k/2025-challenge-demos",
      root=PATH_TO_DATASET_ROOT,
      tasks=LIST_OF_TASK_NAMES,
      download_videos=False,
      local_only=False,
      force_cache_sync=True
  )
  ```
  Specify task names under `tasks`, or set to `None` if you want to update all tasks.
  If you are only downloading the dataset, you can safely exit the program after seeing something like
  ```
  Generating train split: 23346708 examples [03:25, 112135.76 examples/s]
  ```
* Pull the latest code from `v3.7.1` ([79f0f5](https://github.com/StanfordVL/BEHAVIOR-1K/commit/79f0f5fb74a6063b67c6c0ce64f85fd96f2ba429)), then *within* the behavior conda environment run:
  ```
  ./setup.sh --bddl --omnigibson --dataset
  ``` 

**Full Changelog**: https://github.com/StanfordVL/BEHAVIOR-1K/compare/v3.7.0...v3.7.1

## What's Changed
* Fix: broken links on b100 doc by @hang-yin in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1820
* Update eval by @wensi-ai in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1821


**Full Changelog**: https://github.com/StanfordVL/BEHAVIOR-1K/compare/v3.7.0...v3.7.1

### BEHAVIOR-1K v3.7.0 — 태그 `v3.7.0`

게시: 2025-09-02T20:08:46Z · https://github.com/StanfordVL/BEHAVIOR-1K/releases/tag/v3.7.0

In this release, we add support for Isaac Sim 4.5, and support all features for the BEHAVIOR challenge including teleoperation and evaluation. Visit https://behavior.stanford.edu for more information!

## What's Changed
* Remove Python install in tests to avoid weird dependency issue by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/965
* Update issue templates by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/971
* Re-create dev docker image and remove requirements-dev.txt by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/979
* Remove OmniGibson from Docker dev image by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/985
* fix mixing rule transition rule tests by @ChengshuLi in https://github.com/StanfordVL/BEHAVIOR-1K/pull/938
* Unpin dependency versions by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/988
* Fix transform utils test by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/989
* Fix/robot self collision by @cremebrule in https://github.com/StanfordVL/BEHAVIOR-1K/pull/939
* Make following arm targets in empty_action optional by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/963
* Fix Key Array Caching by @hang-yin in https://github.com/StanfordVL/BEHAVIOR-1K/pull/962
* Address Contact Issues by @hang-yin in https://github.com/StanfordVL/BEHAVIOR-1K/pull/980
* Feat/controller dependencies by @cremebrule in https://github.com/StanfordVL/BEHAVIOR-1K/pull/990
* Add optional dependencies for OMPL and curobo, include both in Docker image by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1032
* fix torch compiled transform utils by @ChengshuLi in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1033
* Update Docker to uninstall cuda toolkit & start pushing actions runner image on CI too by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1034
* Feat/curobo improved by @ChengshuLi in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1019
* toggle async rendering to avoid GUI freezing by @cremebrule in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1046
* fix flaky tests by @ChengshuLi in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1042
* Fix profiling, remove unnecessary CI steps by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1009
* External asset + robot import script by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/902
* Optional Numpy Compute Backend by @cremebrule in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1051
* Enable and comply with ruff linter by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1065
* Reduce numba requirement version to match Isaac Sim by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1076
* auto robot import z offset by @cremebrule in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1079
* Fix various controller issues by @ChengshuLi in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1073
* Integrate cuRobo into action primitives & primitives refactoring by @hang-yin in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1027
* Update download artifact for CI workflow by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1084
* VR support for Data Wrapper by @hang-yin in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1061
* fix cloth loading problems: caused by non-watertight, and potential multiple pieces by @Jianghanxiao in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1063
* Feat/robot fixes by @cremebrule in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1075
* make robots always call get/set_joint_positions/velocities rather tha… by @ChengshuLi in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1090
* New cloth mechanisms, fixes to asset import, new meta link format, and more by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1087
* Fix meta link geom prim path inference by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1123
* Fix meta link path and jitscript regressions from latest PRs by @cgokmen in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1125
* Fix/playback scale by @cremebrule in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1119
* Fixes/curobo particle by @ChengshuLi in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1120
* add automatic pre-processing of wildcard bddl instances by @cremebrule in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1150
* add checkpointing mechanism during demo collection by @cremebrule in https://github.com/StanfordVL/BEHAVIOR-1K/pull/1152
* Upgrade to isaac sim 4.2 & VR by @hang-yin in https://github.com/StanfordVL/BEHAVIOR-1K/pull/906

**Full Changelog**: https://github.com/StanfordVL/BEHAVIOR-1K/compare/v1.1.1...v3.7.0

### OmniGibson v1.1.1 — 태그 `v1.1.1`

게시: 2024-10-04T05:21:21Z · https://github.com/StanfordVL/BEHAVIOR-1K/releases/tag/v1.1.1

This release contains the below bugfixes to bugs affecting installation, particle systems, action primitives, profiling, and segmentation images:

## What's Changed
* Fix missing rename behavior in old glibc versions by @cgokmen in https://github.com/StanfordVL/OmniGibson/pull/927
* Fix pynvml for profiling VRAM by @hang-yin in https://github.com/StanfordVL/OmniGibson/pull/928
* Enforce flatcache off when using micro particle systems by @hang-yin in https://github.com/StanfordVL/OmniGibson/pull/930
* Check for missing test xml by @hang-yin in https://github.com/StanfordVL/OmniGibson/pull/913
* Instance Segmentation Refactor by @hang-yin in https://github.com/StanfordVL/OmniGibson/pull/932
* Fix transition rule and cuRobo tests by @ChengshuLi in https://github.com/StanfordVL/OmniGibson/pull/936
* action primitives bugfix for no op action IK controller by @yyf20001230 in https://github.com/StanfordVL/OmniGibson/pull/933


**Full Changelog**: https://github.com/StanfordVL/OmniGibson/compare/v1.1.0...v1.1.1

### OmniGibson v1.1.0 — 태그 `v1.1.0`

게시: 2024-10-02T03:29:04Z · https://github.com/StanfordVL/BEHAVIOR-1K/releases/tag/v1.1.0

We are excited to announce the latest release of OmniGibson, bringing significant enhancements to our simulation platform for embodied AI research and development.

## Highlights

- **Fast Installation with pip**: You can now install OmniGibson and Isaac Sim via pip! For detailed
  installation instructions, please visit our [official documentation](https://behavior.stanford.edu/omnigibson/getting_started/installation.html).
- **Multiple/Parallel Environments**: Support for reinforcement learning with optimized performance
  by running multiple environments at once as in Isaac Gym.
- **Data Collection and Playback**: You can now use our data collection environment wrappers for demonstration data.
- **Primitive Skill Library**: We now provide more robust action primitives and faster IK, with primitives
  now covered under integration tests for reliability.
- **Torch Integration**: OmniGibson now works exclusively with `torch` tensors instead of `numpy` arrays, in 
  anticipation of full GPU-based simulation and data pipeline.
- **cuRobo Integration**: OmniGibson now supports some accelerated motion planning capabilities from curobo.
  This integration will be extended in the near future to fully remove the OMPL dependency in primitives.
- **Scene Graph Generation**: OmniGibson can now generate scene graphs, compatible with single or multiple agents.
- **New Robots**: We support a number of new robots and provide improvements for our existing robots.
- **Documentation Overhaul**: Detailed OG modules and tutorials are now provided in the documentation.


## Breaking Changes

- OmniGibson now uses `torch` instead of `numpy` as is numerical computation backend, in anticipation of
  full GPU-based simulation features that we plan to include in the next release. This means that users
  now need to provide torch tensors when calling APIs instead of numpy arrays.
- OmniGibson is no longer compatible with Isaac Sim 2023.1.1 - we recommend upgrading to 4.1. __OmniGibson
  is not yet compatible with Isaac Sim 4.2.__
- Local pose APIs have been removed and unified into the `set_position_orientation` and `get_position_orientation`
  APIs with the `frame` argument.
- Absolute prim paths may no longer be provided when creating objects and prims. You will need to provide
  a relative prim path which will be applied on top of the object's scene's prim, in order to support
  multiple environments when necessary.
- Inverse Kinematics controllers have been updated to use a local approximation ("differential IK")
  instead of solving the full IK problem on every step, providing significantly better speed especially
  in multi-environment contexts.


We encourage all users to upgrade to this latest version to take advantage of these improvements and new features. As always, we welcome your feedback and contributions to help make OmniGibson even better.

## Merged PRs
* Replicator hot fix by @hang-yin in https://github.com/StanfordVL/OmniGibson/pull/667
* Optimize segmentation remapping performance by @hang-yin in https://github.com/StanfordVL/OmniGibson/pull/669
* Remove dev docker image, add colab image & improve example/installation usability by @cgokmen in https://github.com/StanfordVL/OmniGibson/pull/670
* Black formatter by @hang-yin in https://github.com/StanfordVL/OmniGibson/pull/672
* Add black precommit hook & use pre-commit CI for linter by @cgokmen in https://github.com/StanfordVL/OmniGibson/pull/673
* Check Isaac Sim version by @hang-yin in https://github.com/StanfordVL/OmniGibson/pull/680
* Fix tests for v1.0.0 asset release by @ChengshuLi in https://github.com/StanfordVL/OmniGibson/pull/696
* Update profiling by @wensi-ai in https://github.com/StanfordVL/OmniGibson/pull/678
* Update intrinsic matrix by @hang-yin in https://github.com/StanfordVL/OmniGibson/pull/692
* Fix primitives tests by @sujaygarlanka in https://github.com/StanfordVL/OmniGibson/pull/701
* Feat/docs update v2 by @cremebrule in https://github.com/StanfordVL/OmniGibson/pull/693
* Fix/general improvements by @cremebrule in https://github.com/StanfordVL/OmniGibson/pull/698
* Run tests faster (and make them easier to re-run) by splitting them up by @cgokmen in https://github.com/StanfordVL/OmniGibson/pull/682
* Make the README test badge show main instead of og-develop by @cgokmen in https://github.com/StanfordVL/OmniGibson/pull/675
* Remove global config by @hang-yin in https://github.com/StanfordVL/OmniGibson/pull/702
* Require positive scales and validate local transforms by @cgokmen in https://github.com/StanfordVL/OmniGibson/pull/636
* Add and apply isort by @cgokmen in https://github.com/StanfordVL/OmniGibson/pull/674
* lazy access of gm values by @hang-yin in https://github.com/StanfordVL/OmniGibson/pull/631
* remove friction override for tiago by @cremebrule in https://github.com/StanfordVL/OmniGibson/pull/710
* fix small seg semantic id bug by @cremebrule in https://github.com/StanfordVL/OmniGibson/pull/722
* Primitives merge to rl by @cgokmen in https://github.com/StanfordVL/OmniGibson/pull/728
* Merge experimental RL updates into Hang's PR by @cgokmen in https://github.com/StanfordVL/OmniGibson/pull/729
* add flexibility to bddl sampling by @cremebrule in https://github.com/StanfordVL/OmniGibson/pull/732
* Merge franka family by @wensi-ai in https://github.com/StanfordVL/OmniGibson/pull/725
* fix render_product syncing when updating camera width / height by @cremebrule in https://github.com/StanfordVL/OmniGibson/pull/735
* Parallel environments support & RL toolkit by @cgokmen in https://github.com/StanfordVL/OmniGibson/pull/699
* [WIP] RL training infra for primitives by @sujaygarlanka in https://github.com/StanfordVL/OmniGibson/pull/282
* Fix geodesic_reward==None bug by @hang-yin in https://github.com/StanfordVL/OmniGibson/pull/718
* Compatibility fixes for Isaac Sim 4.0.0 by @cgokmen in https://github.com/StanfordVL/OmniGibson/pull/776
* Revive transition rules by @ChengshuLi in https://github.com/StanfordVL/OmniGibson/pull/781
* Update Docker images to Isaac Sim 4.0.0, add both-version compatibility to kit files by @cgokmen in https://github.com/StanfordVL/OmniGibson/pull/778
* fix typo in controller base by @ChengshuLi in https://github.com/StanfordVL/OmniGibson/pull/788
* Fix gymnasium flatdim by @hang-yin in https://github.com/StanfordVL/OmniGibson/pull/792
* Doc update multiple envs by @hang-yin in https://github.com/StanfordVL/OmniGibson/pull/789
* Improve documentation for Tasks, Save/Load, BEHAVIOR Tasks and Knowledgebase by @ChengshuLi in https://github.com/StanfordVL/OmniGibson/pull/795
* Cem docs by @cgokmen in https://github.com/StanfordVL/OmniGibson/pull/798
* Add import robot and speed optimization tutorials by @wensi-ai in https://github.com/StanfordVL/OmniGibson/pull/793
* Some quick docs fixes by @cgokmen in https://github.com/StanfordVL/OmniGibson/pull/799
* Feat/doc update josiah by @cremebrule in https://github.com/StanfordVL/OmniGibson/pull/794
* Feat/auto robot idx by @cremebrule in https://github.com/StanfordVL/OmniGibson/pull/797
* Remove outdated examples README, fix geodesic potential, revive behavior task test by @ChengshuLi in https://github.com/StanfordVL/OmniGibson/pull/804
* Improve ControllableObjectViewAPI to work with different robot types by @cgokmen in https://github.com/StanfordVL/OmniGibson/pull/803
* Fix scene graph by @hang-yin in https://github.com/StanfordVL/OmniGibson/pull/740
* Clean up launch / clear interface and fix some bugs by @cgokmen in https://github.com/StanfordVL/OmniGibson/pull/809
* Support OmniGibson and Isaac Sim installation via pip by @yyf20001230 in https://github.com/StanfordVL/OmniGibson/pull/796
* Reorganize docs and fix some small bugs by @cgokmen in https://github.com/StanfordVL/OmniGibson/pull/813
* Fix draw boudning box example by @hang-yin in https://github.com/StanfordVL/OmniGibson/pull/816
* Fix two controller issues by @cgokmen in https://github.com/StanfordVL/OmniGibson/pull/817
* Documentation for remote streaming by @hang-yin in https://github.com/StanfordVL/OmniGibson/pull/712
* isaacsim new version support and windows installation fix by @yyf20001230 in https://github.com/StanfordVL/OmniGibson/pull/829
* Bbox test by @hang-yin in https://github.com/StanfordVL/OmniGibson/pull/821
* Add robot load and drive test by @hang-yin in https://github.com/StanfordVL/OmniGibson/pull/819
* Support multiple agent in scene graph by @hang-yin in https://github.com/StanfordVL/OmniGibson/pull/836
* Data Collection / Playback Environments by @cremebrule in https://github.com/StanfordVL/OmniGibson/pull/811
* fix jacobian bug by @cremebrule in https://github.com/StanfordVL/OmniGibson/pull/838
* fix demo collection bugt by @wensi-ai in https://github.com/StanfordVL/OmniGibson/pull/840
* NumPy -> Torch by @hang-yin in https://github.com/StanfordVL/OmniGibson/pull/780
* Disable gravity compensation by @cgokmen in https://github.com/StanfordVL/OmniGibson/pull/855
* Fix base footprint offset not applied in ControllableObjectViewAPI by @cgokmen in https://github.com/StanfordVL/OmniGibson/pull/857
* add outer loop to simulator by @cremebrule in https://github.com/StanfordVL/OmniGibson/pull/864
* Add differential IK option to do IK without Lula by @cgokmen in https://github.com/StanfordVL/OmniGibson/pull/858
* Fix/osc by @cremebrule in https://github.com/StanfordVL/OmniGibson/pull/863
* do not require floor plane by @cremebrule in https://github.com/StanfordVL/OmniGibson/pull/869
* remove lula from IK controller; always use differential IK by @ChengshuLi in https://github.com/StanfordVL/OmniGibson/pull/876
* fix clothification for scaled cloth by @ChengshuLi in https://github.com/StanfordVL/OmniGibson/pull/860
* Torch mini followup - bug fixes by @hang-yin in https://github.com/StanfordVL/OmniGibson/pull/867
* Unify world/local/scene pose getters and setters by @yyf20001230 in https://github.com/StanfordVL/OmniGibson/pull/786
* update dependency versions, remove torchvision dep by @cremebrule in https://github.com/StanfordVL/OmniGibson/pull/881
* New robot (R1 & A1) with robot fixes by @wensi-ai in https://github.com/StanfordVL/OmniGibson/pull/868
* Feat/curobo by @cremebrule in https://github.com/StanfordVL/OmniGibson/pull/878
* Various small fixes by @cgokmen in https://github.com/StanfordVL/OmniGibson/pull/889
* limit numpy <2.0 by @cremebrule in https://github.com/StanfordVL/OmniGibson/pull/894
* Bug fixes by @hang-yin in https://github.com/StanfordVL/OmniGibson/pull/885
* add sanity check for object-level scale, robot can accept scale as Iterable by @ChengshuLi in https://github.com/StanfordVL/OmniGibson/pull/896
* Replicator hot fix by @hang-yin in https://github.com/StanfordVL/OmniGibson/pull/900
* Rewrite particle modifier demo by @yyf20001230 in https://github.com/StanfordVL/OmniGibson/pull/779
* Minor update to isaac sim setup file by @yyf20001230 in https://github.com/StanfordVL/OmniGibson/pull/886
* Eric's release fixes by @ChengshuLi in https://github.com/StanfordVL/OmniGibson/pull/905
* Pre-release bug fixes by @hang-yin in https://github.com/StanfordVL/OmniGibson/pull/901
* Pre release doc update by @hang-yin in https://github.com/StanfordVL/OmniGibson/pull/917
* PyPI release infra, version bump by @cgokmen in https://github.com/StanfordVL/OmniGibson/pull/919
* Improved installation via an install script instead of setup.sh and metapackage by @cgokmen in https://github.com/StanfordVL/OmniGibson/pull/910
* Fix/misc bugs by @cremebrule in https://github.com/StanfordVL/OmniGibson/pull/920
* Revive action primitives examples and tests by @yyf20001230 in https://github.com/StanfordVL/OmniGibson/pull/842
* Fix docker build issue due to deprecated mamba arg by @cgokmen in https://github.com/StanfordVL/OmniGibson/pull/921
* Remove dependency on numba by @cgokmen in https://github.com/StanfordVL/OmniGibson/pull/924
* Replicator Patch for INVALID segmentation label by @hang-yin in https://github.com/StanfordVL/OmniGibson/pull/923
* Bump asset URL version by @cgokmen in https://github.com/StanfordVL/OmniGibson/pull/925

## New Contributors
* @yyf20001230 made their first contribution in https://github.com/StanfordVL/OmniGibson/pull/796

**Full Changelog**: https://github.com/StanfordVL/OmniGibson/compare/v1.0.0...v1.1.0

### OmniGibson v1.0.0 — 태그 `v1.0.0`

게시: 2024-03-18T06:57:36Z · https://github.com/StanfordVL/BEHAVIOR-1K/releases/tag/v1.0.0

We are happy to announce the first full release of BEHAVIOR-1K with 1,004 pre-sampled task instances now fully functional! See docs at https://behavior.stanford.edu/omnigibson/getting_started/installation.html

## **Highlights**
- Expanded interactive scenes
- Enhanced object interaction
- Teleoperation System
- Advanced feature support: action primitives, transition rules, etc.

## **Change log**
- Assets:
  - We now offer all 50 interactive traversable [scenes](https://behavior.stanford.edu/omnigibson/modules/scene.html)
  - Fillable object meshes have been updated with tighter, more accurate fillable volumes
  - Comprehensive meta link annotations
  - Full sliceable synset coverage
  - 1004 pre-sampled task initial conditions from the B1K activity suite

- Simulation:
  - Operational Space Controller support
  - New teleoperation system supporting OpenXR (Meta Quest), SteamVR (HTC VIVE), SpaceMouse, and Keyboard teleoperation
  - Assisted grasping support
  - More robust cloth generation
  - Support for external sensors
  - Controller logic has been significantly refactored to properly handle staggered action and physics timestep differences
  - Controller performance has been optimized, and gravity compensation has been fixed
  - Lazy import: importing OmniGibson no longer immediately launches the simulator
  - Poses in USD, Fabric, and PhysX are now synchronized without taking simulation steps
  - Full transition rule support
  - Deterministic global labels for semantic segmentation and consistent labels for instance segmentation
  - Segmentation support for particle systems
  - More accurate bounding box computation
  - Improved support for non-uniformly scaled meshes
  - Improved task sampling support
  - Action primitives support

- Usability:
  - Comprehensive documentation, including modules, tutorials, and updated examples
  - Logging has been pruned for a cleaner output.
  - Profiling support
  - A Colab notebook for running OmniGibson without local installation is now available

**Full Changelog**: https://github.com/StanfordVL/OmniGibson/compare/v0.2.1...v1.0.0
