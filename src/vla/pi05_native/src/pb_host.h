// PiBehavior (2025 BEHAVIOR 1st place) host side: input/output transforms and the B1K wrapper
// (stage voting, 26->20 cubic compression, correction rules, rolling soft inpainting), reproduced from
// IliaLarchenko/behavior-1k-solution with the 2026 eval adaptations of the alstar8 fork:
//   src/b1k/shared/b1k_proprio.py, policies/b1k_policy.py, transforms_normalize.py, policies/pi_behavior_policy.py,
//   shared/eval_b1k_wrapper.py, shared/correction_rules.py
#pragma once
#include <deque>
#include <string>
#include <vector>

#include "weights.h"

namespace pi05 {

struct PbSpec {
  int ah = 30, ad = 32, nact = 23;
  std::vector<int> stages;          // per task (2025 table)
  std::vector<int> delta_mask;      // 23
  std::vector<double> s_mean, s_std;  // state (32)
  std::vector<double> a_pt_mean, a_pt_std;  // per timestep actions [ah][32]
  std::vector<double> corr_L;       // [ah*ad]^2 correlated noise factor
};

bool load_pb_spec(const WeightFile& wf, PbSpec* s, std::string* err);

// 61-d compact (2026 eval) or 256-d legacy proprio -> 23-d state (grippers mapped to [-1, 1]), float32 like numpy
void pb_extract_state(const float* proprio, int n, float* state23);
// normalized, zero-padded state (32, f32) and its 32 bin tokens (jnp.digitize - 1, clipped to [0, 255])
void pb_state_tokens(const PbSpec& s, const float* state23, float* state_norm32, int* tokens32);
// kept absolute actions [keep][23] -> model-space initial actions [keep][32] (delta vs current state, per-timestep
// z-score of timesteps 0..keep-1, zero pad)
void pb_initial_actions(const PbSpec& s, const double* kept, int keep, const float* state23, float* out);
// model output [ah][32] (normalized, delta) -> absolute robot actions [ah][23] (float64)
void pb_postprocess(const PbSpec& s, const float* raw, const float* state_norm32, double* out);
// noise = z . L^T (f32)
void pb_correlate(const PbSpec& s, const float* z, float* noise);

struct PbWrapperCfg {
  int actions_to_execute = 26, actions_to_keep = 4, execute_in_n_steps = 20, history_len = 3, votes_to_promote = 2;
  bool apply_eval_tricks = true;
};

// One environment slot of B1KPolicyWrapper (eval_b1k_wrapper.py act()).
struct PbSlot {
  int task = -1;
  int stage = 0;
  int forced_stage = -1;  // >= 0: stage given from outside (planner), voting disabled
  std::deque<int> history;
  std::vector<double> last;        // [n][23]
  int n_last = 0, index = 0, step = 0, predictions = 0;
  std::vector<double> next_init;   // [keep][23] or empty
  int last_pred_stage = -1;
};

// apply_correction_rules + returns corrected stage (correction_rules.py:255-295)
int pb_correction_rules(int task, int stage, const float* state23, std::vector<double>& actions, int ah);
bool pb_gripper_variation(const std::vector<double>& actions, int n_check);
// scipy interp1d(kind="cubic") (not-a-knot spline) of [n][23] rows onto `target` evenly spaced samples
std::vector<double> pb_cubic_resample(const std::vector<double>& a, int n, int target);
// stage voting (eval_b1k_wrapper.py:153-188); max_stage from the model's own (2025) table
void pb_vote(PbSlot& s, int max_stage, int predicted, const PbWrapperCfg& c);

}  // namespace pi05
