// Host-side policy transforms of the openpi B1K pi05 policy, reproduced exactly (float64 where numpy uses it):
//   B1KInputs.extract_state (b1k_policy.py:22-41) -> Normalize z-score (transforms.py:137-139)
//   -> PaligemmaTokenizer pi05 prompt with discretized state (tokenizer.py:22-48)
//   ... model ...
//   Unnormalize (transforms.py:170-173) -> MappedAbsoluteActions (transforms.py:272-290) -> first 23 dims
#pragma once
#include <string>
#include <vector>

#include "tokenizer.h"
#include "weights.h"

namespace pi05 {

struct RobotSpec {
  std::vector<std::pair<bool, std::vector<int>>> proprio;  // (is_sum, indices) in order
  std::vector<std::pair<std::vector<int>, std::vector<int>>> delta;  // (action idx, state idx)
  std::vector<int> grippers;
  std::vector<std::string> cams;
  std::string name;
  int action_dim = 23;
  std::vector<double> s_mean, s_std, a_mean, a_std;
  int max_proprio_index() const;
};

struct PreparedInput {
  std::vector<int> tokens;       // valid tokens only (<= max_len), bos first
  bool truncated = false;
  std::vector<float> state_f32;  // normalized state as the model sees it (padded to action_dim_model)
  std::string prompt_text;       // full pi05 prompt string
};

bool load_robot_spec(const WeightFile& wf, RobotSpec* rs, std::string* err);
std::string clean_prompt(const std::string& s);  // python: s.strip().replace("_"," ").replace("\n"," ")
int discretize_state(double x);                  // np.digitize(x, linspace(-1,1,257)[:-1]) - 1
void prepare_input(const RobotSpec& rs, const Tokenizer& tok, const float* proprio, int n_proprio,
                   const std::string& prompt, int max_len, int model_action_dim, PreparedInput* out);
// raw: [horizon][model_action_dim] f32 in normalized units -> out: [horizon][rs.action_dim] f64
void postprocess_actions(const RobotSpec& rs, const float* raw, int horizon, int model_action_dim,
                         const std::vector<float>& state_f32, double* out);

}  // namespace pi05
