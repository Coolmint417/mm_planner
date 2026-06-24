# BC Pretraining + DAgger + PPO Fine-tuning Training Paradigm for a Route-Conditioned Flying-Crawling Drone Planner

## 1. Overview

This document summarizes a recommended training paradigm for a route-conditioned multimodal planner for a flying-crawling drone.

The planner receives:

- Onboard RGB image observation.
- Historical self-motion trajectory over the past `n_traj_encoder` steps.
- Ordered task waypoints that the drone must visit sequentially.

The planner outputs:

- Discrete motion mode.
- Future flight waypoints.
- Crawling action.
- Transition action for landing, takeoff, attachment, detachment, etc.

The recommended training pipeline is:

```text
DINOv2 visual representation
        ↓
Behavior Cloning pretraining
        ↓
DAgger closed-loop correction
        ↓
Residual PPO reinforcement learning fine-tuning
        ↓
Sim2Real adaptation
```

The key idea is:

```text
BC teaches the model to imitate expert behavior.
DAgger teaches the model how to recover from its own mistakes.
PPO fine-tunes the policy to improve task success, robustness, and efficiency.
```

Directly training the whole policy from scratch with reinforcement learning is not recommended, because the exploration space is too large and the flying-crawling mode transition is high risk.

---

## 2. Policy Definition

Let the policy be:

\[
\pi_\theta(a_t \mid o_t)
\]

where the observation is:

\[
o_t = \{I_t, \tau_t, G_t\}
\]

with:

- \(I_t\): onboard RGB image.
- \(\tau_t\): historical trajectory features, including past motion modes, position, velocity, acceleration, attitude, and angular velocity.
- \(G_t\): current and future task waypoint window.

The action is hybrid:

\[
a_t = \{m_t, w_t, c_t, q_t\}
\]

where:

- \(m_t\): discrete motion mode.
- \(w_t\): future flight waypoint sequence.
- \(c_t\): crawling command.
- \(q_t\): transition command.

Motion modes can be:

```text
0 FLY
1 CRAWL
2 LAND
3 TAKEOFF
4 ATTACH
5 DETACH
6 HOVER
```

The planner can be decomposed as:

\[
\pi_\theta(a_t \mid o_t)
=
\pi_\theta(m_t \mid o_t)
\cdot
\pi_\theta(w_t \mid o_t, m_t)
\cdot
\pi_\theta(c_t \mid o_t, m_t)
\cdot
\pi_\theta(q_t \mid o_t, m_t)
\]

In implementation, all heads can be computed in parallel, but losses are applied only to the relevant head according to the expert mode.

---

## 3. Stage 0: Visual Representation Initialization

The vision encoder uses DINOv2 ViT-B/14.

Recommended initial setting:

```text
DINOv2 backbone: frozen
DINOv2 projection layer: trainable
Planner modules: trainable
```

The DINOv2 encoder maps image observations to visual features:

\[
z^{img}_t = f_{DINO}(I_t)
\]

If patch tokens are used:

\[
P_t = \{p_{t,1}, p_{t,2}, \dots, p_{t,N}\}
\]

Attention pooling can be used:

\[
\alpha_i =
\frac{\exp(s_i)}
{\sum_j \exp(s_j)}
\]

\[
z^{img}_t = \sum_i \alpha_i p_{t,i}
\]

where \(s_i\) is a learned score for the \(i\)-th patch token.

At the beginning, freezing DINOv2 improves stability, reduces memory usage, and avoids overfitting to synthetic textures.

---

## 4. Stage 1: Behavior Cloning Pretraining

### 4.1 Purpose

Behavior Cloning, or BC, is supervised imitation learning.

Given an expert dataset:

\[
\mathcal{D}_E = \{(o_t, a^E_t)\}
\]

the policy is trained to imitate expert actions:

\[
\min_\theta
\mathbb{E}_{(o,a^E)\sim \mathcal{D}_E}
\left[
\mathcal{L}(\pi_\theta(o), a^E)
\right]
\]

BC should be used as the first major training stage.

---

### 4.2 Recommended Expert Data Sources

Use a mixture of:

1. Rule-based expert planner.
2. Artificially generated route-following expert trajectories.
3. Low-level classical planners such as RRT, A*, MPC, minimum-snap trajectory generation.
4. Human teleoperation demonstrations.
5. Failure recovery demonstrations.

Recommended data composition for the first version:

```text
70% rule-based / algorithmic expert data
30% human demonstration and manually corrected edge cases
```

The dataset should include:

- Pure flight between task points.
- Flight-to-wall landing.
- Attachment and detachment.
- Wall crawling.
- Crawling-to-flight transition.
- Recovery from failed or near-failed transitions.
- Multi-waypoint route following.

---

## 5. BC Loss Function

The full BC loss is a weighted sum:

\[
\mathcal{L}_{BC}
=
\lambda_m \mathcal{L}_{mode}
+
\lambda_w \mathcal{L}_{wp}
+
\lambda_c \mathcal{L}_{crawl}
+
\lambda_q \mathcal{L}_{trans}
+
\lambda_s \mathcal{L}_{smooth}
+
\lambda_g \mathcal{L}_{goal}
\]

Recommended initial weights:

```python
lambda_mode = 1.0
lambda_waypoint = 1.0
lambda_crawl = 1.0
lambda_transition = 1.0
lambda_smooth = 0.05
lambda_goal = 0.1
```

---

### 5.1 Motion Mode Classification Loss

The mode head outputs:

\[
l_t \in \mathbb{R}^{K}
\]

where \(K\) is the number of modes.

The mode probability is:

\[
p_\theta(m_t=k\mid o_t)
=
\frac{\exp(l_{t,k})}{\sum_j \exp(l_{t,j})}
\]

The mode loss is cross entropy:

\[
\mathcal{L}_{mode}
=
-\log p_\theta(m^E_t \mid o_t)
\]

For imbalanced modes, use class weighting:

\[
\mathcal{L}_{mode}
=
-\omega_{m^E_t}
\log p_\theta(m^E_t \mid o_t)
\]

Rare modes such as `LAND`, `ATTACH`, `DETACH`, and `TAKEOFF` should receive larger weights.

Example:

```python
class_weights = torch.tensor([
    1.0,  # FLY
    1.0,  # CRAWL
    3.0,  # LAND
    3.0,  # TAKEOFF
    5.0,  # ATTACH
    5.0,  # DETACH
    1.0,  # HOVER
])
```

---

### 5.2 Probabilistic Autoregressive Waypoint Loss

The flight waypoint head is recommended to be probabilistic.

For a future waypoint sequence:

\[
W_t = \{w_{t,1}, w_{t,2}, \dots, w_{t,N}\}
\]

where:

\[
w_{t,i} =
[\Delta x, \Delta y, \Delta z, \Delta yaw]
\]

The autoregressive factorization is:

\[
p_\theta(W_t \mid o_t)
=
\prod_{i=1}^{N}
p_\theta(w_{t,i} \mid w_{t,<i}, o_t)
\]

A diagonal Gaussian version predicts:

\[
\mu_{t,i}, \log\sigma_{t,i}
=
f_\theta(o_t, w_{t,<i})
\]

\[
p_\theta(w_{t,i}\mid w_{t,<i},o_t)
=
\mathcal{N}
\left(
w_{t,i};
\mu_{t,i},
\operatorname{diag}(\sigma^2_{t,i})
\right)
\]

The waypoint negative log-likelihood loss is:

\[
\mathcal{L}_{wp}
=
-\sum_{i=1}^{N}
\log
p_\theta(w^E_{t,i}\mid w^E_{t,<i},o_t)
\]

In batch form:

```python
std = torch.exp(log_std)
dist = torch.distributions.Normal(mu, std)

log_prob = dist.log_prob(expert_waypoints)
loss_waypoint = -log_prob.sum(dim=[1, 2]).mean()
```

This loss is applied only when the expert mode is `FLY`.

```python
fly_mask = expert_mode == FLY
loss_waypoint = -dist.log_prob(
    expert_waypoints[fly_mask]
).sum(dim=[1, 2]).mean()
```

Clamp `log_std` for numerical stability:

```python
log_std = torch.clamp(log_std, min=-5.0, max=2.0)
```

---

### 5.3 Why NLL Instead of MSE?

If the head outputs a deterministic waypoint:

\[
\hat{w}_{t,i}
\]

then a common loss is:

\[
\mathcal{L}_{MSE}
=
\| \hat{w}_{t,i} - w^E_{t,i} \|^2
\]

If the head outputs a Gaussian distribution:

\[
w^E_{t,i} \sim \mathcal{N}(\mu_{t,i}, \sigma_{t,i}^2)
\]

the NLL for one dimension is:

\[
\mathcal{L}_{NLL}
=
\frac{1}{2}
\left(
\frac{w^E-\mu}{\sigma}
\right)^2
+
\log\sigma
+
C
\]

If \(\sigma\) is fixed, minimizing NLL is approximately equivalent to minimizing MSE.

Therefore:

```text
MSE is a special case of Gaussian NLL with fixed variance.
NLL is more flexible because the model also learns uncertainty.
```

---

### 5.4 Diagonal Gaussian vs Full Covariance

A diagonal Gaussian assumes:

\[
\Sigma =
\operatorname{diag}(\sigma_1^2,\sigma_2^2,\sigma_3^2,\sigma_4^2)
\]

This does not mean the action variables are physically independent. It means:

```text
Given the current observation and hidden state, the residual uncertainty of each output dimension is approximated as conditionally independent.
```

For a first implementation, diagonal Gaussian is recommended because it is stable and easy to train.

A more expressive version is per-waypoint full covariance:

\[
p(w_{t,i}\mid w_{t,<i},o_t)
=
\mathcal{N}(\mu_{t,i}, \Sigma_{t,i})
\]

where:

\[
\Sigma_{t,i} = L_{t,i} L_{t,i}^{T}
\]

and \(L\) is a Cholesky factor. This can capture correlations among:

```text
dx_body, dy_body, dz_body, dyaw
```

However, full covariance is more complex and less stable. It is better as a second-stage improvement.

---

### 5.5 Waypoint Smoothness Loss

To encourage smooth trajectories:

\[
v_i = w_{i+1} - w_i
\]

\[
a_i = v_{i+1} - v_i
\]

\[
\mathcal{L}_{smooth}
=
\frac{1}{N-2}
\sum_{i=1}^{N-2}
\|a_i\|^2
\]

In code:

```python
def waypoint_smoothness_loss(waypoints):
    velocity = waypoints[:, 1:] - waypoints[:, :-1]
    acceleration = velocity[:, 1:] - velocity[:, :-1]
    return (acceleration ** 2).mean()
```

Use the predicted mean trajectory:

```python
loss_smooth = waypoint_smoothness_loss(mu[fly_mask])
```

---

### 5.6 Goal Progress Loss

Let the current task goal in the body frame be:

\[
g_t \in \mathbb{R}^3
\]

Let the final predicted waypoint be:

\[
w_{t,N}^{pos} \in \mathbb{R}^3
\]

A directional goal progress loss can be:

\[
\mathcal{L}_{goal}
=
1 -
\frac{
g_t^\top w_{t,N}^{pos}
}{
\|g_t\| \|w_{t,N}^{pos}\|
}
\]

This encourages the local trajectory to move toward the current route waypoint.

Use a small weight because overly strong goal progress loss may encourage unsafe straight-line motion.

---

### 5.7 Crawling Action Loss

For crawling action:

\[
c_t = [v_x^{surf}, v_y^{surf}, \dot{\psi}^{surf}]
\]

If deterministic:

\[
\mathcal{L}_{crawl}
=
\operatorname{SmoothL1}
(c^\theta_t, c^E_t)
\]

This loss is applied only when expert mode is `CRAWL`.

If probabilistic, use Gaussian NLL similarly to the waypoint head.

---

### 5.8 Transition Action Loss

The transition head outputs:

```text
contact_pos:     [x, y, z]
surface_normal:  [nx, ny, nz]
yaw:             ψ
approach_speed:  v
```

Recommended losses:

\[
\mathcal{L}_{contact}
=
\operatorname{SmoothL1}
(p^\theta, p^E)
\]

\[
\mathcal{L}_{normal}
=
1 -
\cos(n^\theta, n^E)
\]

\[
\mathcal{L}_{yaw}
=
\operatorname{SmoothL1}
(\psi^\theta, \psi^E)
\]

\[
\mathcal{L}_{speed}
=
\operatorname{SmoothL1}
(v^\theta, v^E)
\]

Then:

\[
\mathcal{L}_{trans}
=
\mathcal{L}_{contact}
+
\mathcal{L}_{normal}
+
\mathcal{L}_{yaw}
+
\mathcal{L}_{speed}
\]

This loss is applied only for transition modes:

```text
LAND, TAKEOFF, ATTACH, DETACH
```

---

## 6. Stage 2: DAgger Closed-Loop Correction

### 6.1 Motivation

Behavior Cloning trains on states visited by the expert:

\[
s \sim d_{\pi_E}
\]

but during deployment, the learned policy visits states induced by itself:

\[
s \sim d_{\pi_\theta}
\]

This causes distribution shift.

A small prediction error can move the drone into a state never seen in the expert dataset, causing more errors and eventual failure.

---

### 6.2 DAgger Objective

DAgger aims to train the policy on the state distribution induced by the learned policy:

\[
\min_\theta
\mathbb{E}_{s\sim d_{\pi_\theta}}
\left[
\mathcal{L}
\left(
\pi_\theta(s),
\pi_E(s)
\right)
\right]
\]

Instead of only:

\[
\min_\theta
\mathbb{E}_{s\sim d_{\pi_E}}
\left[
\mathcal{L}
\left(
\pi_\theta(s),
\pi_E(s)
\right)
\right]
\]

In other words:

```text
BC learns what the expert does on expert states.
DAgger learns what the expert would do on the model's own visited states.
```

---

### 6.3 DAgger Algorithm

Initialize with expert dataset:

\[
\mathcal{D}_0 = \mathcal{D}_E
\]

Train initial policy:

\[
\pi_{\theta_0} = \operatorname{BC}(\mathcal{D}_0)
\]

For iteration \(k=1,\dots,K\):

1. Roll out current policy \(\pi_{\theta_{k-1}}\) in simulation.
2. Collect visited observations \(o_t\).
3. Query expert policy \(\pi_E(o_t)\) for labels.
4. Aggregate data:

\[
\mathcal{D}_k
=
\mathcal{D}_{k-1}
\cup
\{(o_t,\pi_E(o_t))\}
\]

5. Retrain or continue training:

\[
\pi_{\theta_k}
=
\operatorname{BC}(\mathcal{D}_k)
\]

---

### 6.4 Mixed Policy Execution

To avoid catastrophic failures early in DAgger, execute a mixture of expert and learned policy:

\[
\pi_k
=
\beta_k \pi_E
+
(1-\beta_k)\pi_{\theta_k}
\]

where:

\[
\beta_k \rightarrow 0
\]

Example schedule:

```python
def beta_schedule(k):
    return max(0.0, 0.8 * (0.5 ** k))
```

Example values:

```text
k = 0: beta = 0.8
k = 1: beta = 0.4
k = 2: beta = 0.2
k = 3: beta = 0.1
k = 4: beta = 0.05
```

Even when the model action is executed, the stored label should be the expert action.

---

### 6.5 DAgger in the Flying-Crawling Drone Task

In this problem, the DAgger state is the complete model input:

```text
o_t = {
    rgb_t,
    traj_mode_ids_t,
    traj_continuous_t,
    task_waypoints_t,
    task_waypoint_mask_t
}
```

The expert label is:

```text
a_t^E = {
    expert_mode_t,
    expert_flight_waypoints_t,
    expert_crawl_action_t,
    expert_transition_t
}
```

Important DAgger states to collect:

1. Flight path deviation.
2. Near-collision states.
3. Flight-to-landing transition boundary.
4. Failed attachment states.
5. Crawling route deviation.
6. Detachment and takeoff recovery.
7. High-uncertainty states from the probabilistic waypoint head.
8. States where the mode head has low confidence.

DAgger uses the same loss as BC. The difference is the data distribution, not the loss function.

---

## 7. Stage 3: Residual PPO Reinforcement Learning Fine-tuning

### 7.1 Why Use RL After BC and DAgger?

BC and DAgger imitate expert behavior, but the expert may not be optimal.

Reinforcement learning can further optimize:

- Task completion rate.
- Energy consumption.
- Path length.
- Safety.
- Robustness to disturbances.
- Attachment success.
- Smoothness.
- Mode-switching efficiency.

However, training the whole policy from scratch with RL is not recommended.

Recommended approach:

```text
Use Residual PPO after BC + DAgger.
```

---

### 7.2 Residual Policy Formulation

Let the BC/DAgger policy output a nominal action:

\[
a^{BC}_t = \pi_{BC}(o_t)
\]

Train a residual policy:

\[
\Delta a_t = \pi_{\phi}^{res}(o_t)
\]

The executed action is:

\[
a_t = a^{BC}_t + \Delta a_t
\]

For flight waypoints:

\[
W_t = W^{BC}_t + \Delta W_t
\]

For crawling action:

\[
c_t = c^{BC}_t + \Delta c_t
\]

For transition action:

\[
q_t = q^{BC}_t + \Delta q_t
\]

In the first RL version, keep the discrete mode fixed from the BC/DAgger policy and let PPO only tune continuous residuals.

This is more stable than allowing PPO to change all modes immediately.

---

### 7.3 MDP Formulation

Define the MDP:

\[
\mathcal{M} = (\mathcal{S},\mathcal{A},P,r,\gamma)
\]

where:

- \(\mathcal{S}\): observation/state space.
- \(\mathcal{A}\): residual continuous action space.
- \(P\): simulator dynamics.
- \(r\): reward.
- \(\gamma\): discount factor.

The residual policy is:

\[
\pi_\phi(\Delta a_t \mid o_t)
\]

The value function is:

\[
V_\psi(o_t)
\]

---

### 7.4 PPO Objective

The policy gradient objective is:

\[
J(\phi)
=
\mathbb{E}
\left[
\sum_t
\gamma^t r_t
\right]
\]

PPO uses a clipped surrogate objective.

Define probability ratio:

\[
r_t(\phi)
=
\frac{
\pi_\phi(\Delta a_t \mid o_t)
}{
\pi_{\phi_{old}}(\Delta a_t \mid o_t)
}
\]

The PPO clipped objective is:

\[
\mathcal{L}^{CLIP}_{PPO}(\phi)
=
\mathbb{E}_t
\left[
\min
\left(
r_t(\phi) A_t,
\operatorname{clip}
(r_t(\phi),1-\epsilon,1+\epsilon)A_t
\right)
\right]
\]

The value loss is:

\[
\mathcal{L}_{value}
=
\mathbb{E}_t
\left[
(V_\psi(o_t)-R_t)^2
\right]
\]

The entropy bonus is:

\[
\mathcal{L}_{entropy}
=
\mathbb{E}_t
[
\mathcal{H}(\pi_\phi(\cdot\mid o_t))
]
\]

The total PPO loss to minimize is commonly:

\[
\mathcal{L}_{PPO}
=
-\mathcal{L}^{CLIP}_{PPO}
+
c_v \mathcal{L}_{value}
-
c_e \mathcal{L}_{entropy}
\]

---

### 7.5 Advantage Estimation

Use Generalized Advantage Estimation, or GAE:

\[
\delta_t
=
r_t
+
\gamma V(o_{t+1})
-
V(o_t)
\]

\[
A_t
=
\sum_{l=0}^{\infty}
(\gamma\lambda)^l
\delta_{t+l}
\]

where:

- \(\gamma\): reward discount factor.
- \(\lambda\): GAE smoothing parameter.

Typical values:

```python
gamma = 0.99
gae_lambda = 0.95
clip_epsilon = 0.2
```

---

### 7.6 Reward Design

The reward should be route-conditioned and safety-aware.

A recommended reward:

\[
r_t
=
r_{progress}
+
r_{reach}
+
r_{finish}
+
r_{attach}
+
r_{takeoff}
-
r_{collision}
-
r_{energy}
-
r_{smooth}
-
r_{invalid}
-
r_{switch}
\]

Progress reward:

\[
r_{progress}
=
k_p
\left(
d_{t-1}^{goal}
-
d_t^{goal}
\right)
\]

Goal reaching reward:

\[
r_{reach}
=
\begin{cases}
R_{reach}, & \text{if current task waypoint is reached}\\
0, & \text{otherwise}
\end{cases}
\]

Full task completion reward:

\[
r_{finish}
=
\begin{cases}
R_{finish}, & \text{if all waypoints are completed}\\
0, & \text{otherwise}
\end{cases}
\]

Collision penalty:

\[
r_{collision}
=
\begin{cases}
-R_{collision}, & \text{if collision occurs}\\
0, & \text{otherwise}
\end{cases}
\]

Energy penalty:

\[
r_{energy}
=
-k_E E_t
\]

Smoothness penalty:

\[
r_{smooth}
=
-k_S \|a_t-a_{t-1}\|^2
\]

Invalid mode penalty:

\[
r_{invalid}
=
\begin{cases}
-R_{invalid}, & \text{if CRAWL is selected without contact}\\
-R_{invalid}, & \text{if FLY is selected while adhesion is active}\\
0, & \text{otherwise}
\end{cases}
\]

Mode switch penalty:

\[
r_{switch}
=
-k_M \mathbb{1}[m_t \neq m_{t-1}]
\]

The reward must depend on the current route index. Otherwise, the policy may skip intermediate task waypoints.

---

## 8. Recommended Training Schedule

### Stage 1: BC Pretraining

```text
Input data:
    Rule expert data
    Human teleoperation data
    Successful planner-generated trajectories

Vision:
    Freeze DINOv2 backbone

Train:
    Projection layer
    Historical trajectory encoder
    Task waypoint encoder
    Fusion transformer
    Prediction heads

Loss:
    Mode CE
    Waypoint NLL
    Crawl loss
    Transition loss
    Smoothness loss
    Goal progress loss
```

Recommended duration:

```text
Train until validation BC loss and closed-loop short-horizon success stop improving.
```

---

### Stage 2: DAgger

```text
For each DAgger iteration:
    1. Roll out the current policy in simulation.
    2. Collect model-visited states.
    3. Query expert labels.
    4. Add new data to dataset.
    5. Continue BC training.
```

Use mixed policy execution early:

```text
early DAgger:
    high beta, more expert execution

late DAgger:
    low beta, mostly model execution
```

Important states:

```text
near failure
mode switching boundary
attachment failure
route deviation
high uncertainty
```

Loss remains the same as BC.

---

### Stage 3: Residual PPO

```text
Freeze:
    DINOv2 backbone
    optionally most of the pretrained planner

Train:
    residual policy head
    value head
    optionally last few fusion layers
```

Initial RL scope:

```text
Do not let PPO change discrete mode immediately.
Use BC/DAgger mode output.
PPO only modifies continuous residual actions.
```

After stable training:

```text
Optionally allow PPO to adjust mode selection with strong safety constraints.
```

---

## 9. Practical Recommendations

### 9.1 Start Simple

First train:

```text
BC only
DINOv2 frozen
diagonal Gaussian waypoint head
deterministic crawl and transition heads
```

Then add:

```text
DAgger
Residual PPO
probabilistic crawl/transition heads
full covariance waypoint head
```

---

### 9.2 Use Diagonal Gaussian First

For the waypoint head:

```text
First version:
    autoregressive diagonal Gaussian

Second version:
    per-waypoint full covariance Gaussian

Third version:
    mixture density or diffusion trajectory head
```

---

### 9.3 Do Not Skip DAgger

DAgger is often more important than PPO for this task, because it directly addresses closed-loop distribution shift.

Recommended order:

```text
BC → DAgger → Residual PPO
```

not:

```text
BC → PPO
```

---

### 9.4 Monitor These Metrics

During validation and simulation rollout, monitor:

```text
Mode classification accuracy
Waypoint NLL
Waypoint tracking error
Trajectory smoothness
Task waypoint reach rate
Full route completion rate
Collision rate
Attachment success rate
Crawling success rate
Mode switch frequency
Energy consumption
Recovery success after deviation
```

---

## 10. Summary

The recommended training paradigm is:

```text
1. Use DINOv2 ViT-B/14 as a pretrained frozen visual backbone.
2. Use Behavior Cloning to initialize the route-conditioned multimodal planner.
3. Use a probabilistic autoregressive waypoint head and train it with NLL.
4. Use cross-entropy for mode prediction.
5. Use SmoothL1 / cosine losses for crawling and transition outputs.
6. Use DAgger to collect model-visited states and expert labels.
7. Use Residual PPO to fine-tune continuous actions after the BC+DAgger policy is stable.
8. Keep mode selection fixed during early PPO for stability.
9. Only later consider full-policy PPO, full covariance waypoint distributions, or diffusion trajectory heads.
```

The most stable practical pipeline is:

```text
Frozen DINOv2
    +
BC pretraining
    +
Selective DAgger
    +
Residual PPO
```

This pipeline balances learning efficiency, closed-loop robustness, and safety for a flying-crawling multimodal drone planner.
