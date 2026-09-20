# PPO notes: learning to summarize from human feedback

These are my notes on [*Learning to summarize from human feedback*](https://arxiv.org/abs/2009.01325). I kept getting confused about how one score for a finished summary can train an LLM at every generated token.

## The models

The **policy** is the LLM writing the summary. In the paper, it starts from an LLM fine-tuned on Reddit TL;DR. Three other models have different jobs:

| Model | Job during PPO | Updated? |
| --- | --- | --- |
| Policy / actor | Writes the next token | Yes |
| SFT reference | Supplies probabilities for the KL penalty | No |
| Reward model | Scores a completed summary | No |
| Critic / value model | Predicts future reward from a partial summary | Yes |

The reward model is initialized from the SFT model and then trained on preferences. The critic is initialized as a **separate copy** of the trained reward model. Copying its weights does not mean they keep sharing parameters. The critic and policy are separate Transformers, so critic updates cannot alter the policy.

## Where the reward model comes from

The [released dataset](https://huggingface.co/datasets/openai/summarize_from_feedback) has a `comparisons` part: a post, two summaries, and `choice`, the index of the human-preferred summary. `choice: 1` means `summaries[1]` won; it is **not** a reward value of 1. Each summary's `policy` field says which model generated it. The dataset's `axis` ratings are for evaluation, not the paper's pairwise reward model training.

The reward model learns to give the chosen complete summary a higher scalar score:

$$
\mathcal{L}_{\mathrm{RM}}
= -\log\sigma\!\left(r_\theta(x,y^+) - r_\theta(x,y^-)\right),
$$

where $y^+$ is the chosen summary and $y^-$ is the rejected one.

This is supervised training **before PPO**. PPO uses the trained reward model but does not update it.

## The reward for one rollout

After the policy generates a summary $y$ for post $x$, the reward model scores the **completed** summary once. The paper defines:

$$
R(x,y)=r_\theta(x,y)-\beta\log\frac{\pi_\phi^{\mathrm{RL}}(y\mid x)}{\pi^{\mathrm{SFT}}(y\mid x)}.
$$

The second term penalizes movement away from the fixed SFT reference. Let the generated tokens be $y_0,\ldots,y_{T-1}$, with terminal state $T$. Write $h_t=(x,y_0,\ldots,y_{t-1})$ for the post and tokens already generated. An LLM's log probability of a summary is a sum of token log probabilities, so:

$$
\log\frac{\pi_\phi^{\mathrm{RL}}(y\mid x)}{\pi^{\mathrm{SFT}}(y\mid x)}
=\sum_{t=0}^{T-1}\left[
\log\pi_\phi^{\mathrm{RL}}(y_t\mid h_t)
-\log\pi^{\mathrm{SFT}}(y_t\mid h_t)
\right].
$$

For token-level bookkeeping, each generated token can receive $-\beta(\log p_t^{\mathrm{RL}}-\log p_t^{\mathrm{SFT}})$, with the reward model's one score added at the final token. Summing them gives the paper's whole-summary reward. The reward model itself does **not** grade partial summaries.

## What the critic predicts

Say the LLM has written `Dialectic is a method of reasoning...`. The critic predicts the reward **still to come if generation continues**. It is not rating that unfinished sentence as a finished summary.

Ignoring KL for a moment: if the completed summary scores `0.9` and the critic predicted `0.4` from that prefix, we can train the critic toward the observed `0.9`. Across many rollouts it learns an *expected* future outcome, since a prefix can have different continuations.

With KL included, the remaining return from a prefix includes the final reward model score and the KL contributions **from that prefix onward**. Earlier KL contributions are already in the past. Initializing the critic from the reward model only gives it starting weights; PPO rollouts teach it to make these prefix predictions.

## From reward and value to advantage

Advantage is calculated **after** a rollout. My rough mental model is:

$$
A_t\approx G_t-V_t,
$$

where $G_t$ is the observed remaining return and $V_t$ is the critic's prediction.

Positive means better than expected; negative means worse. Near zero means the critic predicted the outcome well. It does **not** mean the summary was especially good.

The paper uses generalized advantage estimation (GAE) with $\gamma=1$ and $\lambda=0.95$. For each generated token, work backward through its rewards and critic predictions:

$$
\begin{aligned}
\delta_t &= r_t+\gamma V_{t+1}-V_t,\\
A_t &= \delta_t+\gamma\lambda A_{t+1},\\
V_T &= A_T=0,\\
\widehat G_t^{\lambda} &= V_t+A_t.
\end{aligned}
$$

The final reward can therefore influence earlier tokens. Here $V_t$ and $A_t$ are calculated from the collected rollout and held fixed while PPO updates the models. The critic's new prediction $V_{\psi}(h_t)$ is trained toward the fixed target $\widehat G_t^{\lambda}$, for example with $\bigl(V_{\psi}(h_t)-\widehat G_t^{\lambda}\bigr)^2$. Because GAE uses intermediate value predictions, this target need not equal the raw Monte Carlo remaining return exactly.

## Where the advantage goes

PPO uses an advantage $A_t$ **at each generated token**, although the reward model scores the completed summary only once. It compares that token's probability under the updated policy with its probability under the **old policy that generated this rollout**:

$$
\rho_t(\phi)
=\frac{\pi_\phi(y_t\mid h_t)}
{\pi_{\mathrm{old}}(y_t\mid h_t)},
$$

$$
\mathcal{L}^{\mathrm{policy}}_t
=-\min\!\left(
\rho_t A_t,\;
\mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t
\right).
$$

The loss is reduced over valid generated tokens in the batch, excluding prompt and padding tokens. The critic has its own value loss. The `old` policy in this ratio is different from the fixed **SFT reference** used in the KL penalty. If KL was already included in rewards used to compute advantages, adding the same KL penalty again to the policy loss would count it twice.

## PPO and DPO

I may try [DPO](https://arxiv.org/abs/2305.18290) first. It trains directly on chosen/rejected pairs, without a separate reward model, critic, PPO rollouts, or advantage calculation. It still uses a frozen reference policy. PPO is the path to understanding the critic and token-level credit assignment.

## Papers

- [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
