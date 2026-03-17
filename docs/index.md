---
layout: default
title: Home
statusClass: disabled
finalClass: disabled
---

<img
  src="{{ '/assets/hero-ai-invaders.svg' | relative_url }}"
  class="img-fluid rounded mb-3"
  alt="Stylized Space Invaders alien facing a simple AI network diagram"
  loading="lazy"
/>

Welcome to the RLSpaceInvaders project! This is a reinforcement learning initiative to train autonomous agents to play Space Invaders using advanced machine learning algorithms.

## Quick Links

<div class="row mt-4">
  <div class="col-md-4 mb-3">
    <div class="card h-100">
      <div class="card-header">
        <h5 class="mb-0">📋 Proposal</h5>
      </div>
      <div class="card-body d-flex flex-column align-items-start">
        <p>Learn about our project goals, methodology, and evaluation plan.</p>
        <a href="{{ '/proposal.html' | relative_url }}" class="btn btn-primary btn-sm mt-auto">Read Proposal →</a>
      </div>
    </div>
  </div>
  <div class="col-md-4 mb-3">
    <div class="card h-100">
      <div class="card-header">
        <h5 class="mb-0">📊 Status</h5>
      </div>
      <div class="card-body d-flex flex-column align-items-start">
        <p>Check out our current progress and project milestones.</p>
        <a href="{{ '/status.html' | relative_url }}" class="btn btn-primary btn-sm mt-auto">View Status →</a>
      </div>
    </div>
  </div>
  <div class="col-md-4 mb-3">
    <div class="card h-100">
      <div class="card-header">
        <h5 class="mb-0">🎯 Final Report</h5>
      </div>
      <div class="card-body d-flex flex-column align-items-start">
        <p>See our final results and conclusions.</p>
        <a href="{{ '/final.html' | relative_url }}" class="btn btn-primary btn-sm mt-auto">Read Final Report →</a>
      </div>
    </div>
  </div>
</div>

## What's New

The [Final Report]({{ '/final.html' | relative_url }}) has just been published.

- **DQN:** Finalized hyperparameter tuning with temporal frame stacking (k=4), pushing the tuned agent to an average score of 42.95 and 324 frames of survival at 1M timesteps — a major improvement over the untuned stacked baseline. Final average reward: 32.
- **QR-DQN:** Distributional Q-learning paid off substantially, with QR-DQN reaching an average score of 76.5 and 588 survival frames at 1M steps, significantly outperforming DQN on both metrics. Final average reward: 50.
- **Rainbow DQN:** Achieved the highest overall performance, scoring 106.0 on average at 1M frames and continuing to improve all the way to ~292.0 at 5M frames — the only algorithm that showed no plateau, thanks to its distributional and noisy-network components. Final average reward: 125.
- **PPO:** Evaluated across four experimental configurations (Baseline Dual Decay, OpenAI Protocol, Genesis Reward Shaping, and Apex) plus a component ablation study isolating the roles of FrameStack, GAE, and learning-rate decay. Final average reward: 73.
- Added a final cross-algorithm comparison of all four models, summarizing trade-offs in learning stability, sample efficiency, and peak reward.

<video controls width="100%">
  <source src="assets/final/RLSpaceInvaders%20Final%20Report.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## 💻 Source Code

Checkout our implementation on GitHub:

**[View on GitHub →](https://github.com/zachlai23/RLGalaga)**
