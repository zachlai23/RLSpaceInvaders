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

- Added a full PPO approach section covering the clipped surrogate objective and Generalized Advantage Estimation (GAE).
- Included PPO training dynamics analysis across four experimental configurations with quantitative results and hyperparameter tables.
- Added a PPO component ablation study examining the effects of removing FrameStack, GAE, and learning-rate decay.
- Included a final cross-algorithm performance comparison of PPO, DQN, QR-DQN, and Rainbow DQN in the MinAtar Space Invaders environment.

<video controls width="100%">
  <source src="assets/status/RLSpaceInvaders%20Progress%20Report.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## 💻 Source Code

Checkout our implementation on GitHub:

**[View on GitHub →](https://github.com/zachlai23/RLGalaga)**
