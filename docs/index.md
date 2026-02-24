---
layout: default
title: Home
statusClass: disabled
finalClass: disabled
---

<div class="card mt-2 mb-4">
  <div class="card-header">
    <h5 class="mb-0">🚀 Welcome</h5>
  </div>
  <div class="card-body">
    <p class="mb-0">Welcome to the RLSpaceInvaders project! This is a reinforcement learning initiative to train autonomous agents to play Space Invaders using advanced machine learning algorithms.</p>
  </div>
</div>

## Quick Links

<div class="row mt-4">
  <div class="col-md-4 mb-3">
    <div class="card h-100">
      <div class="card-header">
        <h5 class="mb-0">📋 Proposal</h5>
      </div>
      <div class="card-body">
        <p>Learn about our project goals, methodology, and evaluation plan.</p>
        <a href="{{ '/proposal.html' | relative_url }}" class="btn btn-primary btn-sm">Read Proposal →</a>
      </div>
    </div>
  </div>
  <div class="col-md-4 mb-3">
    <div class="card h-100">
      <div class="card-header">
        <h5 class="mb-0">📊 Status</h5>
      </div>
      <div class="card-body">
        <p>Check out our current progress and project milestones.</p>
        <a href="{{ '/status.html' | relative_url }}" class="btn btn-primary btn-sm">View Status →</a>
      </div>
    </div>
  </div>
  <div class="col-md-4 mb-3">
    <div class="card h-100">
      <div class="card-header">
        <h5 class="mb-0">🎯 Final Report</h5>
      </div>
      <div class="card-body">
        <p>See our final results and conclusions.</p>
        <span class="btn btn-secondary btn-sm disabled" aria-disabled="true">Coming Soon</span>
      </div>
    </div>
  </div>
</div>

## What's New

The [Status page]({{ '/status.html' | relative_url }}) has just been updated with a new status report.

- Added a full multi-algorithm report covering DQN, QRDQN, PPO, and Rainbow DQN.
- Included updated evaluation results with performance tables, learning-curve visuals, and baseline comparisons.
- Added a clearer section on remaining goals, expected challenges, resources used, and a project video summary.

<video controls width="100%">
  <source src="assets/status/RLSpaceInvaders%20Progress%20Report.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## 💻 Source Code

Checkout our implementation on GitHub:

**[View on GitHub →](https://github.com/zachlai23/RLGalaga)**
