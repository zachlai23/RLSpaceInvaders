"""
Prioritized Experience Replay Buffer with N-step Returns.

Combines:
  - Prioritized Experience Replay / PER (Schaul et al., 2015)
    Uses a sum-tree for O(log N) priority sampling.
  - Multi-step Returns (Sutton, 1988)
    Computes n-step bootstrapped targets before storing transitions.
"""

import numpy as np
from collections import deque


class SumTree:
    """
    Binary min-heap where each leaf stores a priority and each internal node
    stores the sum of its children.  Supports O(log N) add, update, and
    stratified sampling.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.n_entries = 0
        self.write = 0  # circular write pointer

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _propagate(self, idx: int, delta: float):
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left] or self.tree[right] == 0:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def total(self) -> float:
        return self.tree[0]

    def add(self, priority: float, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float):
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, delta)

    def get(self, s: float):
        """Return (tree_idx, priority, data) for sample value s."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    def __len__(self) -> int:
        return self.n_entries


class PrioritizedReplayBuffer:
    """
    PER buffer with n-step return accumulation.

    N-step logic:
      - Transitions are staged in an n_step_buffer (sliding window).
      - Once n transitions are seen, the oldest is committed to the
        sum-tree with its n-step return R_t^n.
      - At episode end, remaining staged transitions are flushed with
        shorter (< n) step returns.
    """

    def __init__(
        self,
        capacity: int,
        n_step: int,
        gamma: float,
        alpha: float = 0.6,
    ):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.alpha = alpha

        self.tree = SumTree(capacity)
        self.max_priority = 1.0

        # Sliding window for n-step accumulation
        self._n_buf: deque = deque()

    # ------------------------------------------------------------------
    # N-step helpers
    # ------------------------------------------------------------------

    def _n_step_info(self):
        """
        Compute (n_reward, n_next_state, n_done) for the current n_buf.
        Iterates forward; stops early at a terminal transition.
        """
        n_reward = 0.0
        n_done = False
        n_next_state = self._n_buf[-1][3]  # default: last element's next state

        for i, (_, _, r, ns, done) in enumerate(self._n_buf):
            n_reward += (self.gamma ** i) * r
            if done:
                n_next_state = ns
                n_done = True
                break

        return n_reward, n_next_state, n_done

    def _commit_oldest(self):
        """Compute n-step return for the oldest staged transition and store it."""
        n_reward, n_next_state, n_done = self._n_step_info()
        s0, a0 = self._n_buf[0][0], self._n_buf[0][1]
        self.tree.add(self.max_priority ** self.alpha, (s0, a0, n_reward, n_next_state, n_done))
        self._n_buf.popleft()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, state, action, reward, next_state, done: bool):
        self._n_buf.append((state, action, reward, next_state, done))

        # Commit the oldest transition once we have n transitions staged
        if len(self._n_buf) >= self.n_step:
            self._commit_oldest()

        # At episode end, flush whatever remains in the staging buffer
        if done:
            while self._n_buf:
                self._commit_oldest()

    def sample(self, batch_size: int, beta: float):
        """
        Stratified sampling weighted by priority.

        Returns:
            states, actions, rewards, next_states, dones  — numpy arrays
            idxs     — list of tree indices (for priority updates)
            weights  — importance-sampling weights, shape (batch_size,)
        """
        batch, idxs, priorities = [], [], []
        segment = self.tree.total / batch_size

        for i in range(batch_size):
            lo, hi = segment * i, segment * (i + 1)
            s = np.random.uniform(lo, hi)
            idx, priority, data = self.tree.get(s)

            # Guard against un-filled slots (shouldn't happen after is_ready check)
            if data is None:
                idx = self.capacity - 1
                priority = max(self.tree.tree[idx], 1e-8)
                data = self.tree.data[0]

            batch.append(data)
            idxs.append(idx)
            priorities.append(priority)

        priorities = np.array(priorities, dtype=np.float64)
        probs = np.clip(priorities / self.tree.total, 1e-8, None)
        weights = (len(self.tree) * probs) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            idxs,
            weights.astype(np.float32),
        )

    def update_priorities(self, idxs, td_errors):
        """Update priorities from absolute TD errors."""
        for idx, err in zip(idxs, td_errors):
            priority = max(float(err), 1e-6)
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority ** self.alpha)

    def is_ready(self, min_size: int) -> bool:
        return len(self.tree) >= min_size

    def __len__(self) -> int:
        return len(self.tree)
