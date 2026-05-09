"""Q-learning routing agent that learns routing policy from ticket outcomes.

This implements a Reinforcement Learning paradigm on top of the supervised
ML pipeline. The agent observes the ML model's predicted queue and priority
(the state), chooses a routing action, and updates its policy based on
reward feedback.
"""

from __future__ import annotations

import numpy as np

# Must match the queue labels your ML model predicts
QUEUE_LABELS = [
    "technical support",
    "billing and payments",
    "account access",
    "security concern",
    "general inquiry",
    "returns and exchanges",
    "product support",
    "service outages and maintenance",
    "feature request",
    "customer service",
]

PRIORITY_LABELS = ["low", "medium", "high", "critical"]

# Actions: route to each queue's team, or escalate
ACTIONS = QUEUE_LABELS + ["escalate"]


class RoutingAgent:
    """A Q-learning agent that learns ticket routing decisions.

    State  = (predicted_queue_index, predicted_priority_index)
    Action = route to a specific team OR escalate to senior staff
    Reward = +1 correct routing, -1 wrong routing, +2 correct escalation
             of a critical/high ticket

    The agent starts with no knowledge (all Q-values = 0) and improves
    its routing policy as it processes more tickets and receives feedback.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount: float = 0.9,
        epsilon: float = 0.2,
        random_state: int = 42,
    ) -> None:
        np.random.seed(random_state)
        # Q-table shape: (queue_states, priority_states, actions)
        self.q_table = np.zeros((len(QUEUE_LABELS), len(PRIORITY_LABELS), len(ACTIONS)))
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon  # exploration rate
        self.decisions_made = 0

    def _state_indices(self, queue: str, priority: str) -> tuple[int, int]:
        """Convert string labels to table indices."""
        q_lower = queue.strip().lower()
        p_lower = priority.strip().lower()
        q_idx = next((i for i, l in enumerate(QUEUE_LABELS) if l in q_lower), 0)
        p_idx = next((i for i, l in enumerate(PRIORITY_LABELS) if l in p_lower), 1)
        return q_idx, p_idx

    def choose_action(self, queue: str, priority: str) -> str:
        """Choose a routing action using epsilon-greedy policy.

        Explores randomly epsilon % of the time, exploits best known
        action otherwise. Exploration decreases as the agent gains
        experience.
        """
        q_idx, p_idx = self._state_indices(queue, priority)
        if np.random.random() < self.epsilon:
            return np.random.choice(ACTIONS)                        # explore
        return ACTIONS[int(np.argmax(self.q_table[q_idx, p_idx]))] # exploit

    def update(self, queue: str, priority: str, action: str, reward: float) -> None:
        """Update the Q-table after observing the reward for an action."""
        q_idx, p_idx = self._state_indices(queue, priority)
        action_idx = ACTIONS.index(action) if action in ACTIONS else 0

        old_q = self.q_table[q_idx, p_idx, action_idx]
        best_future_q = float(np.max(self.q_table[q_idx, p_idx]))

        self.q_table[q_idx, p_idx, action_idx] = (
        old_q + self.lr * (reward + self.gamma * best_future_q - old_q)
        )
        self.decisions_made += 1

    def route(self, queue: str, priority: str) -> str:
        """Main entry point — returns routing decision for a ticket."""
        return self.choose_action(queue, priority)

    def reward_for_outcome(self, action: str, true_queue: str, priority: str) -> float:
        """Calculate reward based on whether the routing decision was correct."""
        p_lower = priority.strip().lower()
        is_critical = p_lower in {"high", "critical"}

        if action == "escalate" and is_critical:
            return 2.0   # correctly escalated urgent ticket
        if action == "escalate" and not is_critical:
            return -0.5  # unnecessary escalation wastes senior staff time
        if action.strip().lower() == true_queue.strip().lower():
            return 1.0   # correct routing
        return -1.0      # wrong team assigned