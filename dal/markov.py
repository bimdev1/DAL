"""Simple first‑order Markov chain implementation.

This module implements a lightweight Markov chain suitable for
modelling transitions between discrete states.  It records counts
of observed state transitions and converts them into probabilities
upon normalisation.  Sampling can then proceed according to the
learnt distribution.  This implementation does not depend on
external libraries and is designed for educational use in the DAL
project.

Example usage:

    from dal.markov import MarkovChain

    # Define a chain over some states
    mc = MarkovChain(["A", "B", "C"])
    mc.observe_sequence(["A", "B", "C", "A", "B"])
    mc.normalize()
    next_state = mc.next_state("A")
    sequence = mc.generate("A", 5)

"""

from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, MutableSequence, Sequence


class MarkovChain:
    """A simple first‑order Markov chain over a finite set of states.

    The chain keeps track of transition counts from state to state.
    Once normalised, these counts become probabilities used for
    sampling the next state given a current state.  Unknown states
    default to a uniform distribution over all states.
    """

    def __init__(self, states: Iterable[Any]) -> None:
        #: List of states known to the chain
        self.states: List[Any] = list(states)
        #: Mapping from state to index in the transition matrix
        self.state_to_index: Dict[Any, int] = {s: i for i, s in enumerate(self.states)}
        size = len(self.states)
        #: Raw counts of transitions (i -> j)
        self._counts: List[List[int]] = [[0] * size for _ in range(size)]
        #: Normalised probabilities (i -> j)
        self._probs: List[List[float]] | None = None

    def observe_sequence(self, sequence: Sequence[Any]) -> None:
        """Record transitions from an observed sequence of states.

        Args:
            sequence: A list of states observed sequentially.  Pairs of
                consecutive states will be counted as transitions.  Unknown
                states are ignored.
        """
        if not sequence:
            return
        for s1, s2 in zip(sequence, sequence[1:]):
            i = self.state_to_index.get(s1)
            j = self.state_to_index.get(s2)
            if i is None or j is None:
                continue
            self._counts[i][j] += 1
        # Invalidate cached probabilities
        self._probs = None

    def normalize(self) -> None:
        """Convert counts into row‑normalised probabilities.

        After calling this method, the chain will have a probability
        distribution for each state describing how likely it is to
        transition to each other state.  Rows with no observed
        transitions are initialised as uniform distributions.
        """
        size = len(self.states)
        self._probs = [[0.0] * size for _ in range(size)]
        for i, row in enumerate(self._counts):
            total = sum(row)
            if total == 0:
                # Uniform distribution if no outgoing transitions
                for j in range(size):
                    self._probs[i][j] = 1.0 / size
            else:
                for j, count in enumerate(row):
                    self._probs[i][j] = count / total

    def next_state(self, current_state: Any) -> Any:
        """Sample a successor state given the current state.

        Args:
            current_state: The state from which to sample the next state.

        Returns:
            A state chosen according to the learnt transition probabilities.
            Unknown states yield a uniform random state.
        """
        if self._probs is None:
            self.normalize()
        idx = self.state_to_index.get(current_state)
        if idx is None:
            return random.choice(self.states)
        probabilities = self._probs[idx]
        next_idx = random.choices(range(len(self.states)), weights=probabilities, k=1)[0]
        return self.states[next_idx]

    def generate(self, start_state: Any, n_steps: int) -> List[Any]:
        """Generate a sequence by repeatedly sampling successor states.

        Args:
            start_state: The initial state of the sequence.
            n_steps: The number of transitions to sample.

        Returns:
            A list containing ``n_steps + 1`` states starting from
            ``start_state`` and including each sampled successor.
        """
        if n_steps <= 0:
            return [start_state]
        sequence: List[Any] = [start_state]
        current = start_state
        for _ in range(n_steps):
            current = self.next_state(current)
            sequence.append(current)
        return sequence


__all__ = ["MarkovChain"]