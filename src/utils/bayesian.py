"""Bayesian calibration and fusion utilities."""

from typing import Dict, List, Tuple
import numpy as np


class BayesianUpdater:
    """Bayesian belief updater for cognitive state estimation.
    
    Provides a mechanism to update state probabilities based on new evidence
    using Bayes' theorem: P(State|Evidence) = P(Evidence|State) * P(State) / P(Evidence)
    """
    
    def __init__(self, initial_priors: Dict[str, float], decay_factor: float = 0.95):
        """Initialize Bayesian updater.
        
        Args:
            initial_priors: Initial probability distribution for states
            decay_factor: Factor to decay priors towards uniform over time
        """
        self.priors = initial_priors
        self.decay_factor = decay_factor
        self.states = list(initial_priors.keys())
        
        # Validate priors
        total = sum(self.priors.values())
        if abs(total - 1.0) > 0.01:
            # Normalize if needed
            for k in self.priors:
                self.priors[k] /= total
    
    def update(
        self,
        evidence_probs: Dict[str, float],
        reliability: float = 1.0
    ) -> Dict[str, float]:
        """Update beliefs based on new evidence.
        
        Args:
            evidence_probs: Probability of each state given the CURRENT evidence alone
                            Represents P(Evidence|State) * P(Evidence_strength) roughly
            reliability: How reliable is this evidence source (0.0 to 1.0)
            
        Returns:
            Updated posterior probabilities
        """
        posteriors = {}
        
        # Apply reliability: if unreliable, evidence pulls towards uniform/priors
        effective_evidence = {}
        for state in self.states:
            # If reliability is 0, evidence is effectively uniform (0.5 or 1/N)
            # Here we model reliability as a weight between the evidence distribution 
            # and the prior distribution
            measured_prob = evidence_probs.get(state, 0.0)
            effective_evidence[state] = measured_prob
            
        # Bayes Update
        # P(S|E) is proportional to P(E|S) * P(S)
        # We treat 'effective_evidence' as the likelihood term P(E|S)
        # (This is a simplified Naive Bayes assumption where evidence maps directly to state likelihoods)
        
        total_unnormalized = 0.0
        temp_posteriors = {}
        
        for state in self.states:
            prior = self.priors.get(state, 1.0 / len(self.states))
            likelihood = effective_evidence.get(state, 0.0)
            
            # Apply reliability weight - mix likelihood with neutral
            weighted_likelihood = (likelihood * reliability) + (0.5 * (1 - reliability))
            
            # Posterior proportional to Likelihood * Prior
            posterior = weighted_likelihood * prior
            temp_posteriors[state] = posterior
            total_unnormalized += posterior
            
        # Normalize with epsilon for numerical stability
        epsilon = 1e-10
        if total_unnormalized > epsilon:
            for state in self.states:
                posteriors[state] = temp_posteriors[state] / total_unnormalized
        else:
            # Fallback to priors if update fails
            posteriors = self.priors.copy()
            
        # Update priors for next step (with decay)
        self._update_priors(posteriors)
        
        return posteriors
    
    def _update_priors(self, posteriors: Dict[str, float]):
        """Update internal priors for the next time-step (Temporal smoothing)."""
        uniform = 1.0 / len(self.states)
        for state in self.states:
            # Decay towards uniform distribution to prevent getting stuck
            current_posterior = posteriors.get(state, uniform)
            self.priors[state] = (current_posterior * self.decay_factor) + (uniform * (1 - self.decay_factor))
            
    def get_priors(self) -> Dict[str, float]:
        """Get current prior beliefs."""
        return self.priors.copy()
