"""
VQ-VAE-HMM Portfolio Optimization System

A comprehensive framework for regime-switching models with advanced 
portfolio optimization and delta hedging strategies.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from src.models.vae_hmm import VAE_HMM, Encoder, Decoder, Prior
from src.portfolio.optimizers import (
    RegimePortfolioOptimizer,
    AttentionPortfolioOptimizer,
    TransformerPortfolioOptimizer,
    BayesianPortfolioOptimizer
)
from src.portfolio.hedging import RegimeDeltaHedger, DynamicDeltaHedger
from src.training.trainer import Trainer, MetaTrainer, WalkForwardTrainer

__all__ = [
    "VAE_HMM",
    "Encoder", 
    "Decoder",
    "Prior",
    "RegimePortfolioOptimizer",
    "AttentionPortfolioOptimizer",
    "TransformerPortfolioOptimizer",
    "BayesianPortfolioOptimizer",
    "RegimeDeltaHedger",
    "DynamicDeltaHedger",
    "Trainer",
    "MetaTrainer",
    "WalkForwardTrainer"
]
