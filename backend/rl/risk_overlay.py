"""Risk Overlay — hard constraints applied after RL decision, before execution.

This module implements the final safety layer that ensures RL-generated actions
comply with risk management rules regardless of what the policy outputs.

Constraints:
1. Regime-based position scaling (reuse existing REGIME_POSITION_SCALE)
2. Per-asset position cap (max 20% single position)
3. Total exposure limit (regime-dependent)
4. Cash buffer enforcement (minimum 5% cash)
5. Daily turnover limit (max 30% single-side turnover)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Regime-based maximum total exposure
REGIME_MAX_EXPOSURE: dict[str, float] = {
    "BULL": 1.0,  # Full investment allowed
    "RANGE": 0.7,  # 70% max exposure
    "BEAR": 0.4,  # 40% max exposure (defensive)
    "CRASH": 0.15,  # 15% minimum exposure (avoid full cash drag)
}

# Regime-based position scale (compatible with existing system)
REGIME_POSITION_SCALE: dict[str, float] = {
    "BULL": 1.0,
    "RANGE": 0.7,
    "BEAR": 0.4,
    "CRASH": 0.15,
}


@dataclass(frozen=True)
class RiskOverlayConfig:
    """Configuration for risk overlay constraints."""

    max_single_position: float = 0.20  # Max 20% in single asset
    min_cash_buffer: float = 0.05  # At least 5% cash
    max_daily_turnover: float = 0.30  # Max 30% single-side turnover per day
    max_total_leverage: float = 1.0  # No leverage (long-only)
    allow_short: bool = False  # Long-only by default


@dataclass
class RiskOverlayResult:
    """Result of risk overlay application."""

    adjusted_action: np.ndarray  # Final action after all constraints
    original_action: np.ndarray  # Original RL output
    constraints_applied: list[str]  # Which constraints were triggered
    regime: str
    total_exposure: float
    cash_ratio: float
    turnover: float


class RiskOverlay:
    """Applies hard risk constraints to RL-generated trading actions.

    Usage:
        overlay = RiskOverlay(config=RiskOverlayConfig())
        result = overlay.apply(action, prev_positions, regime="BULL")
        safe_action = result.adjusted_action
    """

    def __init__(self, config: RiskOverlayConfig = RiskOverlayConfig()):
        self.config = config

    def apply(
        self,
        action: np.ndarray,
        prev_positions: np.ndarray | None = None,
        regime: str = "RANGE",
    ) -> RiskOverlayResult:
        """Apply all risk constraints to the action.

        Args:
            action: Raw RL output, shape (N_assets + 1,).
                    First N elements are target position changes per asset.
                    Last element is cash ratio adjustment.
            prev_positions: Previous position weights, shape (N_assets,).
                           None if first step.
            regime: Current market regime (BULL/RANGE/BEAR/CRASH).

        Returns:
            RiskOverlayResult with adjusted action and diagnostics.
        """
        cfg = self.config
        constraints_applied: list[str] = []
        original = action.copy()

        # Separate asset actions from cash adjustment
        n_assets = len(action) - 1
        asset_action = action[:n_assets].copy()
        cash_adj = action[n_assets] if n_assets > 0 else 0.0

        # --- Constraint 1: Regime-based scaling ---
        scale = REGIME_POSITION_SCALE.get(regime, 0.6)
        if scale < 1.0:
            asset_action *= scale
            constraints_applied.append(f"regime_scale({regime}={scale})")

        # --- Constraint 2: No short selling (if configured) ---
        if not cfg.allow_short:
            neg_mask = asset_action < 0
            if neg_mask.any():
                # Allow reducing positions but not going short
                if prev_positions is not None:
                    # Can sell up to current position
                    min_allowed = -prev_positions
                    asset_action = np.maximum(asset_action, min_allowed)
                else:
                    asset_action = np.maximum(asset_action, 0.0)
                constraints_applied.append("no_short")

        # --- Constraint 3: Per-asset position cap ---
        if prev_positions is not None:
            new_positions = prev_positions + asset_action
        else:
            new_positions = asset_action.copy()

        capped = np.clip(new_positions, 0.0, cfg.max_single_position)
        if not np.allclose(capped, new_positions):
            asset_action = capped - (prev_positions if prev_positions is not None else 0.0)
            constraints_applied.append(f"single_cap({cfg.max_single_position})")

        # --- Constraint 4: Total exposure limit ---
        max_exposure = REGIME_MAX_EXPOSURE.get(regime, 0.6)
        if prev_positions is not None:
            new_positions = prev_positions + asset_action
        else:
            new_positions = asset_action.copy()

        total_exposure = np.sum(np.abs(new_positions))
        if total_exposure > max_exposure and total_exposure > 1e-8:
            shrink = max_exposure / total_exposure
            new_positions *= shrink
            asset_action = new_positions - (prev_positions if prev_positions is not None else 0.0)
            constraints_applied.append(f"exposure_cap({max_exposure})")

        # --- Constraint 5: Cash buffer ---
        if prev_positions is not None:
            new_positions = prev_positions + asset_action
        else:
            new_positions = asset_action.copy()

        invested = np.sum(np.abs(new_positions))
        cash_ratio = 1.0 - invested
        if cash_ratio < cfg.min_cash_buffer:
            # Reduce all positions proportionally to restore cash buffer
            target_invested = 1.0 - cfg.min_cash_buffer
            if invested > 1e-8:
                shrink = target_invested / invested
                new_positions *= shrink
                asset_action = new_positions - (prev_positions if prev_positions is not None else 0.0)
                constraints_applied.append(f"cash_buffer({cfg.min_cash_buffer})")

        # --- Constraint 6: Daily turnover limit ---
        if prev_positions is not None:
            turnover = np.sum(np.abs(asset_action))
            if turnover > cfg.max_daily_turnover:
                shrink = cfg.max_daily_turnover / turnover
                asset_action *= shrink
                constraints_applied.append(f"turnover_cap({cfg.max_daily_turnover})")
                turnover = cfg.max_daily_turnover
        else:
            turnover = np.sum(np.abs(asset_action))

        # Reconstruct full action
        adjusted = np.zeros_like(action)
        adjusted[:n_assets] = asset_action
        adjusted[n_assets] = cash_adj

        # Final positions for reporting
        if prev_positions is not None:
            final_positions = prev_positions + asset_action
        else:
            final_positions = asset_action.copy()

        final_exposure = float(np.sum(np.abs(final_positions)))
        final_cash = max(0.0, 1.0 - final_exposure)

        return RiskOverlayResult(
            adjusted_action=adjusted,
            original_action=original,
            constraints_applied=constraints_applied,
            regime=regime,
            total_exposure=final_exposure,
            cash_ratio=final_cash,
            turnover=float(turnover),
        )


def apply_risk_overlay(
    action: np.ndarray,
    prev_positions: np.ndarray | None = None,
    regime: str = "RANGE",
    config: RiskOverlayConfig = RiskOverlayConfig(),
) -> np.ndarray:
    """Convenience function: apply risk overlay and return adjusted action only.

    Args:
        action: Raw RL action vector.
        prev_positions: Previous positions (None for first step).
        regime: Current market regime.
        config: Risk overlay configuration.

    Returns:
        Adjusted action vector after all constraints.
    """
    overlay = RiskOverlay(config=config)
    result = overlay.apply(action, prev_positions, regime)
    return result.adjusted_action
