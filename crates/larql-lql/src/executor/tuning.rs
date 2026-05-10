//! Centralised tuning constants for the LQL executor.
//!
//! Every value below has been measured (or carefully chosen) during a
//! specific experiment. Keeping them in one place — instead of scattered
//! across `compose`, `balance`, `rebalance`, `compile`, etc. — gives a
//! reader two things:
//!
//!   1. A single audit surface for "what's tunable" without grepping
//!      five files.
//!   2. A measurement-link comment per value, so the next maintainer
//!      can tell which were calibrated and which are heuristics.
//!
//! Scope rules:
//!
//!   - **Constants live here only when they're cross-module values or
//!     have a measurement origin.** A function-local magic that's
//!     genuinely a single-call detail (e.g. `BALANCE_ITERS` for the
//!     greedy loop, exposed for testing) is fine to live in `tuning`.
//!   - Module-private numbers that don't escape the file (e.g. a
//!     `for _ in 0..3` loop counter with a comment explaining why)
//!     should NOT be hoisted up here.
//!
//! ## Section: COMPOSE installs (`mutation::insert::compose`)
//!
//! `GATE_SCALE` is what makes a freshly-installed slot "fire louder"
//! than typical FFN slots so the down-vector reaches the residual
//! stream against thousands of natural competitors at the install
//! layer. Pinned by `install_math_produces_competing_activation` test
//! in `compose.rs`.

/// Multiplier on the unit gate direction at install time.
pub(crate) const GATE_SCALE: f32 = 30.0;

// ── Section: per-INSERT BALANCE pass (`mutation::insert::balance`) ──
//
// These bound the greedy amplify/shrink loop that tunes a single
// install's down-col scale until its target token's canonical
// probability lands inside `[PROB_FLOOR, PROB_CEILING]`. Empirical
// values from the exp 14 vindex compilation: 16 iterations is enough
// to reach steady-state for compose-mode installs at α ∈ [0.05, 0.30].

/// Maximum number of amplify/shrink iterations in `balance_installed`.
pub(crate) const BALANCE_ITERS: usize = 16;
/// Upper bound on the probe top-k inspected for the target token.
pub(crate) const BALANCE_PROBE_TOP_K: usize = 200;
/// Probability floor: below this we amplify the down-col.
pub(crate) const PROB_FLOOR: f64 = 0.30;
/// Probability ceiling: above this we shrink the down-col.
pub(crate) const PROB_CEILING: f64 = 0.95;
/// Multiplier when shrinking (`prob > ceiling`).
pub(crate) const DOWN_SCALE: f32 = 0.7;
/// Multiplier when amplifying (`prob < floor`).
///
/// Set ≈ `1 / DOWN_SCALE + margin` so a one-step shrink and one-step
/// grow don't oscillate around the band; the margin lets us cross the
/// band on the next iteration if we under-shoot.
pub(crate) const UP_SCALE: f32 = 1.6;
/// Stop amplifying after this many iterations of no improvement —
/// guards against `UP_SCALE` saturating into late-layer residual
/// blow-up.
pub(crate) const MAX_STALE: usize = 2;

// ── Section: cross-fact regression check (`mutation::insert::balance`) ──
//
// After the local balance lands, we re-probe a sample of priors to
// make sure the freshly-strengthened down-col hasn't hijacked any
// template-shared sibling. Capped at 16 priors per check, 8
// shrink-passes total.

/// Maximum number of shrink passes when restoring regressed priors.
pub(crate) const CROSS_ITERS: usize = 8;
/// Probability floor below which a prior is considered regressed.
pub(crate) const PRIOR_FLOOR: f64 = 0.20;
/// Maximum number of priors to probe per regression pass.
pub(crate) const MAX_PRIORS_CHECKED: usize = 16;

// ── Section: REBALANCE (`mutation::rebalance`) ──
//
// Smaller scale factors than per-INSERT (0.85/1.15 vs 0.7/1.6) because
// REBALANCE runs over many facts simultaneously and per-iter
// oscillation between competing template-shared facts dominates;
// dampening the per-iter step reduces back-and-forth churn.

/// Default `MAX <iters>` clause for REBALANCE.
pub(crate) const REBALANCE_MAX_ITERS_DEFAULT: u32 = 16;
/// Default FLOOR clause for REBALANCE.
pub(crate) const REBALANCE_FLOOR_DEFAULT: f32 = 0.30;
/// Default CEILING clause for REBALANCE.
pub(crate) const REBALANCE_CEILING_DEFAULT: f32 = 0.90;
/// Per-iter shrink factor in REBALANCE.
pub(crate) const REBALANCE_DOWN_SCALE: f32 = 0.85;
/// Per-iter grow factor in REBALANCE.
pub(crate) const REBALANCE_UP_SCALE: f32 = 1.15;
/// Probe top-k for REBALANCE per-fact prediction.
pub(crate) const REBALANCE_PROBE_TOP_K: usize = 200;

// ── Section: MEMIT (`lifecycle::compile`) ──
//
// `MEMIT_TARGET_ALPHA` is the unit-target nudge magnitude when MEMIT
// is solving in the `target_alpha × embed(target)` shortcut mode (the
// fast path). Validated at α=5 across the v11 200/200 reference.
// `MEMIT_DEFAULT_RIDGE` matches the Python reference's default and
// can be overridden via `LARQL_MEMIT_RIDGE`.

/// Default `target_alpha` passed to `run_memit` when the
/// `LARQL_MEMIT_TARGET_DELTA` opt-in path isn't selected.
pub(crate) const MEMIT_TARGET_ALPHA: f32 = 5.0;
/// Default ridge regularizer for MEMIT solves.
pub(crate) const MEMIT_DEFAULT_RIDGE: f64 = 0.1;
/// Default lambda for `COMPACT MAJOR` MEMIT decomposition.
pub(crate) const MEMIT_COMPACT_LAMBDA: f32 = 1e-3;
/// Floor on per-fact reconstruction cosine before COMPACT MAJOR
/// surfaces a quality warning.
pub(crate) const MEMIT_MIN_RECONSTRUCTION_COS: f32 = 0.95;

// ── Section: introspection (`query::describe`) ──
//
// `GATE_THRESHOLD` is the cutoff above which a feature's gate score
// is considered "active enough to surface" in DESCRIBE output. Tuned
// against Gemma 3 4B; chosen empirically to keep the top-N list
// dominated by genuinely activated features rather than long-tail
// noise.

/// Minimum gate score for a feature to be reported in DESCRIBE.
pub(crate) const DESCRIBE_GATE_THRESHOLD: f32 = 5.0;

// ── Section: canonical MEMIT prompt template ──
//
// The "{relation} of {entity} is" template is what compiled facts
// are keyed on across capture, balance, rebalance, COMPACT MAJOR,
// and the MEMIT solve. Any deviation captures the wrong residual
// and bakes the wrong direction into `down_weights.bin`.

/// Build the canonical MEMIT prompt for a (relation, entity) pair.
/// Embedded `-`/`_` in the relation are normalised to spaces.
pub(crate) fn canonical_prompt(relation: &str, entity: &str) -> String {
    let rel_words = relation.replace(['-', '_'], " ");
    format!("The {rel_words} of {entity} is")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_prompt_normalises_dashes_and_underscores() {
        assert_eq!(
            canonical_prompt("capital-of", "France"),
            "The capital of of France is"
        );
        assert_eq!(
            canonical_prompt("capital_of", "France"),
            "The capital of of France is"
        );
    }

    #[test]
    fn canonical_prompt_handles_simple_relation() {
        assert_eq!(
            canonical_prompt("capital", "France"),
            "The capital of France is"
        );
    }

    #[test]
    fn canonical_prompt_preserves_entity_casing_and_spaces() {
        assert_eq!(
            canonical_prompt("ruler", "United Kingdom"),
            "The ruler of United Kingdom is"
        );
    }

    #[test]
    fn balance_band_is_inclusive_and_ordered() {
        assert!(PROB_FLOOR < PROB_CEILING, "floor must be below ceiling");
        assert!(PROB_FLOOR > 0.0 && PROB_CEILING < 1.0);
    }

    #[test]
    fn rebalance_band_default_is_inside_per_insert_band() {
        // Rebalance dampens around the same target band as per-INSERT
        // balance; defaults should land within one step of each other.
        // Compare in f64 (PROB_FLOOR/CEILING are f64 by design — they
        // feed `predict_with_ffn` which returns probabilities as f64).
        let rb_floor = REBALANCE_FLOOR_DEFAULT as f64;
        let rb_ceiling = REBALANCE_CEILING_DEFAULT as f64;
        assert!(rb_floor >= PROB_FLOOR - 0.05 && rb_floor <= PROB_FLOOR + 0.05);
        assert!(rb_ceiling <= PROB_CEILING);
    }

    #[test]
    fn down_and_up_scale_are_complementary() {
        // Per-INSERT scales should round-trip approximately:
        // applying DOWN then UP should land near 1× rather than overshooting.
        let round_trip = DOWN_SCALE * UP_SCALE;
        assert!(round_trip >= 1.0 && round_trip <= 1.5);
    }
}
