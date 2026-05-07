//! larql-boundary — confidence-gated BOUNDARY ref codec.
//!
//! Implements Phases 1–3 of the BOUNDARY_REF_PROTOCOL spec (see
//! `experiments/43_residual_stream_codec/BOUNDARY_REF_PROTOCOL.md`).
//!
//! # Architecture
//!
//! ```text
//! Phase 1 — codec      residual bytes ↔ f32 slices
//! Phase 2 — metadata   per-boundary confidence fields from logit slices
//! Phase 3 — gate       per-boundary compression decision
//! ```
//!
//! **Model-agnostic.** This crate takes raw `f32` slices (residuals, logits).
//! Forward passes stay in `larql-inference`.
//!
//! # Quick start
//!
//! ```rust
//! use larql_boundary::{codec, gate, metadata};
//! use larql_boundary::gate::BoundaryGateConfig;
//!
//! // --- Phase 1: encode a residual
//! let residual = vec![0.1f32; 2560];
//! let payload = codec::int8::encode(&residual);
//! let decoded = codec::int8::decode(&payload);
//! assert_eq!(decoded.len(), residual.len());
//!
//! // --- Phase 2: compute metadata from logits (caller runs forward pass)
//! let raw_logits = vec![0.0f32; 262_145]; // Gemma 3 vocab size
//! let mut meta = metadata::compute(&raw_logits, None);
//!
//! // --- Phase 3: gate decision
//! let config = BoundaryGateConfig::default(); // calibration_mode = true
//! let decision = gate::apply(&mut meta, &config);
//! ```

pub mod codec;
pub mod frame;
pub mod gate;
pub mod metadata;

// Re-export the most common types at the crate root.
pub use frame::{
    BoundaryAgreement, BoundaryCompression, BoundaryContract, BoundaryFrame, FallbackPolicy,
};
pub use gate::{BoundaryDecision, BoundaryGateConfig};
pub use metadata::BoundaryMetadata;
