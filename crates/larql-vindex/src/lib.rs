//! Vindex — the queryable model format.
//!
//! Storage format, KNN index, load, save, and mutate operations for
//! the .vindex directory format. This crate owns the on-disk format
//! and the in-memory query index.
//!
//! Build pipeline (EXTRACT) and weight management live in `larql-inference`
//! because they need ModelWeights.

extern crate blas_src;

pub mod checksums;
pub mod config;
pub mod describe;
pub mod down_meta;
pub mod error;
pub mod index;
pub mod load;
pub mod mutate;

// Re-export dependencies for downstream crates.
pub use ndarray;
pub use tokenizers;

// Re-export essentials at crate root.
pub use config::{
    ExtractLevel, LayerBands, MoeConfig, VindexConfig, VindexLayerInfo, VindexModelConfig,
    VindexSource,
};
pub use describe::{DescribeEdge, LabelSource};
pub use error::VindexError;
pub use index::{
    FeatureMeta, IndexLoadCallbacks, SilentLoadCallbacks, VectorIndex, WalkHit, WalkTrace,
};
pub use load::{
    load_feature_labels, load_vindex_config, load_vindex_embeddings, load_vindex_tokenizer,
};
