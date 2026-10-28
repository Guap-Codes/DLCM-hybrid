//! High-performance inference engine for Dynamic Large Concept Models (DLCM).
//! 
//! This module provides the Candle-based implementation of the DLCM architecture,
//! optimized for autoregressive generation with dynamic concept-level KV caching.

pub mod kv_cache;
pub mod model;
pub mod sampler;

pub use kv_cache::ConceptKvCache;
pub use model::{DlcmConfig, DlcmModel};
pub use sampler::Sampler;
