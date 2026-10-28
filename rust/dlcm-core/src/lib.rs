//! Core tensor operations for Dynamic Large Concept Models (DLCM).
//! 
//! This module provides hardware-accelerated implementations of the 
//! Dynamic Segmentation and Concept Replication mechanisms described 
//! in the DLCM paper, utilizing the `candle-core` framework.

pub mod boundary;
pub mod replication;
pub mod cuda;

pub use boundary::dynamic_segmentation_and_pooling;
pub use replication::concept_replication;
