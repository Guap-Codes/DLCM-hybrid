//! # Text Generation Sampler
//!
//! Implements advanced decoding strategies including Temperature scaling, 
//! Top-K filtering, and Top-P (Nucleus) sampling for autoregressive language models.

use candle_core::{Result, Tensor};
use rand::{distributions::Distribution, SeedableRng};
use rand::rngs::StdRng;

/// Manages the state and parameters for token sampling during text generation.
pub struct Sampler {
    temperature: f64,
    top_p: f64,
    top_k: usize,
    rng: StdRng,
}

impl Sampler {
    /// Creates a new sampler with the specified decoding parameters.
    ///
    /// * `temperature` - Controls the sharpness of the probability distribution.
    /// * `top_p` - Nucleus sampling threshold (e.g., 0.9 keeps the top 90% probability mass).
    /// * `top_k` - Limits sampling to the top K most likely tokens.
    pub fn new(temperature: f64, top_p: f64, top_k: usize) -> Self {
        Self {
            temperature,
            top_p,
            top_k,
            rng: StdRng::from_entropy(),
        }
    }

    /// Samples the next token from the model's output logits.
    ///
    /// This method extracts the logits to the CPU to perform efficient sorting
    /// and filtering (Top-K, Top-P), which avoids expensive GPU synchronization overhead.
    ///
    /// # Errors
    /// Returns a `candle_core::Error` if tensor extraction or categorical distribution creation fails.
    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        // 1. Extract logits to CPU. 
        // Copying 50k floats (~200KB) is extremely fast and avoids expensive 
        // GPU sorting/synchronization overhead.
        let logits_vec = logits.to_vec1::<f32>()?;

        // 2. Apply Temperature
        let temp = if self.temperature > 0.0 { self.temperature as f32 } else { 1.0 };
        let logits_vec: Vec<f32> = logits_vec.into_iter().map(|x| x / temp).collect();

        // 3. Softmax (Numerically stable)
        let max_logit = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        let mut probs: Vec<f32> = logits_vec.into_iter().map(|x| {
            let e = (x - max_logit).exp();
            sum_exp += e;
            e
        }).collect();

        if sum_exp > 0.0 {
            for p in probs.iter_mut() {
                *p /= sum_exp;
            }
        }

        // 4. Top-K Filtering (Optional)
        if self.top_k > 0 && self.top_k < probs.len() {
            let mut indices: Vec<usize> = (0..probs.len()).collect();
            indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));
            
            let mut allowed = vec![false; probs.len()];
            for i in 0..self.top_k {
                allowed[indices[i]] = true;
            }
            for (i, p) in probs.iter_mut().enumerate() {
                if !allowed[i] {
                    *p = 0.0;
                }
            }
        }

        // 5. Top-P (Nucleus) Filtering
        if self.top_p < 1.0 && self.top_p > 0.0 {
            let mut indices: Vec<usize> = (0..probs.len()).collect();
            indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));

            let mut cumsum = 0.0f32;
            let mut cutoff = probs.len();
            for (i, &idx) in indices.iter().enumerate() {
                cumsum += probs[idx];
                if cumsum > self.top_p as f32 {
                    cutoff = i + 1;
                    break; // Stop as soon as we cross the top_p threshold
                }
            }

            let mut allowed = vec![false; probs.len()];
            for i in 0..cutoff {
                allowed[indices[i]] = true;
            }
            
            let mut sum_allowed = 0.0f32;
            for (i, p) in probs.iter_mut().enumerate() {
                if !allowed[i] {
                    *p = 0.0;
                } else {
                    sum_allowed += *p;
                }
            }

            // Renormalize probabilities so they sum to 1.0
            if sum_allowed > 0.0 {
                for p in probs.iter_mut() {
                    *p /= sum_allowed;
                }
            }
        }

        // Clean up any NaNs just in case of numerical instability
        for p in probs.iter_mut() {
            if p.is_nan() { *p = 0.0; }
        }

        // 6. Sample from the categorical distribution
        let sum_probs: f32 = probs.iter().sum();
        if sum_probs <= 0.0 {
            // Fallback to argmax if sampling fails
            let max_idx = probs.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            return Ok(max_idx as u32);
        }

        let dist = rand::distributions::WeightedIndex::new(&probs)
            .map_err(|e| candle_core::Error::Msg(format!("Sampling error: {}", e)))?;
        
        let sampled_idx = dist.sample(&mut self.rng);
        Ok(sampled_idx as u32)
    }
}



