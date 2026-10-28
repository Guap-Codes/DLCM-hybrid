use candle_core::{DType, Device, Result, Tensor};

/// Manages the Key-Value cache for the Concept-Level Reasoning Backbone.
/// 
/// Unlike standard LLMs that cache $L$ tokens, DLCM only needs to cache $M$ concepts
/// (where $M \approx L / R$). This struct maintains the compressed KV state across
/// autoregressive generation steps.
pub struct ConceptKvCache {
    k_cache: Tensor,
    v_cache: Tensor,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
}

impl ConceptKvCache {
    /// Initializes an empty KV cache for a specific layer.
    ///
    /// # Errors
    /// Returns a `candle_core::Error` if the device fails to allocate the initial zero tensors.
    pub fn new(num_heads: usize, head_dim: usize, device: &Device) -> Result<Self> {
        // Shape: (0, num_heads, head_dim)
        let k_cache = Tensor::zeros((0, num_heads, head_dim), DType::F32, device)?;
        let v_cache = Tensor::zeros((0, num_heads, head_dim), DType::F32, device)?;
        
        Ok(Self {
            k_cache,
            v_cache,
            seq_len: 0,
            num_heads,
            head_dim,
        })
    }

    /// Appends new concept K/V vectors to the cache and returns the updated full cache.
    ///
    /// # Errors
    /// Returns a `candle_core::Error` if tensor concatenation fails or if dimension extraction fails.
    pub fn append(&mut self, k_new: &Tensor, v_new: &Tensor) -> Result<(Tensor, Tensor)> {
        let k = if self.seq_len == 0 {
            k_new.clone()
        } else {
            Tensor::cat(&[&self.k_cache, k_new], 0)?
        };
        
        let v = if self.seq_len == 0 {
            v_new.clone()
        } else {
            Tensor::cat(&[&self.v_cache, v_new], 0)?
        };

        // Use saturating_add to prevent arithmetic side-effects warnings
        self.seq_len = self.seq_len.saturating_add(k_new.dim(0)?);
        self.k_cache = k.clone();
        self.v_cache = v.clone();

        Ok((k, v))
    }

    /// Clears the cache for a new generation sequence.
    ///
    /// # Errors
    /// Returns a `candle_core::Error` if the device fails to allocate the new zero tensors.
    pub fn clear(&mut self, device: &Device) -> Result<()> {
        self.seq_len = 0;
        self.k_cache = Tensor::zeros((0, self.num_heads, self.head_dim), DType::F32, device)?;
        self.v_cache = Tensor::zeros((0, self.num_heads, self.head_dim), DType::F32, device)?;
        Ok(())
    }

    /// Returns the current sequence length (number of cached concepts).
    #[must_use]
    pub const fn seq_len(&self) -> usize {
        self.seq_len
    }
}


