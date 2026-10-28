//! # DLCM Model Architecture
//!
//! Defines the core Rust structures for the Dynamic Large Concept Model (DLCM).
//! This includes the multi-stage pipeline: Encoder, Dynamic Segmenter, Concept Model, and Decoder.
//! All modules are optimized for `candle-core` tensor operations on CUDA/CPU.

#![allow(
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value,
    clippy::arithmetic_side_effects,
    clippy::cast_precision_loss,
    clippy::as_conversions,
    clippy::similar_names
)]

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear_no_bias, Embedding, Linear, VarBuilder, RmsNorm};
use dlcm_core::{dynamic_segmentation_and_pooling, concept_replication};

// =====================================================================
// Configuration
// =====================================================================

/// Configuration parameters for the DLCM architecture.
///
/// Defines the dimensions, layer counts, and hyperparameters for the heterogeneous
/// token-level and concept-level pathways.
#[derive(Debug, Clone)]
pub struct DlcmConfig {
    pub vocab_size: usize,
    pub d_token: usize,
    pub d_concept: usize,
    pub d_scan: usize,
    pub enc_layers: usize,
    pub enc_heads: usize,
    pub enc_ff: usize,
    pub concept_layers: usize,
    pub concept_heads: usize,
    pub concept_ff: usize,
    pub dec_layers: usize,
    pub dec_heads: usize,
    pub dec_ff: usize,
    pub alpha: f64,
    pub threshold: f64,
    pub s_token: f64,
}

impl DlcmConfig {
    /// Returns the default configuration for the 87M proxy model.
    #[must_use]
    pub const fn proxy_87m() -> Self {
        Self {
            vocab_size: 50257,
            d_token: 512,
            d_concept: 1024,
            d_scan: 512,
            enc_layers: 4,
            enc_heads: 8,
            enc_ff: 1024,
            concept_layers: 6,
            concept_heads: 16,
            concept_ff: 1024,
            dec_layers: 4,
            dec_heads: 8,
            dec_ff: 1024,
            alpha: 1.0,
            threshold: 0.5,
            s_token: 1.0,
        }
    }
}

// =====================================================================
// Core Attention & FFN Modules
// =====================================================================

/// Multi-Head Self-Attention module with Query-Key (QK) Normalization.
///
/// Applies RMSNorm to queries and keys before computing scaled dot-product attention,
/// ensuring stable training across heterogeneous widths.
pub struct Attention {
    rms_norm_q: RmsNorm,
    rms_norm_k: RmsNorm,
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Attention {
    /// Initializes a new self-attention layer.
    pub fn new(vb: VarBuilder, d_model: usize, num_heads: usize) -> Result<Self> {
        let head_dim = d_model / num_heads;
        Ok(Self {
            // QK-Norm is applied over the head_dim (last dimension after reshape)
            rms_norm_q: candle_nn::rms_norm(head_dim, 1e-5, vb.pp("rms_norm_q"))?,
            rms_norm_k: candle_nn::rms_norm(head_dim, 1e-5, vb.pp("rms_norm_k"))?,
            wq: linear_no_bias(d_model, d_model, vb.pp("W_q"))?,
            wk: linear_no_bias(d_model, d_model, vb.pp("W_k"))?,
            wv: linear_no_bias(d_model, d_model, vb.pp("W_v"))?,
            wo: linear_no_bias(d_model, d_model, vb.pp("W_o"))?,
            num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    /// Performs the forward pass for self-attention.
    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b_sz, seq_len, d_model) = x.dims3()?;
        
        let q = self.wq.forward(x)?.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?;
        let k = self.wk.forward(x)?.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?;
        let v = self.wv.forward(x)?.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?;

        // Apply QK norm over the head_dim (last dimension)
        let q = self.rms_norm_q.forward(&q)?;
        let k = self.rms_norm_k.forward(&k)?;

        // Make tensors contiguous after transpose for CUDA matmul
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let mut attn_weights = q.matmul(&k.t()?)?;
        attn_weights = (attn_weights * self.scale)?;
        
        if let Some(mask) = mask {
            attn_weights = attn_weights.broadcast_add(mask)?;
        }
        
        let attn_probs = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_probs.matmul(&v)?;
        
        // Also make contiguous after the final transpose before reshaping
        let out = attn_output.transpose(1, 2)?.contiguous()?.reshape((b_sz, seq_len, d_model))?;
        self.wo.forward(&out)
    }
}

/// Cross-Attention module bridging the token and concept latent spaces.
///
/// Queries are derived from the token-level encoder, while Keys and Values
/// are derived from the compressed concept-level reasoning backbone.
pub struct CrossAttention {
    rms_norm_q: RmsNorm,
    rms_norm_k: RmsNorm,
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl CrossAttention {
    /// Initializes a new cross-attention layer.
    pub fn new(vb: VarBuilder, d_token: usize, d_concept: usize, num_heads: usize) -> Result<Self> {
        // In cross-attention, Q is from tokens (d_token), K/V are projected to d_token.
        // So the head_dim for both Q and K/V is d_token / num_heads.
        let head_dim = d_token / num_heads;
        Ok(Self {
            rms_norm_q: candle_nn::rms_norm(head_dim, 1e-5, vb.pp("rms_norm_q"))?,
            rms_norm_k: candle_nn::rms_norm(head_dim, 1e-5, vb.pp("rms_norm_k"))?,
            wq: linear_no_bias(d_token, d_token, vb.pp("W_q"))?,
            wk: linear_no_bias(d_concept, d_token, vb.pp("W_k"))?,
            wv: linear_no_bias(d_concept, d_token, vb.pp("W_v"))?,
            wo: linear_no_bias(d_token, d_token, vb.pp("W_o"))?,
            num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    /// Performs the forward pass for cross-attention.
    pub fn forward(&self, q_in: &Tensor, kv_in: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, _) = q_in.dims3()?;
        let (_, kv_len, _) = kv_in.dims3()?;

        let q = self.wq.forward(q_in)?.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?;
        let k = self.wk.forward(kv_in)?.reshape((b_sz, kv_len, self.num_heads, self.head_dim))?;
        let v = self.wv.forward(kv_in)?.reshape((b_sz, kv_len, self.num_heads, self.head_dim))?;

        // Apply QK norm over the head_dim (last dimension)
        let q = self.rms_norm_q.forward(&q)?;
        let k = self.rms_norm_k.forward(&k)?;

        // Make tensors contiguous after transpose for CUDA matmul
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let mut attn_weights = q.matmul(&k.t()?)?;
        attn_weights = (attn_weights * self.scale)?;
        
        let attn_probs = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_probs.matmul(&v)?;
        
        // Also make contiguous after the final transpose before reshaping
        let out = attn_output.transpose(1, 2)?.contiguous()?.reshape((b_sz, seq_len, q_in.dim(2)?))?;
        self.wo.forward(&out)
    }
}

/// Feed-Forward Network (FFN) utilizing the SwiGLU activation function.
pub struct Mlp {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl Mlp {
    /// Initializes a new SwiGLU MLP layer.
    pub fn new(vb: VarBuilder, d_model: usize, d_ff: usize) -> Result<Self> {
        Ok(Self {
            w1: linear_no_bias(d_model, d_ff, vb.pp("w1"))?,
            w2: linear_no_bias(d_ff, d_model, vb.pp("w2"))?,
            w3: linear_no_bias(d_model, d_ff, vb.pp("w3"))?,
        })
    }

    /// Performs the forward pass for the SwiGLU MLP.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // SwiGLU activation: silu(x @ w1) * (x @ w3) @ w2
        let x1 = self.w1.forward(x)?.silu()?;
        let x3 = self.w3.forward(x)?;
        let x_gate = (x1 * x3)?;
        self.w2.forward(&x_gate)
    }
}

// =====================================================================
// Transformer Blocks
// =====================================================================

/// Standard Pre-Norm Transformer Block used in the Encoder and Concept Model.
pub struct TransformerBlock {
    attn: Attention,
    mlp: Mlp,
    norm1: RmsNorm,
    norm2: RmsNorm,
}

impl TransformerBlock {
    /// Initializes a new standard transformer block.
    pub fn new(vb: VarBuilder, d_model: usize, num_heads: usize, d_ff: usize) -> Result<Self> {
        Ok(Self {
            attn: Attention::new(vb.pp("attn"), d_model, num_heads)?,
            mlp: Mlp::new(vb.pp("mlp"), d_model, d_ff)?,
            norm1: candle_nn::rms_norm(d_model, 1e-5, vb.pp("norm1"))?,
            norm2: candle_nn::rms_norm(d_model, 1e-5, vb.pp("norm2"))?,
        })
    }

    /// Performs the forward pass with residual connections.
    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = x;
        let x_norm = self.norm1.forward(x)?;
        let x_attn = self.attn.forward(&x_norm, mask)?;
        let x = (x + x_attn)?;

        let residual = &x;
        let x_norm2 = self.norm2.forward(&x)?;
        let x_ffn = self.mlp.forward(&x_norm2)?;
        x + x_ffn
    }
}

/// Decoder Block utilizing only Cross-Attention to map concepts back to tokens.
pub struct DecoderBlock {
    cross_attn: CrossAttention,
    mlp: Mlp,
    norm1: RmsNorm,
    norm2: RmsNorm,
}

impl DecoderBlock {
    /// Initializes a new decoder block.
    pub fn new(vb: VarBuilder, d_token: usize, d_concept: usize, num_heads: usize, d_ff: usize) -> Result<Self> {
        Ok(Self {
            cross_attn: CrossAttention::new(vb.pp("cross_attn"), d_token, d_concept, num_heads)?,
            mlp: Mlp::new(vb.pp("mlp"), d_token, d_ff)?,
            norm1: candle_nn::rms_norm(d_token, 1e-5, vb.pp("norm1"))?,
            norm2: candle_nn::rms_norm(d_token, 1e-5, vb.pp("norm2"))?,
        })
    }

    /// Performs the forward pass using replicated concept representations.
    pub fn forward(&self, x: &Tensor, c_rep: &Tensor, _self_attn_mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = x;
        let x_norm = self.norm1.forward(x)?;
        let x_attn = self.cross_attn.forward(&x_norm, c_rep)?;
        let x = (x + x_attn)?;

        let residual = &x;
        let x_norm2 = self.norm2.forward(&x)?;
        let x_ffn = self.mlp.forward(&x_norm2)?;
        x + x_ffn
    }
}

// =====================================================================
// Main DLCM Model
// =====================================================================

/// The complete Dynamic Large Concept Model (DLCM).
///
/// Orchestrates the 4-stage pipeline:
/// 1. Token-Level Encoding
/// 2. Dynamic Segmentation & Concept Pooling
/// 3. Concept-Level Reasoning
/// 4. Token-Level Decoding via Cross-Attention
pub struct DlcmModel {
    embedding: Embedding,
    encoder_layers: Vec<TransformerBlock>,
    seg_wq: Linear, 
    seg_wk: Linear, 
    seg_wup: Linear,
    concept_layers: Vec<TransformerBlock>,
    decoder_layers: Vec<DecoderBlock>,
    final_norm: RmsNorm,
    lm_head: Linear,
    config: DlcmConfig,
}

impl DlcmModel {
    /// Initializes the full DLCM architecture from a `VarBuilder`.
    pub fn new(vb: VarBuilder, config: DlcmConfig) -> Result<Self> {
        let d_token = config.d_token;
        let d_concept = config.d_concept;
        let d_scan = config.d_scan;

        let mut encoder_layers = Vec::with_capacity(config.enc_layers);
        let enc_vb = vb.pp("encoder.layers");
        for i in 0..config.enc_layers {
            encoder_layers.push(TransformerBlock::new(enc_vb.pp(i.to_string()), d_token, config.enc_heads, config.enc_ff)?);
        }

        let seg_vb = vb.pp("segmentation");
        let seg_wq = linear_no_bias(d_token, d_scan, seg_vb.pp("W_q"))?;
        let seg_wk = linear_no_bias(d_token, d_scan, seg_vb.pp("W_k"))?;
        let seg_wup = linear_no_bias(d_token, d_concept, seg_vb.pp("W_up"))?;

        let mut concept_layers = Vec::with_capacity(config.concept_layers);
        let concept_vb = vb.pp("concept_model.layers");
        for i in 0..config.concept_layers {
            concept_layers.push(TransformerBlock::new(concept_vb.pp(i.to_string()), d_concept, config.concept_heads, config.concept_ff)?);
        }

        let mut decoder_layers = Vec::with_capacity(config.dec_layers);
        let dec_vb = vb.pp("decoder.layers");
        for i in 0..config.dec_layers {
            decoder_layers.push(DecoderBlock::new(dec_vb.pp(i.to_string()), d_token, d_concept, config.dec_heads, config.dec_ff)?);
        }

        let final_norm = candle_nn::rms_norm(d_token, 1e-5, vb.pp("decoder.norm"))?;
        let lm_head = linear_no_bias(d_token, config.vocab_size, vb.pp("lm_head"))?;

        Ok(Self {
            embedding: candle_nn::embedding(config.vocab_size, d_token, vb.pp("embedding"))?,
            encoder_layers,
            seg_wq, seg_wk, seg_wup,
            concept_layers,
            decoder_layers,
            final_norm,
            lm_head,
            config,
        })
    }

    /// Executes the full forward pass for prefill or training.
    pub fn forward(&self, input_ids: &Tensor, training: bool) -> Result<(Tensor, Tensor, Tensor)> {
        let b_sz = input_ids.dim(0)?;
        let seq_len = input_ids.dim(1)?;
        let device = input_ids.device();

        let mask = Self::create_causal_mask(seq_len, device)?;

        let mut h = self.embedding.forward(input_ids)?;

        for layer in &self.encoder_layers {
            h = layer.forward(&h, Some(&mask))?;
        }

        let (mut c, b, p) = dynamic_segmentation_and_pooling(
            &h,
            self.seg_wq.weight(),
            self.seg_wk.weight(),
            self.seg_wup.weight(),
            self.config.alpha,
            self.config.threshold,
            training,
        )?;

        let m_len = c.dim(1)?;
        let concept_mask = Self::create_causal_mask(m_len, device)?;
        for layer in &self.concept_layers {
            c = layer.forward(&c, Some(&concept_mask))?;
        }

        let segment_lengths = Self::compute_segment_lengths(&b, seq_len)?;
        let c_reshaped = c.unsqueeze(2)?; 
        let (c_rep, _) = concept_replication(&c_reshaped, &c_reshaped, &segment_lengths, seq_len)?;
        let c_rep = c_rep.squeeze(2)?; 

        let mut h_dec = h;
        for layer in &self.decoder_layers {
            h_dec = layer.forward(&h_dec, &c_rep, Some(&mask))?;
        }

        let h_norm = self.final_norm.forward(&h_dec)?;
        let logits = self.lm_head.forward(&h_norm)?;
        let logits_scaled = (logits * (1.0 / self.config.s_token))?;

        Ok((logits_scaled, b, p))
    }

    /// Wrapper for autoregressive generation. Extracts logits for the last token.
    pub fn forward_step(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (logits, _b, _p) = self.forward(input_ids, false)?;
        let seq_len = logits.dim(1)?;
        
        // 1. narrow(1, seq_len-1, 1) -> [B, 1, V]
        // 2. squeeze(1) -> [B, V]
        // 3. squeeze(0) -> [V] (1D tensor expected by the sampler)
        logits.narrow(1, seq_len.saturating_sub(1), 1)?.squeeze(1)?.squeeze(0) 

    }

    /// Computes the number of tokens belonging to each concept segment.
    fn compute_segment_lengths(b: &Tensor, seq_len: usize) -> Result<Tensor> {
        let segment_ids = (b.cumsum(1)? - 1.0)?.to_dtype(DType::U32)?;
        let max_m = segment_ids.max_all()?.to_scalar::<u32>()? as usize + 1;
        let b_size = b.dim(0)?;
        
        let ones = Tensor::ones((b_size, seq_len), DType::F32, b.device())?;
        let counts = Tensor::zeros((b_size, max_m), DType::F32, b.device())?
            .scatter_add(&segment_ids, &ones, 1)?;
         
        // Cast to I64 because concept_replication expects i64
        counts.to_dtype(DType::I64)

    }

    /// Generates a standard causal attention mask for autoregressive decoding.
    fn create_causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
        let mask: Vec<Vec<f32>> = (0..seq_len)
            .map(|i| (0..seq_len).map(|j| if j > i { f32::NEG_INFINITY } else { 0.0 }).collect())
            .collect();
        Tensor::new(mask, device)?.unsqueeze(0)?.unsqueeze(0)
    }
}


