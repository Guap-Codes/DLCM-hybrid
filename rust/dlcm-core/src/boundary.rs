use candle_core::{DType, Device, Result, Tensor};

/// Performs dynamic boundary detection and concept mean pooling.
/// 
/// # Arguments
/// * `h` - Encoder hidden states `[B, L, d_token]`
/// * `w_q` - Query projection weights `[d_scan, d_token]`
/// * `w_k` - Key projection weights `[d_scan, d_token]`
/// * `w_up` - Concept dimension projection weights `[d_concept, d_token]`
/// * `alpha` - Temperature for sharpening probabilities during training
/// * `threshold` - Hard threshold for inference
/// * `training` - Boolean flag to switch between Bernoulli sampling and hard thresholding
/// 
/// # Returns
/// A tuple of `(C, b, p)` where:
/// - `C`: Pooled concept representations `[B, M, d_concept]`
/// - `b`: Discrete boundary indicators `[B, L]`
/// - `p`: Continuous boundary probabilities `[B, L]`
///
/// # Errors
/// Returns a `candle_core::Error` if tensor shapes are incompatible, 
/// if device allocations fail, or if numerical operations encounter invalid states.
#[allow(
    clippy::many_single_char_names,
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn dynamic_segmentation_and_pooling(
    h: &Tensor,
    w_q: &Tensor,
    w_k: &Tensor,
    w_up: &Tensor,
    alpha: f64,
    threshold: f64,
    training: bool,
) -> Result<(Tensor, Tensor, Tensor)> {
    let device = h.device();
    let (b_size, seq_len, d_token) = h.dims3()?;

    // 1. Project to Query and Key space (Eq. 5)
    // CRITICAL FIX: Candle Linear weights are [out_features, in_features], so we must transpose them!
    let h_flat = h.reshape((b_size * seq_len, d_token))?;
    let q = h_flat.matmul(&w_q.t()?)?.reshape((b_size, seq_len, w_q.dim(0)?))?; // [B, L, d_scan]
    let k = h_flat.matmul(&w_k.t()?)?.reshape((b_size, seq_len, w_k.dim(0)?))?; // [B, L, d_scan]

    // 2. Shift Q by 1 to compare t with t-1
    let q_shifted = if seq_len > 1 {
        let q_0 = Tensor::zeros((b_size, 1, q.dim(2)?), q.dtype(), device)?;
        let q_rest = q.narrow(1, 0, seq_len.saturating_sub(1))?;
        Tensor::cat(&[&q_0, &q_rest], 1)?
    } else {
        q
    };

    // 3. Cosine Similarity (Numerically stable)
    let q_norm = q_shifted.broadcast_div(&(q_shifted.sqr()?.sum_keepdim(2)?.sqrt()? + 1e-8)?)?;
    let k_norm = k.broadcast_div(&(k.sqr()?.sum_keepdim(2)?.sqrt()? + 1e-8)?)?;
    let cos_sim = (q_norm * k_norm)?.sum_keepdim(2)?; // [B, L, 1]

    // 4. Boundary Probability: p_t = (1 - cos_sim) / 2 (Eq. 6)
    let p = (cos_sim.affine(-1.0, 1.0)? * 0.5)?;
    
    // Enforce p_0 = 1.0 (First token always starts a new concept)
    let p_0 = Tensor::ones((b_size, 1, 1), p.dtype(), device)?;
    let p_rest = p.narrow(1, 1, seq_len.saturating_sub(1))?;
    let p = Tensor::cat(&[&p_0, &p_rest], 1)?;

    // 5. Discrete Sampling (Section 3.3.1)
    let b = if training {
        let p_clamped = p.clamp(1e-6, 1.0 - 1e-6)?;
        let inv_alpha = 1.0 / alpha;
        
        // logits = log(p / (1 - p)) / alpha
        let one_minus_p = p_clamped.affine(-1.0, 1.0)?;
        let ratio = p_clamped.broadcast_div(&one_minus_p)?;
        let logits = (ratio.log()? * inv_alpha)?;
        
        // Manual sigmoid: 1 / (1 + exp(-x))
        let p_sharp = (logits.neg()?.exp()? + 1.0)?.recip()?;
        
        // Bernoulli sampling via reparameterization trick: noise < p_sharp
        let noise = Tensor::rand(0.0f32, 1.0f32, p_sharp.dims(), device)?.to_dtype(p_sharp.dtype())?;
        noise.lt(&p_sharp)?.to_dtype(DType::F32)?
    } else {
        // Hard thresholding for inference
        p.ge(threshold)?.to_dtype(DType::F32)?
    };

    // 6. Segment IDs via Cumulative Sum (Eq. 7)
    let segment_ids = (b.cumsum(1)? - 1.0)?;
    let segment_ids_u32 = segment_ids.to_dtype(DType::U32)?;
    
    // Determine max concepts (M) for this batch
    let max_m = segment_ids.max_all()?.to_scalar::<f32>()? as usize + 1;

    // 7. Mean Pooling via scatter_add
    let seg_ids_expanded = segment_ids_u32.expand((b_size, seq_len, d_token))?.contiguous()?;
    
    // Sum the hidden states per segment
    let sum_h = Tensor::zeros((b_size, max_m, d_token), h.dtype(), device)?
        .scatter_add(&seg_ids_expanded, h, 1)?;
    
    // Count tokens per segment to compute the mean
    let ones = Tensor::ones((b_size, seq_len, 1), DType::F32, device)?;
    let counts = Tensor::zeros((b_size, max_m, 1), DType::F32, device)?
        .scatter_add(&segment_ids_u32, &ones, 1)?; // [B, max_m, 1]
    
    // Compute mean: C_raw = sum_h / (counts + epsilon)
    let c_raw = sum_h.broadcast_div(&(counts + 1e-8)?)?;

    // 8. Project to concept dimension
    // Transpose w_up before matmul
    let c_raw_flat = c_raw.reshape((b_size * max_m, d_token))?;
    let c = c_raw_flat.matmul(&w_up.t()?)?.reshape((b_size, max_m, w_up.dim(0)?))?; // [B, max_m, d_concept]

    // Squeeze the last dimension of p and b to return [B, L] for Python compatibility
    let p_squeezed = p.squeeze(2)?;
    let b_squeezed = b.squeeze(2)?;

    Ok((c, b_squeezed, p_squeezed))
}




