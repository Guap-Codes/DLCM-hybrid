// rust/dlcm-core/src/replication.rs
use candle_core::{DType, Device, Result, Tensor};

/// Replicates concept Key/Value vectors to match token sequence length.
/// 
/// This is the Rust equivalent of `torch.repeat_interleave`, optimized for 
/// the specific DLCM use case of preparing tensors for `FlashAttention` Varlen.
/// 
/// # Arguments
/// * `k` - Concept Key states `[B, M, num_heads, d_head]`
/// * `v` - Concept Value states `[B, M, num_heads, d_head]`
/// * `segment_lengths` - Number of tokens per concept `[B, M]`
/// * `target_l` - The target token sequence length `L`
/// 
/// # Returns
/// A tuple of `(K_rep, V_rep)` both of shape `[B, L, num_heads, d_head]`
///
/// # Errors
/// Returns a `candle_core::Error` if tensor shapes are incompatible, 
/// if device allocations fail, or if numerical operations encounter invalid states.
#[allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::indexing_slicing,
    clippy::needless_range_loop
)]
pub fn concept_replication(
    k: &Tensor,
    v: &Tensor,
    segment_lengths: &Tensor,
    target_l: usize,
) -> Result<(Tensor, Tensor)> {
    let device = k.device();
    let (b_size, m_size, num_heads, d_head) = k.dims4()?;

    // 1. Extract segment lengths to CPU to build the index mapping
    // This is a lightweight operation as the index tensor is small (B x L)
    let seg_lens_vec: Vec<Vec<i64>> = segment_lengths.to_vec2()?;
    let mut indices_vec = Vec::with_capacity(b_size * target_l);
    
    for b in 0..b_size {
        let mut current_l = 0;
        for m in 0..m_size {
            let len = seg_lens_vec[b][m] as usize;
            // Clamp to ensure we don't exceed target_l due to floating point rounding errors
            let actual_len = len.min(target_l.saturating_sub(current_l));
            for _ in 0..actual_len {
                indices_vec.push(m as i64);
                current_l += 1;
            }
        }
        // Safety padding: if lengths sum to < target_l, repeat the last concept
        while current_l < target_l {
            indices_vec.push((m_size.saturating_sub(1)) as i64);
            current_l += 1;
        }
    }

    // 2. Create the base index tensor [B, L]
    let indices_tensor = Tensor::from_vec(indices_vec, (b_size, target_l), device)?.to_dtype(DType::I64)?;

    // 3. Flatten K and V to [B * M, num_heads, d_head] for 1D index selection
    let k_flat = k.flatten(0, 1)?;
    let v_flat = v.flatten(0, 1)?;
    
    // 4. Add batch offsets to the indices so they correctly map to the flattened tensor
    // e.g., Batch 1's concept 0 is at index `m_size` in the flattened array
    let batch_offsets = (0..b_size).map(|b| (b * m_size) as i64).collect::<Vec<_>>();
    let offsets_tensor = Tensor::from_vec(batch_offsets, (b_size, 1), device)?.to_dtype(DType::I64)?;
    
    // Use broadcast_add instead of + to handle [B, L] + [B, 1] shapes
    let flat_indices = indices_tensor.broadcast_add(&offsets_tensor)?.flatten(0, 1)?.to_dtype(DType::U32)?;
    
    // 5. Perform the replication via highly optimized index_select
    let k_rep_flat = k_flat.index_select(&flat_indices, 0)?;
    let v_rep_flat = v_flat.index_select(&flat_indices, 0)?;
    
    // 6. Reshape back to [B, L, num_heads, d_head]
    let k_rep = k_rep_flat.reshape((b_size, target_l, num_heads, d_head))?;
    let v_rep = v_rep_flat.reshape((b_size, target_l, num_heads, d_head))?;

    Ok((k_rep, v_rep))
}

