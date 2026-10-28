//! # DLCM Python Bindings (`dlcm-python`)
//!
//! This crate provides PyO3 bindings for the core CUDA/CPU accelerated kernels
//! used in the Dynamic Large Concept Model (DLCM) PyTorch training pipeline.
//! 
//! It bridges the gap between PyTorch's tensor operations and the highly optimized
//! `candle-core` backend, enabling fast dynamic segmentation and concept replication
//! during the forward and backward passes of the DLCM training loop.

use pyo3::prelude::*;
use numpy::{PyReadonlyArrayDyn, PyArrayMethods, PyUntypedArrayMethods, IntoPyArray, PyArray};
use numpy::ndarray::IxDyn;
use candle_core::{Device, Tensor};
use dlcm_core::{dynamic_segmentation_and_pooling, concept_replication};

/// Python wrapper for the fused dynamic segmentation and pooling kernel.
///
/// This function implements the **Dynamic Segmentation** stage (Section 3.3 of the DLCM paper).
/// It computes semantic boundaries by measuring local dissimilarity between adjacent token
/// representations, samples discrete boundaries, and mean-pools the tokens within each
/// segment into a unified concept representation.
///
/// # Arguments
/// * `h` - Encoder hidden states `[B, L, d_token]`
/// * `wq` - Query projection weights `[d_scan, d_token]`
/// * `wk` - Key projection weights `[d_scan, d_token]`
/// * `wup` - Concept dimension projection weights `[d_concept, d_token]`
/// * `alpha` - Temperature for sharpening probabilities during training
/// * `threshold` - Hard threshold for inference boundary detection
/// * `training` - Boolean flag to switch between Bernoulli sampling and hard thresholding
///
/// # Returns
/// A tuple of `(C, b, p)` as NumPy arrays:
/// * `C`: Pooled concept representations `[B, M, d_concept]`
/// * `b`: Discrete boundary indicators `[B, L]`
/// * `p`: Continuous boundary probabilities `[B, L]`
#[pyfunction]
#[allow(
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::needless_pass_by_value,
    clippy::similar_names
)]
fn fused_boundary_op_py<'py>(
    py: Python<'py>,
    h: PyReadonlyArrayDyn<'py, f32>,
    wq: PyReadonlyArrayDyn<'py, f32>,
    wk: PyReadonlyArrayDyn<'py, f32>,
    wup: PyReadonlyArrayDyn<'py, f32>,
    alpha: f64,
    threshold: f64,
    training: bool,
) -> PyResult<(Bound<'py, PyArray<f32, IxDyn>>, Bound<'py, PyArray<f32, IxDyn>>, Bound<'py, PyArray<f32, IxDyn>>)> {
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);

    // Helper to convert PyReadonlyArrayDyn to Candle Tensor
    let array_to_tensor = |arr: &PyReadonlyArrayDyn<'py, f32>| -> PyResult<Tensor> {
        // Simplified: shape() returns &[usize], so .to_vec() is all we need
        let shape = arr.shape().to_vec();
        let vec = arr.as_slice()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            .to_vec();
        Tensor::from_vec(vec, shape, &device)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    };

    let h_tensor = array_to_tensor(&h)?;
    let wq_tensor = array_to_tensor(&wq)?;
    let wk_tensor = array_to_tensor(&wk)?;
    let wup_tensor = array_to_tensor(&wup)?;

    let (c, b, p) = dynamic_segmentation_and_pooling(&h_tensor, &wq_tensor, &wk_tensor, &wup_tensor, alpha, threshold, training)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    // Helper to convert Candle Tensor back to NumPy Array
    let tensor_to_pyarray = |t: &Tensor| -> PyResult<Bound<'py, PyArray<f32, IxDyn>>> {
        let shape = t.dims().to_vec();
        
        // Explicitly map candle_core::Error to PyErr before using `?`
        let flat_t = t.flatten_all()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let vec: Vec<f32> = flat_t.to_vec1::<f32>()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        
        let arr = numpy::ndarray::Array::from_shape_vec(IxDyn(&shape), vec)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        
        // into_pyarray directly returns the safe Bound<'py, PyArray<...>> type
        Ok(arr.into_pyarray(py))
    };

    let c_py = tensor_to_pyarray(&c)?;
    let b_py = tensor_to_pyarray(&b)?;
    let p_py = tensor_to_pyarray(&p)?;

    Ok((c_py, b_py, p_py))
}

/// Python wrapper for the concept replication kernel.
///
/// This function implements the **Concept Replication** strategy (Section 4.1 of the DLCM paper)
/// used to optimize the decoder's causal cross-attention. Because tokens must attend to
/// variable-length concepts, the resulting attention mask is irregular (L x M).
///
/// To enable the use of highly optimized Flash Attention kernels, this function replicates
/// (via `repeat_interleave`) the concept Keys and Values to match the token sequence length (L x L),
/// creating a standard causal mask where keys and values are locally constant within each concept segment.
///
/// # Arguments
/// * `k` - Concept Key states `[B, M, num_heads, d_head]`
/// * `v` - Concept Value states `[B, M, num_heads, d_head]`
/// * `seg_lens` - Number of tokens per concept `[B, M]`
/// * `target_l` - The target token sequence length `L`
///
/// # Returns
/// A tuple of `(K_rep, V_rep)` as NumPy arrays, both of shape `[B, L, num_heads, d_head]`
#[pyfunction]
#[allow(
    clippy::type_complexity,
    clippy::needless_pass_by_value
)]
fn concept_replication_py<'py>(
    py: Python<'py>,
    k: PyReadonlyArrayDyn<'py, f32>,
    v: PyReadonlyArrayDyn<'py, f32>,
    seg_lens: PyReadonlyArrayDyn<'py, i64>,
    target_l: usize,
) -> PyResult<(Bound<'py, PyArray<f32, IxDyn>>, Bound<'py, PyArray<f32, IxDyn>>)> {
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);

    let k_shape = k.shape().to_vec();
    let k_vec = k.as_slice().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?.to_vec();
    let k_tensor = Tensor::from_vec(k_vec, k_shape, &device)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let v_shape = v.shape().to_vec();
    let v_vec = v.as_slice().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?.to_vec();
    let v_tensor = Tensor::from_vec(v_vec, v_shape, &device)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let seg_lens_shape = seg_lens.shape().to_vec();
    let seg_lens_vec = seg_lens.as_slice().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?.to_vec();
    let seg_lens_tensor = Tensor::from_vec(seg_lens_vec, seg_lens_shape, &device)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let (k_rep, v_rep) = concept_replication(&k_tensor, &v_tensor, &seg_lens_tensor, target_l)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let tensor_to_pyarray = |t: &Tensor| -> PyResult<Bound<'py, PyArray<f32, IxDyn>>> {
        let shape = t.dims().to_vec();
        let flat_t = t.flatten_all()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let vec: Vec<f32> = flat_t.to_vec1::<f32>()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        
        let arr = numpy::ndarray::Array::from_shape_vec(IxDyn(&shape), vec)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        
        Ok(arr.into_pyarray(py))
    };

    let k_rep_py = tensor_to_pyarray(&k_rep)?;
    let v_rep_py = tensor_to_pyarray(&v_rep)?;

    Ok((k_rep_py, v_rep_py))
}

/// The Python module definition for `_rust_ops`.
///
/// Exposes the custom CUDA/CPU kernels to the Python training environment,
/// allowing PyTorch to call into the optimized Candle backend for the
/// most computationally intensive parts of the DLCM forward pass.
#[pymodule]
fn _rust_ops(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fused_boundary_op_py, m)?)?;
    m.add_function(wrap_pyfunction!(concept_replication_py, m)?)?;
    Ok(())
}


