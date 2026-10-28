//! # DLCM Inference CLI
//!
//! Standalone command-line interface for generating text using the Dynamic Large Concept Model (DLCM).
//! It loads `safetensors` weights, initializes the Candle-based CUDA/CPU backend, and performs
//! autoregressive generation with configurable sampling parameters.

use clap::Parser;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use dlcm_inference::model::{DlcmConfig, DlcmModel};
use dlcm_inference::sampler::Sampler;
use tokenizers::Tokenizer;
use std::time::Instant;

/// Command-line arguments for the DLCM inference engine.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the model weights in `safetensors` format.
    #[arg(long, default_value = "dlcm_proxy.safetensors")]
    weights: String,
    
    /// The initial text prompt to seed the generation.
    #[arg(long, default_value = "The future of artificial intelligence is")]
    prompt: String,
    
    /// Maximum number of tokens to generate.
    #[arg(long, default_value_t = 50)]
    max_tokens: usize,
    
    /// Sampling temperature. Higher values increase randomness.
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,
}


/// Main entry point for the inference engine.
///
/// # Errors
/// Returns a `candle_core::Error` if model loading, tokenization, or tensor operations fail.
#[allow(
    clippy::indexing_slicing,
    clippy::cast_precision_loss,
    clippy::as_conversions
)]
fn main() -> candle_core::Result<()> {
    let args = Args::parse();
    
    let device = Device::cuda_if_available(0)?;
    println!("🖥️  Using device: {device:?}");
    
    // 1. Load Tokenizer
    let tokenizer = Tokenizer::from_pretrained("gpt2", None)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to load tokenizer: {e}")))?;
    
    // 2. Load Model Weights (Memory-mapped for instant loading)
    let weights_path = &args.weights;
    println!("📥 Loading weights from {weights_path}...");
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], candle_core::DType::F32, &device)? };
    
    let config = DlcmConfig::proxy_87m();
    let model = DlcmModel::new(vb, config)?;
    
    // 3. Encode Prompt
    let prompt_str = args.prompt.as_str();
    let encoding = tokenizer.encode(prompt_str, true)
        .map_err(|e| candle_core::Error::Msg(format!("Tokenizer error: {e}")))?;
    let mut tokens = encoding.get_ids().to_vec();
    
    println!("🚀 Starting generation...\n{prompt_str} ");
    
    let mut sampler = Sampler::new(args.temperature, 0.9, 50);
    let start_time = Instant::now();
    
    // 4. Autoregressive Generation Loop
    for _ in 0..args.max_tokens {
        let context_size = tokens.len().min(512); // Max seq len
        let start_pos = tokens.len().saturating_sub(context_size);
        
        // Use .get() to safely slice without triggering clippy::indexing_slicing panics
        let context_tokens = tokens.get(start_pos..).unwrap_or(&[]);
        
        let input_tensor = Tensor::new(context_tokens, &device)?.unsqueeze(0)?;
        
        // Forward pass (returns logits for the last token)
        let logits = model.forward_step(&input_tensor)?;
        
        // Sample next token
        let next_token = sampler.sample(&logits)?;
        tokens.push(next_token);
        
        // Decode and print
        let decoded = tokenizer.decode(&[next_token], false)
            .map_err(|e| candle_core::Error::Msg(format!("Decode error: {e}")))?;
        print!("{decoded}");
        
        // Stop on EOS
        if next_token == 50256 { // GPT-2 EOS token
            break;
        }
    }
    
    let elapsed = start_time.elapsed();
    let tokens_per_sec = args.max_tokens as f64 / elapsed.as_secs_f64();
    let max_tokens = args.max_tokens;
    let secs = elapsed.as_secs_f64();
    println!("\n\n⚡ Generated {max_tokens} tokens in {secs:.2}s ({tokens_per_sec:.1} tok/s)");
    
    Ok(())
}



