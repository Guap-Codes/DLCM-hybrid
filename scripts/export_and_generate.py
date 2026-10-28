# scripts/export_and_generate.py
import subprocess
import os
from safetensors import safe_open

def main():
    weights_path = "/content/dlcm_proxy_trained.safetensors"
    
    print("🔍 Inspecting safetensors keys for 'norm', 'lm_head', 'embedding'...")
    try:
        with safe_open(weights_path, framework="pt", device="cpu") as f:
            all_keys = sorted(f.keys())
            
            print("\n--- ALL Keys containing 'norm', 'lm_head', or 'embedding' ---")
            for k in all_keys:
                if "norm" in k.lower() or "lm_head" in k.lower() or "embedding" in k.lower():
                    print(f"  {k}")
            
            print("\n--- Last 20 Keys in Safetensors ---")
            for k in all_keys[-20:]:
                print(f"  {k}")
            print("-" * 40 + "\n")
    except Exception as e:
        print(f"⚠️ Could not inspect weights: {e}")

    binary_path = "./target/release/dlcm-inference"
    
    if not os.path.exists(binary_path):
        print("🦀 Compiling Rust Inference Engine...")
        subprocess.run(["cargo", "build", "--release", "--manifest-path", "rust/dlcm-inference/Cargo.toml"], check=True)
    
    print("⚡ Running Rust Inference Engine on Trained Weights...\n")
    prompt = "The concept of dynamic compression in neural networks is"
    
    result = subprocess.run(
        [
            binary_path, 
            "--weights", weights_path, 
            "--prompt", prompt,
            "--max-tokens", "50",
            "--temperature", "0.8"
        ],
        capture_output=True,
        text=True
    )
    
    print("="*50)
    print("🤖 GENERATED TEXT (Rust Backend):")
    print("="*50)
    print(result.stdout)
    
    if result.returncode != 0:
        print("❌ Rust execution failed:")
        print(result.stderr)
        raise RuntimeError("Inference failed")

if __name__ == "__main__":
    main()
