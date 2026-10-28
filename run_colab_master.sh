#!/bin/bash
set -e

SESSION_NAME="dlcm-full-pipeline"

# =====================================================================
# 1. Securely read HF_TOKEN from local .env file
# =====================================================================
HF_TOKEN_VAL=""
if [ -f .env ]; then
    HF_TOKEN_VAL=$(grep -E '^HF_TOKEN=' .env | cut -d'=' -f2- | tr -d '\r')
    echo "🔑 Found HF_TOKEN in local .env file."
else
    echo "⚠️ No .env file found."
fi

echo "🧹 Cleaning up any previous sessions..."
colab stop -s $SESSION_NAME 2>/dev/null || true

echo "🚀 Step 1: Provisioning T4 GPU on Google Colab..."
colab new -s $SESSION_NAME --gpu T4

echo "📤 Step 2: Uploading project files..."
colab upload -s $SESSION_NAME dlcm-payload.tar.gz /content/dlcm-payload.tar.gz

echo "🏃 Step 3: Full Pipeline (Tests -> Training -> Inference)..."
cat << PYEOF | colab exec -s $SESSION_NAME
import os
import subprocess
import sys
import time
import glob

os.environ["HF_TOKEN"] = """$HF_TOKEN_VAL"""

print("📦 Installing system dependencies...")
subprocess.run(["apt-get", "update", "-qq"], check=True)
subprocess.run(["apt-get", "install", "-qq", "-y", "build-essential", "pkg-config", "libssl-dev", "patchelf"], check=True)

print("🐍 Installing Python dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytorch-lightning", "python-dotenv", "maturin", "safetensors", "tokenizers", "datasets", "transformers"], check=True)

print("🦀 Installing Rust toolchain...")
subprocess.run("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y", shell=True, check=True)
os.environ["PATH"] = f"{os.environ['HOME']}/.cargo/bin:" + os.environ["PATH"]

print("Extracting project files...")
subprocess.run(["tar", "-xzf", "/content/dlcm-payload.tar.gz", "-C", "/content"], check=True)

print("🛠️ Patching candle-kernels for CUDA 12.x...")
subprocess.run("cargo fetch --manifest-path /content/Cargo.toml", shell=True, check=True)
subprocess.run("""find ~/.cargo/registry/src -name "compatibility.cuh" -exec sed -i 's/__half __hmax_nan(__half a, __half b)/__half disabled_hmax_nan(__half a, __half b)/g' {} +""", shell=True, check=True)
subprocess.run("""find ~/.cargo/registry/src -name "compatibility.cuh" -exec sed -i 's/__half __hmin_nan(__half a, __half b)/__half disabled_hmin_nan(__half a, __half b)/g' {} +""", shell=True, check=True)
print("✅ Patch applied.")

print("⚙️ Compiling Python Rust Extension (dlcm-python)...")
process = subprocess.Popen(["maturin", "build", "--release", "--manifest-path", "/content/rust/dlcm-python/Cargo.toml"], cwd="/content")
process.wait()
if process.returncode != 0: raise RuntimeError("maturin build failed")

wheels = glob.glob("/content/target/wheels/*.whl")
subprocess.run([sys.executable, "-m", "pip", "install", "--force-reinstall", wheels[0]], check=True)
print("✅ Python extension installed.")

env = os.environ.copy()
env["PYTHONPATH"] = "/content/python"

def run_python_script(script_path, use_torchrun=False):
    print(f"\n{'='*60}\n🧪 RUNNING: {script_path}\n{'='*60}")
    if use_torchrun:
        cmd = ["torchrun", "--nproc_per_node=1", "--standalone", f"/content/{script_path}"]
    else:
        cmd = ["python3", "-u", f"/content/{script_path}"]
        
    process = subprocess.Popen(cmd, env=env, cwd="/content", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    process.wait()
    if process.returncode != 0: raise RuntimeError(f"Failed at {script_path}")
    print(f"\n✅ PASSED: {script_path}")

# 1. Unit Tests
for test in ["tests/python/test_nano_pipeline.py", "tests/python/test_shapes.py", "tests/python/test_global_parser.py", "tests/python/test_mup.py"]:
    run_python_script(test, use_torchrun=("global_parser" in test))

# 2. Proxy Training (Exports weights to /content/dlcm_proxy_trained.safetensors)
run_python_script("tests/python/train_proxy.py")

# 3. Compile Rust Inference Binary (Uses cached target/ from maturin build!)
print("\n" + "="*60)
print("🦀 COMPILING RUST INFERENCE ENGINE (dlcm-inference)")
print("="*60)
process = subprocess.Popen(["cargo", "build", "--release", "--manifest-path", "/content/rust/dlcm-inference/Cargo.toml"], cwd="/content")
process.wait()
if process.returncode != 0: raise RuntimeError("cargo build failed")
print("✅ Rust inference binary compiled.")

# 4. Run Inference
print("\n" + "="*60)
print("🤖 RUNNING RUST INFERENCE")
print("="*60)
run_python_script("scripts/export_and_generate.py")

print("\n" + "="*60)
print("🎉 FULL PIPELINE COMPLETED SUCCESSFULLY!")
print("="*60)
time.sleep(2)
PYEOF

echo "🧹 Step 4: Cleaning up and releasing the GPU..."
colab stop -s $SESSION_NAME
rm -f dlcm-payload.tar.gz
echo "✅ All done! The GPU has been released."
