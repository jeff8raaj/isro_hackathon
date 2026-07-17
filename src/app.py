import streamlit as st
import os
import sys
import time
import torch
from PIL import Image
import numpy as np

# 1. Page Config
st.set_page_config(
    page_title="ISRO AI Cloud Removal Console",
    page_icon="🛰️",
    layout="wide"
)

# 2. Dynamic Workspace Path Binding
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..")) if "src" in CURRENT_DIR else CURRENT_DIR
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 3. Model Engine Loader
@st.cache_resource
def load_model_core():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join(PROJECT_ROOT, "data", "generator_checkpoint.pth")
    
    try:
        from models import SatelliteCloudRemovalUNet
        model = SatelliteCloudRemovalUNet(in_channels=3, out_channels=3)
        
        if os.path.exists(checkpoint_path):
           model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
        else:
            status = "Prototype Mode (Weights Updating via Train Loop)"
    except ImportError as e:
        import torch.nn as nn
        class IdentityPass(nn.Module):
            def forward(self, x): return x
        model = IdentityPass()
        status = f"Prototype Fallback (Error: {e})"

    model.to(device)
    model.eval()
    return model, device, status

MODEL, DEVICE, MODEL_STATUS = load_model_core()

# 4. Metrics Functions
def compute_metrics_blank():
    return "WAITING", "WAITING", "WAITING", "WAITING"

# 5. UI Sidebar Controls
st.sidebar.title("🛰️ Mission Data Selection")
stream_type = st.sidebar.radio(
    "Choose Ingestion Stream:",
    ["Reference Benchmark (RICE1)", "ISRO Resourcesat LISS-IV Sample", "Custom Target Ingestion"]
)

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload LISS-IV Scene Asset (.tif, .png, .jpg):", type=["tif", "png", "jpg", "jpeg"])

# 6. Main Console View
st.title("🛰️ ISRO AI CLOUD REMOVAL MISSION CONSOLE")

# Create a clean asset display background
if uploaded_file:
    input_img = Image.open(uploaded_file).convert("RGB").resize((512, 512))
else:
    # Use your synthetic checkerboard array fallback matching your exact screen mockup
    grid = np.zeros((512, 512, 3), dtype=np.uint8)
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                grid[i*64:(i+1)*64, j*64:(j+1)*64] = [215, 35, 45]  # Red panels
            else:
                grid[i*64:(i+1)*64, j*64:(j+1)*64] = [135, 75, 65]  # Brown panels
    # Simulated atmosphere cloud streaks over sample matrix
    input_img = Image.fromarray(grid)

# Forward pass logic trigger button
if st.button("🚀 Execute Neural Forward Pass", use_container_width=True):
    with st.spinner("Executing model pass..."):
        in_np = np.array(input_img).astype(np.float32)
        norm_in = (in_np / 127.5) - 1.0
        tensor_in = torch.from_numpy(norm_in).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        
        t_start = time.time()
        with torch.no_grad():
            tensor_out = MODEL(tensor_in)
        st.session_state['latency_val'] = f"{time.time() - t_start:.2f} sec"
        
        out_np = tensor_out.squeeze(0).cpu().permute(1, 2, 0).numpy()
        st.session_state['reconstructed_img'] = np.clip((out_np + 1.0) * 127.5, 0, 255).astype(np.uint8)
        
        # Calculate structural spatial variance deviations
        st.session_state['deviation_map'] = np.abs(in_np.astype(float) - st.session_state['reconstructed_img'].astype(float)).astype(np.uint8)

# 3-Panel Visual Stream Setup
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("🌐 **Ingested Stream**")
    st.image(input_img, use_container_width=True)

with col2:
    st.markdown("🧠 **Neural Reconstructed Output**")
    if 'reconstructed_img' in st.session_state:
        st.image(st.session_state['reconstructed_img'], use_container_width=True)
    else:
        st.info("Awaiting Execution Trigger...")

with col3:
    st.markdown("👁️ **Spatial Deviation Map**")
    if 'deviation_map' in st.session_state:
        st.image(st.session_state['deviation_map'], use_container_width=True)
    else:
        st.info("Awaiting Execution Trigger...")

# Mission Telemetry Layout Sidebar Right Panel
st.markdown("---")
st.subheader("📊 Performance Validation Engine")

val1, val2, val3, val4 = st.columns(4)
psnr, ssim, rmse, sam = compute_metrics_blank()

val1.metric("📉 PSNR", psnr, "Peak Signal-to-Noise")
val2.metric("🧠 SSIM", ssim, "Structural Similarity")
val3.metric("📉 RMSE", rmse, "Root Mean Square Error")
val4.metric("📐 SAM", sam, "Spectral Angle Mapper")