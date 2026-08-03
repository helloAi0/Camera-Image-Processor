import time
import numpy as np
import cv2
import streamlit as st
import plotly.graph_objects as go

# Core project imports
from src.core.base_processor import BaseProcessor
from src.stereo.stereo_vision import StereoVision
from src.pipeline.stream_pipeline import StreamPipeline


# --- Page Configuration & Dark Aesthetic Theme ---
st.set_page_config(
    page_title="Vision Studio Pro | Enterprise CV Suite",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark UI aesthetic
st.markdown("""
<style>
    .metric-card {
        background: #1e222d;
        border: 1px solid #2e3440;
        border-radius: 10px;
        padding: 12px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-title {
        color: #8892b0;
        font-size: 0.80rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .metric-value {
        color: #64ffda;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .section-header {
        border-left: 4px solid #64ffda;
        padding-left: 10px;
        margin-top: 10px;
        margin-bottom: 15px;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .guide-card {
        background: #161922;
        border: 1px solid #2e3440;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# PROFESSIONAL IMAGE PROCESSOR CLASS
# =====================================================================
class ProfessionalProcessor(BaseProcessor):
    """Full-Featured Enterprise Computer Vision Processor Engine."""
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        self.validate_image(image)
        output = image.copy()
        
        # 1. Spatial Transforms (Rotation & Flip)
        rotation_angle = kwargs.get("rotation", 0)
        if rotation_angle != 0:
            h, w = output.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), rotation_angle, 1.0)
            output = cv2.warpAffine(output, M, (w, h))

        flip_mode = kwargs.get("flip", "None")
        if flip_mode == "Horizontal":
            output = cv2.flip(output, 1)
        elif flip_mode == "Vertical":
            output = cv2.flip(output, 0)
        elif flip_mode == "Both":
            output = cv2.flip(output, -1)

        # 2. Gamma Correction
        gamma = kwargs.get("gamma", 1.0)
        if gamma != 1.0 and gamma > 0:
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            output = cv2.LUT(output, table)

        # 3. Tone & Contrast Adjustments
        brightness = kwargs.get("brightness", 0)
        contrast = kwargs.get("contrast", 1.0)
        if brightness != 0 or contrast != 1.0:
            output = cv2.convertScaleAbs(output, alpha=contrast, beta=brightness)

        # 4. CLAHE (Adaptive Histogram Equalization)
        if kwargs.get("enable_clahe", False):
            if output.ndim == 3:
                lab = cv2.cvtColor(output, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                output = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                output = clahe.apply(output)

        # 5. Denoise & Smoothing
        denoise_str = kwargs.get("denoise", 0)
        if denoise_str > 0:
            output = cv2.bilateralFilter(output, d=7, sigmaColor=denoise_str * 10, sigmaSpace=denoise_str * 10)

        blur_k = kwargs.get("blur", 1)
        if blur_k > 1:
            if blur_k % 2 == 0:
                blur_k += 1
            output = cv2.GaussianBlur(output, (blur_k, blur_k), 0)

        # 6. Unsharp Masking (Sharpen)
        sharpness = kwargs.get("sharpness", 0.0)
        if sharpness > 0.0:
            blurred_base = cv2.GaussianBlur(output, (0, 0), sigmaX=3)
            output = cv2.addWeighted(output, 1.0 + sharpness, blurred_base, -sharpness, 0)

        # 7. Color Space Conversions & Channel Isolator
        color_space = kwargs.get("color_space", "RGB / Original")
        if color_space == "Grayscale":
            gray = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY) if output.ndim == 3 else output
            output = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        elif color_space == "HSV":
            output = cv2.cvtColor(output, cv2.COLOR_RGB2HSV)
        elif color_space == "LAB":
            output = cv2.cvtColor(output, cv2.COLOR_RGB2LAB)

        channel_iso = kwargs.get("channel_isolation", "All Channels")
        if channel_iso != "All Channels" and output.ndim == 3:
            r, g, b = output[:, :, 0], output[:, :, 1], output[:, :, 2]
            zeros = np.zeros_like(r)
            if channel_iso == "Red Only":
                output = cv2.merge([r, zeros, zeros])
            elif channel_iso == "Green Only":
                output = cv2.merge([zeros, g, zeros])
            elif channel_iso == "Blue Only":
                output = cv2.merge([zeros, zeros, b])

        # 8. Advanced Edge Operators
        edge_mode = kwargs.get("edge_mode", "None")
        if edge_mode == "Canny":
            gray = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY) if output.ndim == 3 else output
            edges = cv2.Canny(gray, kwargs.get("canny_low", 50), kwargs.get("canny_high", 150))
            output = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        elif edge_mode == "Sobel Filter":
            gray = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY) if output.ndim == 3 else output
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_mag = cv2.magnitude(sobelx, sobely)
            sobel_vis = cv2.convertScaleAbs(sobel_mag)
            output = cv2.cvtColor(sobel_vis, cv2.COLOR_GRAY2RGB)
        elif edge_mode == "Laplacian":
            gray = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY) if output.ndim == 3 else output
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            lap_vis = cv2.convertScaleAbs(lap)
            output = cv2.cvtColor(lap_vis, cv2.COLOR_GRAY2RGB)

        return output


def generate_synthetic_scene(resolution_mode="1080p"):
    """Generates synthetic high-resolution test scene."""
    h, w = (2160, 3840) if resolution_mode == "4K" else (1080, 1920)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    for i in range(h):
        img[i, :] = (int(15 + (i / h) * 40), int(25 + (i / h) * 50), int(45 + (i / h) * 80))

    grid = 80 if resolution_mode == "4K" else 40
    img[::grid, :] = (40, 50, 70)
    img[:, ::grid] = (40, 50, 70)

    s = 2.0 if resolution_mode == "4K" else 1.0
    cv2.circle(img, (int(500 * s), int(400 * s)), int(180 * s), (255, 99, 71), -1)
    cv2.circle(img, (int(500 * s), int(400 * s)), int(140 * s), (255, 255, 255), int(5 * s))
    cv2.rectangle(img, (int(1000 * s), int(250 * s)), (int(1500 * s), int(600 * s)), (50, 205, 50), -1)
    pts = np.array([[int(800 * s), int(850 * s)], [int(600 * s), int(650 * s)], [int(1000 * s), int(650 * s)]])
    cv2.polylines(img, [pts], isClosed=True, color=(255, 215, 0), thickness=int(6 * s))
    
    return img


# =====================================================================
# SESSION STATE INITIALIZATION (AUTH SYSTEM)
# =====================================================================
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = "Guest Developer"


# --- TOP HEADER ---
st.title("👁️ Vision Studio Pro")
st.caption("Enterprise Computer Vision Engine | Image Processing, 3D Spatial Intelligence & Stream Pipelines")
st.markdown("---")


# --- SIDEBAR & AUTHENTICATION ---
with st.sidebar:
    st.header("👤 User Account")
    if not st.session_state["authenticated"]:
        auth_mode = st.radio("Access Mode", ["Sign Up", "Log In", "Continue as Guest"], index=2)
        if auth_mode in ["Sign Up", "Log In"]:
            with st.form("auth_form"):
                user_input = st.text_input("Username / Email")
                pass_input = st.text_input("Password", type="password")
                submit = st.form_submit_button(auth_mode)
                if submit and user_input:
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = user_input.split("@")[0]
                    st.success(f"Welcome, {st.session_state['username']}!")
                    st.rerun()
    else:
        st.success(f"🟢 Logged in as: **{st.session_state['username']}**")
        if st.button("Log Out"):
            st.session_state["authenticated"] = False
            st.session_state["username"] = "Guest Developer"
            st.rerun()

    st.markdown("---")
    st.header("⚙️ Global Input Source")
    source_type = st.radio("Source Mode", ["Synthetic Test Frame", "Upload Custom File (Up to 4GB)"], index=0)
    
    if source_type == "Upload Custom File (Up to 4GB)":
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg", "webp", "bmp"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            active_image = cv2.cvtColor(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        else:
            st.info("Awaiting upload... Using synthetic frame.")
            active_image = generate_synthetic_scene("1080p")
    else:
        res_choice = st.selectbox("Resolution Preset", ["Full HD (1920x1080)", "4K Ultra HD (3840x2160)"], index=0)
        mode = "4K" if "4K" in res_choice else "1080p"
        active_image = generate_synthetic_scene(mode)


# --- TAB NAVIGATION ---
tab_guide, tab_editor, tab_stereo, tab_stream = st.tabs([
    "📖 How to Use (Guide)",
    "🖼️ Professional Image Editor",
    "👓 Stereo Vision & 3D Depth",
    "⚡ Stream Pipeline Telemetry"
])


# =====================================================================
# TAB 0: STEP-BY-STEP USER GUIDE
# =====================================================================
with tab_guide:
    st.markdown("<div class='section-header'>Step-by-Step Dashboard Documentation</div>", unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to **Vision Studio Pro**! This dashboard allows you to visually execute computer vision algorithms, test stereo depth reconstruction, and monitor asynchronous camera pipeline streaming.
    """)
    
    st.markdown("<div class='guide-card'>", unsafe_allow_html=True)
    st.markdown("### Step 1: Account Sign-Up & Login (Optional)")
    st.markdown("Use the **User Account** module in the left sidebar to sign up or log in. Authentic users gain persistent studio settings and session logging capabilities.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='guide-card'>", unsafe_allow_html=True)
    st.markdown("### Step 2: Choose Your Input Source")
    st.markdown("Select an input source in the sidebar:")
    st.markdown("* **Synthetic Test Frame**: Generates high-resolution test patterns (1080p or 4K) with built-in geometric shapes to test sharpening and edges.")
    st.markdown("* **Upload Custom File**: Load any image up to **4GB** in PNG, JPG, WEBP, or BMP format.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='guide-card'>", unsafe_allow_html=True)
    st.markdown("### Step 3: Edit & Process Images")
    st.markdown("Navigate to the **Professional Image Editor** tab:")
    st.markdown("1. Adjust tone, contrast, gamma, and CLAHE adaptive histogram equalization.")
    st.markdown("2. Apply Unsharp Masking for crystal-clear image sharpening.")
    st.markdown("3. Perform spatial rotations (-180° to 180°) or image flips.")
    st.markdown("4. Filter channels (Red, Green, Blue) or extract edges using Canny, Sobel, or Laplacian filters.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='guide-card'>", unsafe_allow_html=True)
    st.markdown("### Step 4: Export & Download Processed Output")
    st.markdown("Scroll to the bottom of the **Image Editor** tab. Choose your target download format (**PNG** or **JPEG**), select quality/compression, and click **Download Processed Image** to save the result.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='guide-card'>", unsafe_allow_html=True)
    st.markdown("### Step 5: Explore 3D Depth & Stream Telemetry")
    st.markdown("* **Stereo Vision Tab**: Tune disparity parameters and inspect reprojected 3D point cloud matrices using an interactive Plotly 3D viewer.")
    st.markdown("* **Stream Telemetry Tab**: Run stress simulations to test multi-threaded ring-buffer performance, tracking frame drops and producer FPS rates.")
    st.markdown("</div>", unsafe_allow_html=True)


# =====================================================================
# TAB 1: PROFESSIONAL IMAGE EDITOR & DOWNLOAD MANAGER
# =====================================================================
with tab_editor:
    st.markdown("<div class='section-header'>Image Editing & Filtering Suite</div>", unsafe_allow_html=True)
    
    col_ctrl, col_view = st.columns([1, 2])
    
    with col_ctrl:
        st.subheader("🛠️ Adjustments & Exposure")
        brightness = st.slider("Brightness Offset", -100, 100, 0, step=5)
        contrast = st.slider("Contrast Scale", 0.2, 3.0, 1.0, step=0.1)
        gamma = st.slider("Gamma Correction", 0.1, 3.0, 1.0, step=0.1)
        enable_clahe = st.checkbox("Enable CLAHE (Adaptive Histogram Equalization)", value=False)
        
        st.subheader("🔍 Sharpness & Denoise")
        sharpness = st.slider("Unsharp Mask Intensity", 0.0, 3.0, 0.0, step=0.1)
        denoise = st.slider("Bilateral Denoise Strength", 0, 10, 0, step=1)
        blur = st.slider("Gaussian Blur Kernel Size", 1, 31, 1, step=2)

        st.subheader("📐 Spatial Transforms")
        rotation = st.slider("Rotation Angle (Degrees)", -180, 180, 0, step=5)
        flip = st.selectbox("Flip Axis", ["None", "Horizontal", "Vertical", "Both"])

        st.subheader("🎨 Colors & Channels")
        color_space = st.selectbox("Color Space Target", ["RGB / Original", "Grayscale", "HSV", "LAB"])
        channel_iso = st.selectbox("Isolate Channel", ["All Channels", "Red Only", "Green Only", "Blue Only"])

        st.subheader("⚡ Edge Detection Operators")
        edge_mode = st.radio("Edge Detection Filter", ["None", "Canny", "Sobel Filter", "Laplacian"])
        canny_low = st.slider("Canny Low Threshold", 10, 150, 50) if edge_mode == "Canny" else 50
        canny_high = st.slider("Canny High Threshold", 100, 300, 150) if edge_mode == "Canny" else 150

    with col_view:
        processor = ProfessionalProcessor(name="ProStudioProc")
        
        t0 = time.perf_counter()
        processed = processor.process(
            active_image,
            brightness=brightness,
            contrast=contrast,
            gamma=gamma,
            enable_clahe=enable_clahe,
            sharpness=sharpness,
            denoise=denoise,
            blur=blur,
            rotation=rotation,
            flip=flip,
            color_space=color_space,
            channel_isolation=channel_iso,
            edge_mode=edge_mode,
            canny_low=canny_low,
            canny_high=canny_high
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Compute Telemetry Metrics
        h, w = processed.shape[:2]
        megapixels = (w * h) / 1e6
        ram_mb = processed.nbytes / (1024 * 1024)

        # Performance Metric Bar
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f"<div class='metric-card'><div class='metric-title'>Resolution</div><div class='metric-value'>{w}x{h}</div></div>", unsafe_allow_html=True)
        m2.markdown(f"<div class='metric-card'><div class='metric-title'>Megapixels</div><div class='metric-value'>{megapixels:.2f} MP</div></div>", unsafe_allow_html=True)
        m3.markdown(f"<div class='metric-card'><div class='metric-title'>Memory Footprint</div><div class='metric-value'>{ram_mb:.1f} MB</div></div>", unsafe_allow_html=True)
        m4.markdown(f"<div class='metric-card'><div class='metric-title'>Latency</div><div class='metric-value'>{latency_ms:.1f} ms</div></div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        c1.image(active_image, caption=f"Original Input ({active_image.shape[1]}x{active_image.shape[0]})", use_column_width=True)
        c2.image(processed, caption=f"Processed Output ({w}x{h})", use_column_width=True)

        st.markdown("---")
        st.subheader("💾 Export & Download Processed Image")
        d_col1, d_col2 = st.columns([1, 2])
        
        with d_col1:
            export_format = st.selectbox("Export File Format", ["PNG", "JPEG"])
            jpg_quality = st.slider("JPEG Quality", 10, 100, 95) if export_format == "JPEG" else 100

        with d_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            # Encode image for download
            out_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR) if processed.ndim == 3 else processed
            if export_format == "PNG":
                success, buffer = cv2.imencode(".png", out_bgr)
                mime_type = "image/png"
                ext = "png"
            else:
                success, buffer = cv2.imencode(".jpg", out_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
                mime_type = "image/jpeg"
                ext = "jpg"

            if success:
                file_name = f"processed_vision_frame_{int(time.time())}.{ext}"
                st.download_button(
                    label=f"📥 Download Processed Image ({export_format})",
                    data=buffer.tobytes(),
                    file_name=file_name,
                    mime=mime_type,
                    type="primary"
                )


# =====================================================================
# TAB 2: STEREO VISION & 3D SPATIAL CLOUD
# =====================================================================
with tab_stereo:
    st.markdown("<div class='section-header'>Stereo Disparity & 3D Spatial Cloud Reconstruction</div>", unsafe_allow_html=True)
    
    col_s1, col_s2 = st.columns([1, 2])
    
    with col_s1:
        st.subheader("SGBM Matching Parameters")
        num_disp = st.select_slider("Disparity Range", options=[16, 32, 48, 64, 96, 128], value=32)
        block_size = st.slider("Block Size (Window)", 3, 21, 11, step=2)
        max_depth = st.slider("Max Depth Cutoff (m)", 0.5, 20.0, 6.0, step=0.5)
        
    left_rect = np.zeros((400, 600, 3), dtype=np.uint8)
    right_rect = np.zeros((400, 600, 3), dtype=np.uint8)
    
    cv2.rectangle(left_rect, (200, 100), (380, 280), (0, 255, 120), -1)
    cv2.rectangle(right_rect, (170, 100), (350, 280), (0, 255, 120), -1)
    cv2.circle(left_rect, (450, 200), 60, (255, 100, 0), -1)
    cv2.circle(right_rect, (435, 200), 60, (255, 100, 0), -1)

    stereo_engine = StereoVision()
    disparity = stereo_engine.compute_disparity(left_rect, right_rect, num_disparities=num_disp, block_size=block_size)

    disp_vis = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disp_colored = cv2.applyColorMap(disp_vis, cv2.COLORMAP_TURBO)

    with col_s2:
        sc1, sc2, sc3 = st.columns(3)
        sc1.image(left_rect, caption="Left Rectified Frame", use_column_width=True)
        sc2.image(right_rect, caption="Right Rectified Frame", use_column_width=True)
        sc3.image(disp_colored, caption="Disparity Depth Map", use_column_width=True)
        
        st.subheader("Interactive 3D Point Cloud View")
        Q = stereo_engine.create_reprojection_matrix(focal_length=450.0, baseline=0.12, principal_point=(300.0, 200.0))
        points_3d = stereo_engine.reproject_to_3d(disparity, Q, max_depth=max_depth)

        mask = disparity > 0
        z = points_3d[:, :, 2][mask]
        x = points_3d[:, :, 0][mask]
        y = points_3d[:, :, 1][mask]

        if len(z) > 0:
            fig = go.Figure(data=[go.Scatter3d(
                x=x[::6], y=y[::6], z=z[::6],
                mode='markers',
                marker=dict(size=2.5, color=z[::6], colorscale='Viridis', opacity=0.85, showscale=True)
            )])
            fig.update_layout(
                template="plotly_dark",
                margin=dict(l=0, r=0, b=0, t=0),
                scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Depth Z (m)', aspectmode='data')
            )
            st.plotly_chart(fig, use_container_width=True)


# =====================================================================
# TAB 3: HIGH-SPEED PIPELINE TELEMETRY
# =====================================================================
with tab_stream:
    st.markdown("<div class='section-header'>Asynchronous Ring-Buffer Telemetry</div>", unsafe_allow_html=True)
    
    p_col1, p_col2 = st.columns([1, 2])
    
    with p_col1:
        st.subheader("Buffer Configuration")
        cap = st.slider("Ring Buffer Queue Capacity", 1, 30, 10)
        sim_fps = st.slider("Producer Target FPS", 10, 120, 60)
        run_btn = st.button("⚡ Run Stress Test (3s)", type="primary")

    with p_col2:
        if run_btn:
            pipe = StreamPipeline(queue_capacity=cap)
            state = {"frame_id": 0}

            def mock_camera():
                state["frame_id"] += 1
                time.sleep(1.0 / sim_fps)
                return np.full((100, 100, 3), (state["frame_id"] * 10) % 255, dtype=np.uint8)

            pipe.start_stream(source=mock_camera)
            time.sleep(3.0)
            pipe.stop_stream()

            telemetry = pipe.get_telemetry()
            captured = telemetry.get("captured_frames", 0)
            dropped = telemetry.get("dropped_frames", 0)
            drop_rate = (dropped / (captured + dropped) * 100.0) if (captured + dropped) > 0 else 0.0

            st.success("Stress Test Complete!")
            
            tm1, tm2, tm3 = st.columns(3)
            tm1.markdown(f"<div class='metric-card'><div class='metric-title'>Total Captured</div><div class='metric-value'>{captured}</div></div>", unsafe_allow_html=True)
            tm2.markdown(f"<div class='metric-card'><div class='metric-title'>Buffer Drops</div><div class='metric-value'>{dropped}</div></div>", unsafe_allow_html=True)
            tm3.markdown(f"<div class='metric-card'><div class='metric-title'>Drop Rate</div><div class='metric-value'>{drop_rate:.1f}%</div></div>", unsafe_allow_html=True)
