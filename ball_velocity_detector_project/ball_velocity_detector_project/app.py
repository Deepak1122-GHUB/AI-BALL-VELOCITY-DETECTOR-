
import streamlit as st
import cv2
import time
import numpy as np
import math
import tempfile
import os
from pathlib import Path
from collections import deque
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Ball Velocity Detector",
    page_icon="🎾",
    layout="wide"
)

st.title("Ball Velocity Detector")
st.caption(
    "AI-Powered Ball Detection, Tracking and Velocity Estimation — experimental research build"
)

# ---------------- LOAD MODEL ----------------
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "ball_detector.pt"

@st.cache_resource
def load_model(model_path):
    # Ultralytics can load local weights or download supported official weights.
    return YOLO(str(model_path))

# ---------------- SIDEBAR ----------------
mode = st.sidebar.radio(
    "Choose Mode",
    ("Image", "Upload Video", "Webcam")
)

st.sidebar.subheader("Calibration")
ppm = st.sidebar.number_input(
    "Pixels per metre",
    min_value=1.0,
    max_value=10000.0,
    value=80.0,
    step=1.0,
    help="Enter a measured pixels-per-metre value for your camera setup."
)
st.sidebar.caption(
    "Speed accuracy depends on correct calibration and camera geometry."
)

st.sidebar.subheader("Calibration method")
calibration_method = st.sidebar.selectbox(
    "Distance calibration", ("Manual pixels per metre", "Four-point homography", "Known ball diameter")
)

with st.sidebar.expander("Four-point homography coordinates", expanded=False):
    st.caption("Enter four image points and matching flat-plane coordinates in metres. Coordinates must be in the same order (top-left, top-right, bottom-right, bottom-left). This maps a plane, not arbitrary 3D ball motion.")
    src_text = st.text_area("Image points x,y (one per line)", "100,100\n500,100\n500,400\n100,400", key="src_points")
    dst_text = st.text_area("Plane points X,Y in metres (one per line)", "0,0\n2,0\n2,1.5\n0,1.5", key="dst_points")

with st.sidebar.expander("Lens distortion correction", expanded=False):
    st.caption("Optional OpenCV camera matrix and distortion coefficients. Leave disabled unless measured with camera calibration images.")
    use_undistort = st.checkbox("Enable distortion correction", value=False)
    fx = st.number_input("Camera fx (pixels)", min_value=1.0, value=800.0)
    fy = st.number_input("Camera fy (pixels)", min_value=1.0, value=800.0)
    cx = st.number_input("Camera cx (pixels)", min_value=0.0, value=640.0)
    cy = st.number_input("Camera cy (pixels)", min_value=0.0, value=360.0)
    k1 = st.number_input("Distortion k1", value=0.0, format="%.6f")
    k2 = st.number_input("Distortion k2", value=0.0, format="%.6f")
    p1 = st.number_input("Distortion p1", value=0.0, format="%.6f")
    p2 = st.number_input("Distortion p2", value=0.0, format="%.6f")

st.sidebar.subheader("Detector / tracker comparison")
st.sidebar.caption("Run the same clip separately with YOLO11n, YOLO11s, YOLO11m and custom weights, then compare the displayed metrics. A fine-tuned model requires your labelled dataset and trained weights.")
model_choice = st.sidebar.selectbox("Detector weights", ("Custom ball_detector.pt", "YOLO11n (yolo11n.pt)", "YOLO11s (yolo11s.pt)", "YOLO11m (yolo11m.pt)", "Custom weights path"))
custom_model_path = st.sidebar.text_input("Custom weights path (if selected)", str(DEFAULT_MODEL_PATH))
tracker_method = st.sidebar.selectbox("Tracker", ("Proximity (single best detection)", "ByteTrack"), help="ByteTrack uses Ultralytics tracking. The proximity option follows the highest-confidence detection with a jump threshold.")

st.sidebar.subheader("Scale and blur experiments")
ball_diameter_m = st.sidebar.number_input("Known ball diameter (metres)", min_value=0.001, max_value=1.0, value=0.067, step=0.001, format="%.3f")
blur_length_px = st.sidebar.number_input("Measured blur streak length (pixels; 0 if unavailable)", min_value=0.0, max_value=5000.0, value=0.0, step=1.0)
exposure_seconds = st.sidebar.number_input("Exposure time (seconds)", min_value=0.000001, max_value=1.0, value=0.001, format="%.6f")
calibration_error_pct = st.sidebar.slider("Calibration uncertainty (%)", 0.0, 30.0, 5.0, 0.5)
localization_noise_px = st.sidebar.slider("Centre localisation uncertainty (pixels)", 0.1, 20.0, 2.0, 0.1)
timing_jitter_ms = st.sidebar.slider("Timing jitter (milliseconds)", 0.0, 30.0, 1.0, 0.5)

st.sidebar.subheader("Detection and Estimator Experiments")
confidence_threshold = st.sidebar.slider(
    "Detection confidence", min_value=0.10, max_value=0.90,
    value=0.40, step=0.05
)
estimator_method = st.sidebar.selectbox(
    "Velocity estimator",
    ("Finite difference", "Windowed linear regression", "Projectile-model fit", "Compare all")
)
smoothing_window = st.sidebar.slider(
    "Smoothing window (detections)", min_value=3, max_value=15,
    value=5, step=1
)
jump_threshold = st.sidebar.slider(
    "Maximum accepted centre jump (pixels)", min_value=20,
    max_value=1000, value=250, step=10,
    help="Large jumps may be false detections. Increase this for fast balls or wide shots."
)

tracking_method = st.sidebar.selectbox(
    "Position smoothing / tracking",
    ("Raw detections", "Kalman filter"),
    help="Kalman filtering smooths centre positions; it does not replace object identity tracking."
)
processing_width = st.sidebar.selectbox(
    "Detection resolution",
    ("Original", "640 px wide", "1280 px wide"),
    help="Lower resolution may run faster but can miss small/fast balls. Coordinates are mapped back to the original frame."
)
experiment_lighting = st.sidebar.selectbox("Lighting condition (record for sensitivity study)", ("Not recorded", "Bright outdoor", "Indoor bright", "Dim indoor", "Backlit"))
experiment_camera_angle = st.sidebar.selectbox("Camera angle (record for sensitivity study)", ("Not recorded", "Side-on", "Elevated", "Low angle", "Oblique/perspective"))
reference_speed_mps = st.sidebar.number_input("Reference speed for validation (m/s; 0 if unknown)", min_value=0.0, max_value=200.0, value=0.0, step=0.1)

model_map = {
    "Custom ball_detector.pt": str(DEFAULT_MODEL_PATH),
    "YOLO11n (yolo11n.pt)": "yolo11n.pt",
    "YOLO11s (yolo11s.pt)": "yolo11s.pt",
    "YOLO11m (yolo11m.pt)": "yolo11m.pt",
    "Custom weights path": custom_model_path,
}
selected_model_path = model_map[model_choice]
if model_choice == "Custom ball_detector.pt" and not DEFAULT_MODEL_PATH.exists():
    st.error(f"Model file not found: {DEFAULT_MODEL_PATH}")
    st.stop()
if model_choice == "Custom weights path" and not Path(selected_model_path).exists():
    st.error(f"Custom model file not found: {selected_model_path}")
    st.stop()
try:
    model = load_model(selected_model_path)
except Exception as exc:
    st.error(f"Could not load model '{selected_model_path}'. Check the weights file and internet access for official YOLO weights. Details: {exc}")
    st.stop()

# ---------------- HELPER FUNCTIONS ----------------
def draw_trajectory(points, frame, color=(0, 255, 255)):
    for i in range(1, len(points)):
        cv2.line(frame, points[i - 1], points[i], color, 2)
    return frame


def estimate_speed(p1, p2, dt, pixels_per_metre):
    """Finite-difference speed in m/s from two image-centre observations."""
    if dt <= 0 or pixels_per_metre <= 0:
        return 0.0
    dx = (p2[0] - p1[0]) / pixels_per_metre
    dy = (p2[1] - p1[1]) / pixels_per_metre
    return math.hypot(dx, dy) / dt


def estimate_history_speeds(history, pixels_per_metre, window=5):
    """Return speed estimates from (x, y, timestamp) observations.

    Projectile fit models image y as a quadratic in time; it is a 2D image-plane
    approximation, not a calibrated 3D ballistic reconstruction.
    """
    points = list(history)[-max(3, int(window)):]
    if len(points) < 2 or pixels_per_metre <= 0:
        return {}
    t = np.asarray([p[2] for p in points], dtype=float)
    x = np.asarray([p[0] for p in points], dtype=float)
    y = np.asarray([p[1] for p in points], dtype=float)
    t = t - t[-1]
    if len(points) >= 2 and t[-1] == t[0]:
        return {}

    # Finite difference uses the most recent two valid detections.
    dt = points[-1][2] - points[-2][2]
    finite = estimate_speed(points[-2][:2], points[-1][:2], dt, pixels_per_metre) if dt > 0 else None
    estimates = {}
    if finite is not None:
        estimates["Finite difference"] = finite

    if len(points) >= 3 and np.ptp(t) > 0:
        vx_px = float(np.polyfit(t, x, 1)[0])
        vy_px = float(np.polyfit(t, y, 1)[0])
        estimates["Windowed linear regression"] = math.hypot(vx_px, vy_px) / pixels_per_metre
        # Fit x(t) linearly and y(t) quadratically, then evaluate velocity at latest time (t=0).
        x_coef = np.polyfit(t, x, 1)
        y_coef = np.polyfit(t, y, 2)
        vx_proj = float(x_coef[0])
        vy_proj = float(y_coef[1])
        estimates["Projectile-model fit"] = math.hypot(vx_proj, vy_proj) / pixels_per_metre
    return estimates


class CentreKalman:
    """Small constant-velocity Kalman filter for image-plane centre positions."""
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 4.0
        self.initialized = False

    def update(self, point):
        measurement = np.array([[np.float32(point[0])], [np.float32(point[1])]])
        if not self.initialized:
            self.kf.statePost = np.array([[point[0]], [point[1]], [0], [0]], dtype=np.float32)
            self.initialized = True
            return int(point[0]), int(point[1])
        prediction = self.kf.predict()
        corrected = self.kf.correct(measurement)
        return int(corrected[0, 0]), int(corrected[1, 0])


def prepare_frame_for_detection(frame):
    """Resize for inference and return (frame_for_model, x_scale, y_scale)."""
    h, w = frame.shape[:2]
    target = None
    if processing_width == "640 px wide":
        target = 640
    elif processing_width == "1280 px wide":
        target = 1280
    if target is None or w == target:
        return frame, 1.0, 1.0
    scale = target / float(w)
    resized = cv2.resize(frame, (target, max(1, int(round(h * scale)))))
    return resized, w / float(target), h / float(resized.shape[0])


def calculate_uncertainty(values):
    """Descriptive spread of available estimates, not a validated confidence interval."""
    clean = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(clean) < 2:
        return None
    return float(np.std(clean, ddof=1))


def predict_landing(points):
    if len(points) < 2:
        return None

    x1, y1 = points[-2]
    x2, y2 = points[-1]

    dx = x2 - x1
    dy = y2 - y1

    return int(x2 + dx * 3), int(y2 + dy * 3)


def parse_xy_lines(text, expected=4):
    points = []
    for line in text.strip().splitlines():
        vals = [v.strip() for v in line.split(",")]
        if len(vals) != 2:
            raise ValueError("Each point must be written as x,y on its own line.")
        points.append([float(vals[0]), float(vals[1])])
    if len(points) != expected:
        raise ValueError(f"Exactly {expected} points are required.")
    return np.asarray(points, dtype=np.float32)


def get_homography():
    try:
        src = parse_xy_lines(src_text)
        dst = parse_xy_lines(dst_text)
        matrix = cv2.getPerspectiveTransform(src, dst)
        if abs(np.linalg.det(matrix)) < 1e-12:
            return None
        return matrix
    except Exception:
        return None


if calibration_method == "Four-point homography" and get_homography() is None:
    st.sidebar.warning("Homography coordinates are invalid; manual pixels-per-metre fallback will be used until four valid point pairs are entered.")


def map_point_to_metres(point, homography):
    mapped = cv2.perspectiveTransform(np.asarray([[[float(point[0]), float(point[1])]]], dtype=np.float32), homography)[0, 0]
    return float(mapped[0]), float(mapped[1])


def preprocess_camera(frame):
    if not use_undistort:
        return frame
    h, w = frame.shape[:2]
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    distortion = np.array([k1, k2, p1, p2, 0.0], dtype=np.float64)
    # Calibration matrix must correspond to this image resolution; users should calibrate it for the actual camera.
    return cv2.undistort(frame, camera_matrix, distortion)


def get_point_scale(center, box):
    if calibration_method == "Four-point homography":
        H = get_homography()
        if H is not None:
            return map_point_to_metres(center, H), "homography"
        return (center[0] / ppm, center[1] / ppm), "manual-fallback"
    if calibration_method == "Known ball diameter":
        x1, y1, x2, y2 = box
        apparent_diameter = max(1.0, float(max(x2 - x1, y2 - y1)))
        metres_per_pixel = ball_diameter_m / apparent_diameter
        return (center[0] * metres_per_pixel, center[1] * metres_per_pixel), "diameter-approximation"
    return (center[0] / ppm, center[1] / ppm), "manual"


def detect_ball(frame):
    frame = preprocess_camera(frame)
    model_frame, scale_x, scale_y = prepare_frame_for_detection(frame)
    if tracker_method == "ByteTrack":
        results = model.track(model_frame, conf=confidence_threshold, verbose=False, persist=True, tracker="bytetrack.yaml")[0]
    else:
        results = model(model_frame, conf=confidence_threshold, verbose=False)[0]

    if results.boxes is None or len(results.boxes) == 0:
        return None

    # Use the highest-confidence detection.
    best_index = int(results.boxes.conf.argmax().item())
    box = results.boxes.xyxy[best_index].cpu().numpy().astype(float)
    box[[0, 2]] *= scale_x
    box[[1, 3]] *= scale_y
    box = box.astype(int)
    confidence = float(results.boxes.conf[best_index].item())

    x1, y1, x2, y2 = box
    center = ((x1 + x2) // 2, (y1 + y2) // 2)

    return (x1, y1, x2, y2), center, confidence


def annotate_frame(
    frame, detection, trajectory, speed, landing_point, fps
):
    frame = frame.copy()

    if detection is not None:
        (x1, y1, x2, y2), center, confidence = detection

        cv2.rectangle(
            frame, (x1, y1), (x2, y2), (0, 255, 0), 2
        )
        cv2.circle(frame, center, 5, (0, 255, 0), -1)

        cv2.putText(
            frame,
            f"Ball {confidence:.2f}",
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

    draw_trajectory(trajectory, frame)

    if speed is not None:
        cv2.putText(
            frame,
            f"Speed: {speed:.2f} m/s | {speed * 3.6:.2f} km/h",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (50, 50, 255),
            2
        )

    if landing_point is not None:
        height, width = frame.shape[:2]
        lx, ly = landing_point

        if 0 <= lx < width and 0 <= ly < height:
            cv2.circle(frame, landing_point, 7, (255, 0, 255), -1)
            cv2.putText(
                frame,
                "Predicted position",
                (min(lx + 8, width - 180), max(ly - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 255),
                2
            )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 0),
        2
    )

    return frame


def show_graphs(speeds, trajectory, speed_slot, trajectory_slot):
    if speeds:
        speed_slot.line_chart(
            {"Speed (m/s)": speeds},
            height=220
        )

    if len(trajectory) >= 2:
        fig, ax = plt.subplots(figsize=(5, 3))
        xs, ys = zip(*trajectory)
        ax.plot(xs, ys, marker="o")
        ax.set_title("Ball Trajectory")
        ax.set_xlabel("X position (pixels)")
        ax.set_ylabel("Y position (pixels)")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        trajectory_slot.pyplot(fig)
        plt.close(fig)


# ---------------- IMAGE MODE ----------------
if mode == "Image":
    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        frame = np.array(image)

        with st.spinner("Detecting ball..."):
            results = model(frame, conf=confidence_threshold, verbose=False)[0]
            annotated = cv2.cvtColor(
                results.plot(), cv2.COLOR_BGR2RGB
            )

        st.image(
            annotated,
            caption="Ball Detection Results",
            use_container_width=True
        )
        st.write(f"Detections: {len(results.boxes)}")


# ---------------- VIDEO UPLOAD MODE ----------------
elif mode == "Upload Video":
    st.subheader("Upload and Analyse a Video")

    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=["mp4", "mov", "avi", "mkv"],
        key="video_upload"
    )

    if uploaded_video is not None:
        st.video(uploaded_video)

        if st.button("Run Video Analysis", type="primary"):
            suffix = Path(uploaded_video.name).suffix or ".mp4"
            temp_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix
                ) as temp_file:
                    temp_file.write(uploaded_video.getvalue())
                    temp_path = temp_file.name

                cap = cv2.VideoCapture(temp_path)

                if not cap.isOpened():
                    st.error(
                        "Could not open this video. Try an MP4 file."
                    )
                    cap.release()
                    st.stop()

                source_fps = cap.get(cv2.CAP_PROP_FPS)
                if not np.isfinite(source_fps) or source_fps <= 0:
                    source_fps = 30.0

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                progress = st.progress(0)
                status = st.empty()
                video_display = st.empty()

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Speed History")
                    speed_graph = st.empty()
                with col2:
                    st.subheader("Trajectory")
                    trajectory_graph = st.empty()

                previous_position = None
                kalman = CentreKalman()
                trajectory = deque(maxlen=30)
                observation_history = deque(maxlen=max(15, smoothing_window))
                speeds = []
                estimator_results = {name: [] for name in ("Finite difference", "Windowed linear regression", "Projectile-model fit")}
                frame_index = 0
                last_detection = None
                landing_point = None

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Use source-video timestamps, not processing time.
                    current_time = frame_index / source_fps
                    detection = detect_ball(frame)
                    speed = None

                    if detection is not None:
                        box, raw_center, confidence = detection
                        center = kalman.update(raw_center) if tracking_method == "Kalman filter" else raw_center
                        # Reject implausibly large jumps as likely tracking switches.
                        jump_ok = previous_position is None or math.dist(center, previous_position[0]) <= jump_threshold
                        if jump_ok:
                            detection = (detection[0], center, confidence)
                            trajectory.append(center)
                            physical_center, scale_kind = get_point_scale(center, box)
                            observation_history.append((physical_center[0], physical_center[1], current_time))
                            estimates = estimate_history_speeds(observation_history, 1.0, smoothing_window)
                            for name, value in estimates.items():
                                estimator_results[name].append(value)
                            if estimator_method == "Compare all":
                                speed = estimates.get("Windowed linear regression", estimates.get("Finite difference"))
                            else:
                                speed = estimates.get(estimator_method)
                            if speed is not None:
                                speeds.append(speed)
                            previous_position = (center, current_time)
                            landing_point = predict_landing(trajectory)
                            last_detection = detection
                        else:
                            speed = None

                    # Keep the last trajectory visible when the ball
                    # is briefly missed by the detector.
                    annotated = annotate_frame(
                        frame,
                        detection,
                        trajectory,
                        speed,
                        landing_point,
                        source_fps
                    )

                    video_display.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_container_width=True
                    )

                    frame_index += 1

                    if total_frames > 0:
                        progress.progress(
                            min(frame_index / total_frames, 1.0)
                        )

                    status.write(
                        f"Processed {frame_index} frames | "
                        f"Detections tracked: {len(trajectory)}"
                    )

                    # Refresh graphs periodically to keep processing
                    # overhead manageable.
                    if frame_index % 10 == 0:
                        show_graphs(
                            speeds, trajectory,
                            speed_graph, trajectory_graph
                        )

                cap.release()
                progress.progress(1.0)
                show_graphs(
                    speeds, trajectory,
                    speed_graph, trajectory_graph
                )

                st.success("Video analysis completed.")
                st.write(f"Detector weights: `{selected_model_path}` | Tracker mode: `{tracker_method}` | Resolution mode: `{processing_width}")
                st.write(f"Sensitivity settings: confidence={confidence_threshold:.2f}, smoothing={smoothing_window}, jump threshold={jump_threshold}px, lighting={experiment_lighting}, camera angle={experiment_camera_angle}.")
                st.caption("To compare detectors, rerun the same clip with each model selection and record the same metrics. Official YOLO11 weights are general-purpose pretrained models and may not detect your custom ball class as reliably as your fine-tuned weights.")
                if speeds:
                    average_speed = float(np.mean(speeds))
                    st.metric(
                        "Average estimated speed",
                        f"{average_speed:.2f} m/s",
                        f"{average_speed * 3.6:.2f} km/h"
                    )
                    st.metric(
                        "Maximum estimated speed",
                        f"{max(speeds):.2f} m/s"
                    )
                    st.caption(
                        "Estimates depend on calibration, camera geometry, and reliable detections. "
                        "The projectile fit is an image-plane approximation, not true 3D reconstruction."
                    )
                    # Always show estimator comparison after a successful video run,
                    # even if the user selected one estimator for the annotated overlay.
                    st.subheader("Estimator Comparison")
                    st.caption(
                        "All available estimators are evaluated on the accepted detection history. "
                        "Kalman filtering is a position smoother/tracker option, not a separate speed estimator."
                    )
                    comparison_rows = []
                    for name in ("Finite difference", "Windowed linear regression", "Projectile-model fit"):
                        values = estimator_results.get(name, [])
                        finite_values = [float(v) for v in values if v is not None and np.isfinite(v)]
                        if finite_values:
                            comparison_rows.append({
                                "Estimator": name,
                                "Mean speed (m/s)": round(float(np.mean(finite_values)), 3),
                                "Maximum speed (m/s)": round(float(np.max(finite_values)), 3),
                                "Measurements": len(finite_values),
                                "Std. dev. (m/s)": round(float(np.std(finite_values, ddof=1)), 3) if len(finite_values) > 1 else None,
                            })
                        else:
                            comparison_rows.append({
                                "Estimator": name,
                                "Mean speed (m/s)": "Not enough detections",
                                "Maximum speed (m/s)": "Not enough detections",
                                "Measurements": 0,
                                "Std. dev. (m/s)": None,
                            })
                    st.dataframe(comparison_rows, use_container_width=True, hide_index=True)
                    st.caption(
                        "Compare methods using the same video and calibration. The projectile fit is an image-plane approximation; "
                        "the standard deviation describes variation, not validated measurement accuracy."
                    )

                    selected_spread = calculate_uncertainty(speeds)
                    if selected_spread is not None:
                        st.metric("Speed estimate spread (standard deviation)", f"±{selected_spread:.2f} m/s")
                        st.caption("This is descriptive variation across estimates, not a validated measurement uncertainty or guaranteed error bound.")
                    # First-order illustrative uncertainty propagation; requires independent calibration of inputs for formal confidence bounds.
                    representative_speed = average_speed
                    relative_cal = calibration_error_pct / 100.0
                    relative_time = (timing_jitter_ms / 1000.0) * source_fps if source_fps > 0 else 0.0
                    relative_time = min(relative_time, 1.0)  # conservative dimensionless approximation; refine with measured timestamps
                    relative_loc = localization_noise_px / max(1.0, float(jump_threshold))
                    illustrative_uncertainty = representative_speed * math.sqrt(relative_cal**2 + relative_time**2 + relative_loc**2)
                    st.metric("Illustrative propagated uncertainty", f"±{illustrative_uncertainty:.2f} m/s")
                    st.caption("Approximation from the user-entered calibration, timing and localization terms; not a validated confidence interval. To test coverage, compare against a known true speed across repeated trials.")
                    st.caption(f"Experiment metadata: lighting={experiment_lighting}; camera angle={experiment_camera_angle}; inference resolution={processing_width}; confidence={confidence_threshold:.2f}; smoothing window={smoothing_window}; jump threshold={jump_threshold}px.")
                    if reference_speed_mps > 0:
                        absolute_error = abs(average_speed - reference_speed_mps)
                        percentage_error = absolute_error / reference_speed_mps * 100.0
                        st.metric("Absolute error vs reference", f"{absolute_error:.2f} m/s")
                        st.metric("Percentage error vs reference", f"{percentage_error:.1f}%")
                        coverage = abs(average_speed - reference_speed_mps) <= illustrative_uncertainty
                        st.write(f"Reference value falls within illustrative ± uncertainty: **{'Yes' if coverage else 'No'}** (single run only; repeat trials are required for coverage evaluation).")
                    if blur_length_px > 0 and exposure_seconds > 0:
                        if calibration_method == "Known ball diameter" and detection is not None:
                            bx1, by1, bx2, by2 = detection[0]
                            apparent_diameter_px = max(1.0, float(max(bx2 - bx1, by2 - by1)))
                            metres_per_pixel_local = ball_diameter_m / apparent_diameter_px
                            blur_speed = blur_length_px * metres_per_pixel_local / exposure_seconds
                        elif calibration_method == "Manual pixels per metre":
                            blur_speed = blur_length_px / ppm / exposure_seconds
                        else:
                            blur_speed = None
                        if blur_speed is not None:
                            st.metric("Motion-blur cue (m/s)", f"{blur_speed:.2f}")
                            fused_speed = (average_speed + blur_speed) / 2.0
                            st.metric("Exploratory fused speed (equal-weight average)", f"{fused_speed:.2f} m/s", f"{fused_speed * 3.6:.2f} km/h")
                            st.caption("Blur streak length is entered manually. Equal-weight fusion is an experimental baseline, not a statistically optimal fusion; verify both cues against reference speed before reporting accuracy.")
                        else:
                            st.info("Blur speed fusion currently requires manual scale or known-ball-diameter scale. Homography needs a local directional scale for a physically valid blur conversion.")
                else:
                    st.warning(
                        "No consecutive ball detections were available "
                        "to estimate speed. Try a clearer video."
                    )

            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)


# ---------------- WEBCAM MODE ----------------
elif mode == "Webcam":
    st.subheader("Live Ball Detection")
    st.caption(
        "Keep the camera stable. Use a suitable calibration value "
        "for meaningful speed estimates."
    )

    run = st.checkbox("Start Webcam")

    if run:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error(
                "Could not open the webcam. Check camera permissions "
                "and close other apps using the camera."
            )
            st.stop()

        previous_position = None
        kalman = CentreKalman()
        trajectory = deque(maxlen=30)
        observation_history = deque(maxlen=max(15, smoothing_window))
        speeds = []

        col1, col2 = st.columns([2, 1])

        with col1:
            webcam_display = st.empty()

        with col2:
            st.subheader("Speed History")
            speed_graph = st.empty()
            st.subheader("Trajectory")
            trajectory_graph = st.empty()

        stop_message = st.empty()
        frame_count = 0

        try:
            while run:
                start_time = time.perf_counter()
                ret, frame = cap.read()

                if not ret:
                    st.warning("Webcam disconnected.")
                    break

                current_time = time.perf_counter()
                detection = detect_ball(frame)
                speed = None
                landing_point = None

                if detection is not None:
                    box, raw_center, confidence = detection
                    center = kalman.update(raw_center) if tracking_method == "Kalman filter" else raw_center
                    jump_ok = previous_position is None or math.dist(center, previous_position[0]) <= jump_threshold
                    if jump_ok:
                        detection = (detection[0], center, confidence)
                        trajectory.append(center)
                        physical_center, scale_kind = get_point_scale(center, box)
                        observation_history.append((physical_center[0], physical_center[1], current_time))
                        estimates = estimate_history_speeds(observation_history, 1.0, smoothing_window)
                        if estimator_method == "Compare all":
                            speed = estimates.get("Windowed linear regression", estimates.get("Finite difference"))
                        else:
                            speed = estimates.get(estimator_method)
                        if speed is not None:
                            speeds.append(speed)
                        previous_position = (center, current_time)
                        landing_point = predict_landing(trajectory)

                elapsed = time.perf_counter() - start_time
                fps = 1.0 / elapsed if elapsed > 0 else 0.0

                annotated = annotate_frame(
                    frame, detection, trajectory,
                    speed, landing_point, fps
                )

                webcam_display.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    use_container_width=True
                )

                frame_count += 1

                if frame_count % 10 == 0:
                    show_graphs(
                        speeds, trajectory,
                        speed_graph, trajectory_graph
                    )

        finally:
            cap.release()

        stop_message.info(
            "Webcam stopped. Uncheck Start Webcam to finish."
        )
