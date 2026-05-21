import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import cv2
import numpy as np
import threading
import time

# ✅ EXPLICIT DIRECT IMPORTS: Fixes the 'module mediapipe has no attribute solutions' error
import mediapipe.python.solutions.pose as mp_pose
import mediapipe.python.solutions.drawing_utils as mp_draw

# Base Page Setup and Configurations
st.set_page_config(page_title="AI Omni Fitness Trainer", layout="wide")
st.title("🏋️‍♂️ AI Universal Fitness Trainer & Form Corrector")
st.markdown("Real-time form tracking, analytics tracking, and automated client voice guidance across multiple exercises.")

# -------------------------------------------------------------------------
# THREAD-SAFE MEMORY LAYER
# -------------------------------------------------------------------------
# Data buffer shared between the background processing thread and primary web layout thread
state_lock = threading.Lock()
workout_metrics = {
    "reps": 0,
    "phase": "Starting Position",
    "feedback": "Establishing baseline positioning...",
    "vocal_alert": "",
    "alert_time": 0.0
}

def clear_workout_metrics():
    """ Resets execution history state securely when changing exercise modules """
    global workout_metrics
    with state_lock:
        workout_metrics["reps"] = 0
        workout_metrics["phase"] = "Starting Position"
        workout_metrics["feedback"] = "Perfect alignment tracked."
        workout_metrics["vocal_alert"] = ""
        workout_metrics["alert_time"] = 0.0

# -------------------------------------------------------------------------
# SPATIAL BIOMECHANICAL MATH ENGINE
# -------------------------------------------------------------------------
def calculate_spatial_angle(pt_a, pt_b, pt_c):
    """
    Computes structural joint angle orientations over isolated 3D keypoints 
    utilizing vectorized dot product formulations.
    """
    v_a = np.array(pt_a)  # Top coordinate position
    v_b = np.array(pt_b)  # Vertex core tracking center (e.g. Knee, Elbow)
    v_c = np.array(pt_c)  # Bottom terminal joint coordinate

    ba = v_a - v_b
    bc = v_c - v_b

    cos_theta = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cos_theta = np.clip(cos_theta, -1.0, 1.0) # Floating point bounding protection
    
    return np.degrees(np.arccos(cos_theta))

# -------------------------------------------------------------------------
# CORE LIVE ESTIMATION CALLBACK PIPELINE
# -------------------------------------------------------------------------
# Initializing using the fixed, direct module paths
pose_model = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6, model_complexity=1)

def transform_video_feed(frame: av.VideoFrame) -> av.VideoFrame:
    global workout_metrics
    
    # Extract dynamic raw pixel data arrays directly from the incoming streaming frame
    frame_matrix = frame.to_ndarray(format="bgr24")
    frame_matrix = cv2.flip(frame_matrix, 1) # Flip mirror-style for intuitive human tracking orientation
    f_h, f_w, _ = frame_matrix.shape

    # Process pose coordinates using RGB arrays
    rgb_frame = cv2.cvtColor(frame_matrix, cv2.COLOR_BGR2RGB)
    prediction = pose_model.process(rgb_frame)

    if prediction.pose_landmarks:
        nodes = prediction.pose_landmarks.landmark
        
        # Read the current user configuration safely from Streamlit session state memory
        chosen_exercise = st.session_state.get("selected_routine", "Squats (No Dumbbells)")
        
        # Dynamic State Processing Logic based on chosen routine configuration profile
        with state_lock:
            if chosen_exercise == "Bicep Curls (With Dumbbells)":
                # --- CALIBRATION SPECIFICS: RIGHT BICEP CURL TRACKING ---
                shoulder = [nodes[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, nodes[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y, nodes[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
                elbow    = [nodes[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, nodes[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y, nodes[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
                wrist    = [nodes[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, nodes[mp_pose.PoseLandmark.RIGHT_WRIST.value].y, nodes[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]
                hip      = [nodes[mp_pose.PoseLandmark.RIGHT_HIP.value].x, nodes[mp_pose.PoseLandmark.RIGHT_HIP.value].y, nodes[mp_pose.PoseLandmark.RIGHT_HIP.value].z]

                curl_angle = calculate_spatial_angle(shoulder, elbow, wrist)
                sway_angle = calculate_spatial_angle(hip, shoulder, elbow)

                # Form Evaluator Rules
                if sway_angle > 30:
                    workout_metrics["feedback"] = "⚠️ Keep your elbow locked close to your side torso plane!"
                    if time.time() - workout_metrics["alert_time"] > 4.0:
                        workout_metrics["vocal_alert"] = "Keep your elbow close to your side"
                        workout_metrics["alert_time"] = time.time()
                else:
                    workout_metrics["feedback"] = "✅ Excellent structural posture tracked."
                    workout_metrics["vocal_alert"] = ""

                # Repetition Counting State Mechanism
                if curl_angle < 40:
                    workout_metrics["phase"] = "Concentric (Up)"
                elif curl_angle > 150 and workout_metrics["phase"] == "Concentric (Up)":
                    workout_metrics["phase"] = "Eccentric (Down)"
                    workout_metrics["reps"] += 1

                # Visual overlay anchoring data parameters on top of the moving elbow joint node
                text_anchor = tuple(np.multiply(elbow[:2], [f_w, f_h]).astype(int))
                cv2.putText(frame_matrix, f"{int(curl_angle)} deg", text_anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            elif chosen_exercise == "Squats (No Dumbbells)":
                # --- CALIBRATION SPECIFICS: RIGHT SIDE SQUAT TRACKING ---
                hip   = [nodes[mp_pose.PoseLandmark.RIGHT_HIP.value].x, nodes[mp_pose.PoseLandmark.RIGHT_HIP.value].y, nodes[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
                knee  = [nodes[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, nodes[mp_pose.PoseLandmark.RIGHT_KNEE.value].y, nodes[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
                ankle = [nodes[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, nodes[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y, nodes[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]
                shldr = [nodes[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, nodes[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y, nodes[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]

                squat_angle = calculate_spatial_angle(hip, knee, ankle)
                back_angle  = calculate_spatial_angle(shldr, hip, knee)

                # Form Evaluator Rules
                if back_angle < 65:
                    workout_metrics["feedback"] = "⚠️ Warning: Back is rounding too forward! Keep your chest upright."
                    if time.time() - workout_metrics["alert_time"] > 4.0:
                        workout_metrics["vocal_alert"] = "Keep your chest upright"
                        workout_metrics["alert_time"] = time.time()
                else:
                    workout_metrics["feedback"] = "✅ Correct trunk extension confirmed."
                    workout_metrics["vocal_alert"] = ""

                # Repetition Counting State Mechanism
                if squat_angle < 100:
                    workout_metrics["phase"] = "Bottom Position"
                elif squat_angle > 160 and workout_metrics["phase"] == "Bottom Position":
                    workout_metrics["phase"] = "Top Position"
                    workout_metrics["reps"] += 1

                # Visual overlay anchoring data parameters on top of the moving knee joint node
                text_anchor = tuple(np.multiply(knee[:2], [f_w, f_h]).astype(int))
                cv2.putText(frame_matrix, f"{int(squat_angle)} deg", text_anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw full skeletal line segments directly over the live stream output canvas
        mp_draw.draw_landmarks(
            frame_matrix, prediction.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=(60, 240, 90), thickness=2, circle_radius=2),
            mp_draw.DrawingSpec(color=(220, 60, 120), thickness=2, circle_radius=2)
        )

    return av.VideoFrame.from_ndarray(frame_matrix, format="bgr24")

# -------------------------------------------------------------------------
# INTERACTIVE APPLICATION SIDEBAR SELECTION
# -------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Workout Environment Options")
    # Dynamic selection state configuration
    selected_routine = st.selectbox(
        "Choose Target Routine Profile:",
        ["Squats (No Dumbbells)", "Bicep Curls (With Dumbbells)"],
        key="selected_routine",
        on_change=clear_workout_metrics
    )
    st.markdown("---")
    st.info("💡 **Voice System Integration:** When form deviations occur, instructions are automatically processed directly by your speaker hardware using web-native audio modules.")

# -------------------------------------------------------------------------
# INTERACTIVE USER INTERFACE DESIGN
# -------------------------------------------------------------------------
column_left, column_right = st.columns([5, 3])

with column_left:
    st.subheader(f"📹 Dynamic Diagnostic Camera: {selected_routine}")
    
    # ✅ FIXED & MOBILE OPTIMIZED WEBRTC HANDLER
    webrtc_context = webrtc_streamer(
        key="omni-trainer-stream",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=transform_video_feed,
        
        # Explicit media constraints to target mobile selfie cameras efficiently
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "facingMode": "user"  # Opens front camera natively on mobile handsets
            },
            "audio": False
        },
        
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        async_processing=True,
    )

with column_right:
    st.subheader("📊 Live Biomechanical Telemetry")
    
    # Establish static empty slots inside target visual frame structures
    rep_ui_card = st.empty()
    phase_ui_card = st.empty()
    feedback_ui_card = st.empty()
    javascript_audio_hook = st.empty()

    # Continuously poll data channels while video active processes run
    while webrtc_context.state.playing:
        with state_lock:
            ui_reps = workout_metrics["reps"]
            ui_phase = workout_metrics["phase"]
            ui_feedback = workout_metrics["feedback"]
            ui_voice_cmd = workout_metrics["vocal_alert"]

        # Render information into active metrics panels
        rep_ui_card.metric(label="Validated Reps Tracker", value=ui_reps)
        phase_ui_card.info(f"Target Exercise Stage Position: **{ui_phase}**")

        if "⚠️" in ui_feedback:
            feedback_ui_card.error(ui_feedback)
            
            # Injecting direct, client-executed JavaScript audio synthesis elements
            # This allows full audio playbacks directly over live mobile platforms.
            audio_script_payload = f"""
                <script>
                    var voice_synthesis_engine = new SpeechSynthesisUtterance("{ui_voice_cmd}");
                    voice_synthesis_engine.rate = 1.05;
                    window.speechSynthesis.speak(voice_synthesis_engine);
                </script>
            """
            with javascript_audio_hook:
                st.components.v1.html(audio_script_payload, height=0, width=0)
        else:
            feedback_ui_card.success(ui_feedback)
            javascript_audio_hook.empty()

        # Brief execution pause interval sleep loop to mitigate processor overheads
        time.sleep(0.1)