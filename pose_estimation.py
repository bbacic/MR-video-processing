video_file = "./video.mp4"
import cv2
import mediapipe as mp
import math

# Custom pose connections
POSE_CONNECTIONS_BODY = [
    (mp.solutions.pose.PoseLandmark.NOSE, mp.solutions.pose.PoseLandmark.LEFT_EYE_INNER),
    (mp.solutions.pose.PoseLandmark.NOSE, mp.solutions.pose.PoseLandmark.LEFT_EYE),
    (mp.solutions.pose.PoseLandmark.NOSE, mp.solutions.pose.PoseLandmark.LEFT_EYE_OUTER),
    (mp.solutions.pose.PoseLandmark.NOSE, mp.solutions.pose.PoseLandmark.RIGHT_EYE_INNER),
    (mp.solutions.pose.PoseLandmark.NOSE, mp.solutions.pose.PoseLandmark.RIGHT_EYE),
    (mp.solutions.pose.PoseLandmark.NOSE, mp.solutions.pose.PoseLandmark.RIGHT_EYE_OUTER),
    (mp.solutions.pose.PoseLandmark.NOSE, mp.solutions.pose.PoseLandmark.LEFT_EAR),
    (mp.solutions.pose.PoseLandmark.NOSE, mp.solutions.pose.PoseLandmark.RIGHT_EAR),
    (mp.solutions.pose.PoseLandmark.MOUTH_LEFT, mp.solutions.pose.PoseLandmark.MOUTH_RIGHT),
    (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER),
    (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_ELBOW),
    (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_WRIST),
    (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_ELBOW),
    (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_WRIST),
    (mp.solutions.pose.PoseLandmark.LEFT_ELBOW, mp.solutions.pose.PoseLandmark.LEFT_WRIST),
    (mp.solutions.pose.PoseLandmark.RIGHT_ELBOW, mp.solutions.pose.PoseLandmark.RIGHT_WRIST),
    (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_HIP),
    (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_HIP),
    (mp.solutions.pose.PoseLandmark.LEFT_HIP, mp.solutions.pose.PoseLandmark.RIGHT_HIP),
    (mp.solutions.pose.PoseLandmark.LEFT_HIP, mp.solutions.pose.PoseLandmark.LEFT_KNEE),
    (mp.solutions.pose.PoseLandmark.RIGHT_HIP, mp.solutions.pose.PoseLandmark.RIGHT_KNEE),
    (mp.solutions.pose.PoseLandmark.LEFT_KNEE, mp.solutions.pose.PoseLandmark.LEFT_ANKLE),
    (mp.solutions.pose.PoseLandmark.RIGHT_KNEE, mp.solutions.pose.PoseLandmark.RIGHT_ANKLE),
]
def calculate_angle(a, b, c):
    """Calculate the angle between three points"""
    angle_rad = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

def main():
    # Video capture from file
    cap_file = cv2.VideoCapture(video_file)

    # Video capture from camera
    cap_camera = cv2.VideoCapture(0)

    # Set up MediaPipe Pose for file and camera separately
    mp_pose_file = mp.solutions.pose.Pose()
    mp_pose_camera = mp.solutions.pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    while True:
        # Read frame from video file
        ret_file, frame_file = cap_file.read()

        # Read frame from camera
        ret_camera, frame_camera = cap_camera.read()

        if not ret_file or not ret_camera:
            break

        # Convert the image color space from BGR to RGB
        frame_file_rgb = cv2.cvtColor(frame_file, cv2.COLOR_BGR2RGB)
        frame_camera_rgb = cv2.cvtColor(frame_camera, cv2.COLOR_BGR2RGB)

        # Process the frame from video file with MediaPipe Pose
        results_file = mp_pose_file.process(frame_file_rgb)

        # Process the frame from camera with MediaPipe Pose
        results_camera = mp_pose_camera.process(frame_camera_rgb)

        # Convert the image color space back to BGR
        frame_file = cv2.cvtColor(frame_file_rgb, cv2.COLOR_RGB2BGR)
        frame_camera = cv2.cvtColor(frame_camera_rgb, cv2.COLOR_RGB2BGR)

        # Draw pose landmarks on the frame from video file
        if results_file.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame_file,
                results_file.pose_landmarks,
                connections=POSE_CONNECTIONS_BODY,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
            )
            # Calculate and display the angle of the knee joint
            left_hip = results_file.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP]
            left_knee = results_file.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_KNEE]
            left_ankle = results_file.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ANKLE]
            angle_file = calculate_angle(left_hip, left_knee, left_ankle)
            text = f"KNEE ANGLE: {angle_file:.2f}"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x = int((frame_file.shape[1] - text_size[0]) / 2)
            cv2.putText(frame_file, text, (text_x, frame_file.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw pose landmarks on the frame from camera
        if results_camera.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame_camera,
                results_camera.pose_landmarks,
                connections=POSE_CONNECTIONS_BODY,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
            )
            # Calculate and display the angle of the knee joint
            left_hip = results_camera.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP]
            left_knee = results_camera.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_KNEE]
            left_ankle = results_camera.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ANKLE]
            angle_camera = calculate_angle(left_hip, left_knee, left_ankle)
            text = f"KNEE ANGLE: {angle_camera:.2f}"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x = int((frame_camera.shape[1] - text_size[0]) / 2)
            cv2.putText(frame_camera, text, (text_x, frame_camera.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Resize frames to have the same shape
        frame_file_resized = cv2.resize(frame_file, (640, 480))
        frame_camera_resized = cv2.resize(frame_camera, (640, 480))

        # Concatenate frames horizontally
        output = cv2.hconcat([frame_file_resized, frame_camera_resized])

        # Calculate the difference in angles between file and camera
        if results_file.pose_landmarks and results_camera.pose_landmarks:
            angle_diff = angle_file - angle_camera

            # Display the angle difference as percentage on the top of the frame
            text = f"Angle Difference: {abs(angle_diff):.2f}%"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x = int((output.shape[1] - text_size[0]) / 2)
            cv2.putText(output, text, (text_x, text_size[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Check if angle difference is higher than 30%
            if abs(angle_diff) > 30:
                # Get the coordinates of the left knee
                left_knee_camera_x = int(left_knee.x * frame_camera.shape[1])
                left_knee_camera_y = int(left_knee.y * frame_camera.shape[0])
                
                # Draw a red circle on top of the left knee
                cv2.circle(frame_camera, (left_knee_camera_x, left_knee_camera_y), 5, (0, 0, 255), -1)


        # Show the frame
        cv2.imshow("Video and Pose Estimation", output)

        # Exit if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video captures and close the window
    cap_file.release()
    cap_camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
