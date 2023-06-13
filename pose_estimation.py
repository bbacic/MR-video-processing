video_file = "video.mp4"
import cv2
import mediapipe as mp
import math

# Custom pose connections
POSE_CONNECTIONS_BODY = [
    (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER),
    (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_ELBOW),
    (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_ELBOW),
    (mp.solutions.pose.PoseLandmark.LEFT_ELBOW, mp.solutions.pose.PoseLandmark.LEFT_WRIST),
    (mp.solutions.pose.PoseLandmark.RIGHT_ELBOW, mp.solutions.pose.PoseLandmark.RIGHT_WRIST),
    (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_HIP),
    (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_HIP),
    (mp.solutions.pose.PoseLandmark.LEFT_HIP, mp.solutions.pose.PoseLandmark.RIGHT_HIP),
    (mp.solutions.pose.PoseLandmark.RIGHT_HIP, mp.solutions.pose.PoseLandmark.RIGHT_KNEE),
    (mp.solutions.pose.PoseLandmark.RIGHT_KNEE, mp.solutions.pose.PoseLandmark.RIGHT_ANKLE),
]
POSE_CONNECTIONS_KNEE = [
    (mp.solutions.pose.PoseLandmark.LEFT_HIP, mp.solutions.pose.PoseLandmark.LEFT_KNEE),
    (mp.solutions.pose.PoseLandmark.LEFT_KNEE, mp.solutions.pose.PoseLandmark.LEFT_ANKLE),
]

def calculate_angle(a, b, c):
    """Calculate the angle between three points"""
    angle_rad = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360
    if angle_deg > 180:
        angle_deg = 360 - angle_deg
    return angle_deg

def delta(a, b, c):
    # Determine the position of a point regarding the line determined by another two points
    return a.x*b.y+b.x*c.y+c.x*a.y-a.x*c.y-b.x*a.y-c.x*b.y

def distance2(a, b):
    # Calculate the euclidean distance between two points
    return (a.x-b.x)**2+(a.y-b.y)**2

def distance(a, b):
    # Calculate the euclidean distance between two points
    return math.sqrt(distance2(a, b))

def angle(a, b, c):
    # Calculate the measure of the angle B 
    # that is between two lines (AB and BC) 
    # determined by three points (A, B, C)
    # using the cosine theorem
    if distance2(a, b)*distance2(b, c) == 0:
        return 0
    the_angle = math.degrees(math.acos((distance2(a, b)+distance2(b, c)-distance2(a, c))/(2*distance(a, b)*distance(b, c))))
    if delta(a, b, c) > 0:
        return 360-the_angle
    return the_angle

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
        
        angle_diff = None
        # Draw pose landmarks on the frame from video file
        if results_file.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame_file,
                results_file.pose_landmarks,
                connections=POSE_CONNECTIONS_BODY,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=0, circle_radius=0),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=5),
            )
            mp_drawing.draw_landmarks(
                frame_file,
                results_file.pose_landmarks,
                connections=POSE_CONNECTIONS_KNEE,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=0, circle_radius=0),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=5),
            )
            
            # Calculate and display the angle of the knee joint
            left_hip = results_file.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP]
            left_knee = results_file.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_KNEE]
            left_ankle = results_file.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ANKLE]
            # old formula
            #angle_file = calculate_angle(left_hip, left_knee, left_ankle)
            # new formula
            angle_file = angle(left_hip, left_knee, left_ankle)
            text = f"KNEE ANGLE: {angle_file:.2f}"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            text_x = int((frame_file.shape[1] - text_size[0]) / 2)
            cv2.putText(frame_file, text, (text_x, frame_file.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # Draw a white circle for the left knee
            left_knee_x = int(left_knee.x * frame_file.shape[1])
            left_knee_y = int(left_knee.y * frame_file.shape[0])
            cv2.circle(frame_file, (left_knee_x, left_knee_y), 10, (255, 255, 255), -1)
        # Draw pose landmarks on the frame from camera
        if results_camera.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame_camera,
                results_camera.pose_landmarks,
                connections=POSE_CONNECTIONS_BODY,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=0, circle_radius=0),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=5),
            )
            # Calculate and display the angle of the knee joint
            left_hip = results_camera.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP]
            left_knee = results_camera.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_KNEE]
            left_ankle = results_camera.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ANKLE]            
            # old formula
            # angle_camera = calculate_angle(left_hip, left_knee, left_ankle)
            # new formula
            angle_camera = angle(left_hip, left_knee, left_ankle)
            text = f"KNEE ANGLE: {angle_camera:.2f}"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            text_x = int((frame_camera.shape[1] - text_size[0]) / 2)
            cv2.putText(frame_camera, text, (text_x, frame_camera.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            if angle_file and angle_camera:
                angle_diff = ((angle_camera / angle_file) * 100) - 100
                if abs(angle_diff) > 30:
                    text_color = (0, 0, 255)
                else:
                    text_color = (255, 255, 255)
                # Draw a line for the left foot
                mp_drawing.draw_landmarks(
                    frame_camera,
                    results_camera.pose_landmarks,
                    connections=POSE_CONNECTIONS_KNEE,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=0, circle_radius=0),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=text_color, thickness=5),
                )
                # Draw a circle for the left knee
                left_knee_x = int(left_knee.x * frame_camera.shape[1])
                left_knee_y = int(left_knee.y * frame_camera.shape[0])
                cv2.circle(frame_camera, (left_knee_x, left_knee_y), 10, text_color, -1)
        # Resize frames to have the same shape
        frame_file_resized = cv2.resize(frame_file, (640, 480))
        frame_camera_resized = cv2.resize(frame_camera, (640, 480))

        # Concatenate frames horizontally
        output = cv2.hconcat([frame_file_resized, frame_camera_resized])

        # Calculate the difference in angles between file and camera
        if angle_diff:
            # Display the angle difference as percentage on the top of the frame
            text = f"Angle Difference: {abs(angle_diff):.2f}%"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            text_x = int((output.shape[1] - text_size[0]) / 2)

            cv2.putText(output, text, (text_x, text_size[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)

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
