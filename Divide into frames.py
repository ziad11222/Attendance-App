import cv2
import dlib
import os

def divide_video_frames(input_video_path, output_folder):
    video_name = os.path.splitext(os.path.basename(input_video_path))[0]

    # Create subfolder
    video_output_folder = os.path.join(output_folder, video_name)
    if not os.path.exists(video_output_folder):
        os.makedirs(video_output_folder)

    cap = cv2.VideoCapture(input_video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_per_second = 2
    frames_to_extract = fps * frames_per_second
    detector = dlib.get_frontal_face_detector()

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % (fps // frames_per_second) != 0:
            frame_count += 1
            continue
        faces = detector(frame)

        output_path = os.path.join(video_output_folder, f"frame_{frame_count // (fps // frames_per_second)}.jpg")
        cv2.imwrite(output_path, frame)

        frame_count += 1

    cap.release()

def process_videos(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4") or filename.endswith(".mov") or filename.endswith(".MOV"):
            input_video_path = os.path.join(input_folder, filename)
            divide_video_frames(input_video_path, output_folder)

if __name__ == "__main__":
    
    input_folder = r"D:\\Computer Vision\\Dataset"  
    output_folder = r"D:\\Computer Vision\\frames"  

    process_videos(input_folder, output_folder)
