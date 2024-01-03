import cv2
import dlib
import face_recognition
import os

def divide_video_frames(input_video_path, output_folder):

    video_name = os.path.splitext(os.path.basename(input_video_path))[0]

    video_output_folder = os.path.join(output_folder, video_name)
    if not os.path.exists(video_output_folder):
        os.makedirs(video_output_folder)

    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_per_second = 2
    frames_to_extract = fps * frames_per_second

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % (fps // frames_per_second) != 0:
            frame_count += 1
            continue
   
        output_path = os.path.join(video_output_folder, f"frame_{frame_count // (fps // frames_per_second)}.jpg")
        cv2.imwrite(output_path, frame)

        frame_count += 1

    # Release the video capture object
    cap.release()

def face_recognition_on_frames(input_folder, output_folder):

    for video_folder in os.listdir(input_folder):
        video_folder_path = os.path.join(input_folder, video_folder)

        if os.path.isdir(video_folder_path):
        
            face_output_folder = os.path.join(output_folder, video_folder)
            if not os.path.exists(face_output_folder):
                os.makedirs(face_output_folder)

            for filename in os.listdir(video_folder_path):
                frame_path = os.path.join(video_folder_path, filename)

                image = face_recognition.load_image_file(frame_path)

                face_locations = face_recognition.face_locations(image)
                for face_location in face_locations:
                    top, right, bottom, left = face_location
                    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
                output_path = os.path.join(face_output_folder, filename)
                cv2.imwrite(output_path, image)

if __name__ == "__main__":
    input_folder = r"D:\\Computer Vision\\frames"  
    output_folder_faces = r"D:\\Computer Vision\\frames_faces"  

    face_recognition_on_frames(input_folder, output_folder_faces)
