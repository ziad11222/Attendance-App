import face_recognition
import joblib
import os

def train_face_detection_model(input_folder, output_model_path, output_encoder_path):

    face_encodings = []
    labels = []

    for video_folder in os.listdir(input_folder):
        video_folder_path = os.path.join(input_folder, video_folder)

        if os.path.isdir(video_folder_path):
            print(f"Processing video folder: {video_folder}")

            for filename in os.listdir(video_folder_path):
                frame_path = os.path.join(video_folder_path, filename)

                image = face_recognition.load_image_file(frame_path)
                face_encodings_in_frame = face_recognition.face_encodings(image)

                # Check file
                if len(face_encodings_in_frame) == 1:
                    face_encoding = face_encodings_in_frame[0]
                    face_encodings.append(face_encoding)
                    labels.append(video_folder)


    print("Training face recognition model...")

    print("Saving face recognition model and encoder to files...")
    joblib.dump(face_encodings, output_model_path)
    joblib.dump(labels, output_encoder_path)

    print("Model creation completed.")

if __name__ == "__main__":
    input_folder_faces = r"D:\\Computer Vision\\frames_faces"  
    output_model_path = r"D:\\Computer Vision\\face_detection_model.joblib"  
    output_encoder_path = r"D:\\Computer Vision\\face_encoder_labels.joblib" 

    train_face_detection_model(input_folder_faces, output_model_path, output_encoder_path)
