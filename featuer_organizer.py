import os
import shutil

# Set the path for the Video folder and the Features folder
video_folder = "/home/temporaryuser2/Desktop/sdp_module/lstm/drive/Video/Cam2"  # Replace with your Video folder path
features_folder = "/home/temporaryuser2/Desktop/sdp_module/lstm/features_slr"  # Replace with your Feature files folder path

def organize_feature_files():
    # Loop through all subfolders in the Video directory
    for subfolder in os.listdir(video_folder):
        subfolder_path = os.path.join(video_folder, subfolder)
        
        # Check if it's a directory
        if os.path.isdir(subfolder_path):
            # Create corresponding subfolder in the features folder
            feature_subfolder_path = os.path.join(features_folder, subfolder)
            if not os.path.exists(feature_subfolder_path):
                os.makedirs(feature_subfolder_path)
            
            # Loop through video files in the current subfolder
            for video_file in os.listdir(subfolder_path):
                if video_file.endswith(".mp4"):
                    # Extract the base name of the video file and create the corresponding feature file name
                    base_name = os.path.splitext(video_file)[0]
                    feature_file_name = base_name.replace(" ", "_") + ".pt"
                    feature_file_path = os.path.join(features_folder, feature_file_name)
                    
                    # Check if the corresponding feature file exists
                    if os.path.exists(feature_file_path):
                        # Move the .pt file to the corresponding subfolder
                        shutil.move(feature_file_path, os.path.join(feature_subfolder_path, feature_file_name))
                        print(f"Moved: {feature_file_name} to {feature_subfolder_path}")
                    else:
                        print(f"Feature file not found for {video_file}")

# Run the function
organize_feature_files()