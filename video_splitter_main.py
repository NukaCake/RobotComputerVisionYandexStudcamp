import os
import subprocess
import torch
import clip
from PIL import Image
import numpy as np

#scripts used to split video to images for datasets
def extract_frames(video_path, frames_dir, fps=5):
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    else:
        # Clear the frames directory
        for file in os.listdir(frames_dir):
            file_path = os.path.join(frames_dir, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
    # Build ffmpeg command
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f'fps={fps}',
        os.path.join(frames_dir, 'frame%05d.png')
    ]
    print("Running ffmpeg command:")
    print(' '.join(ffmpeg_cmd))
    result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("Error extracting frames:")
        print(result.stderr)
    else:
        print("Frames extracted successfully.")



def compute_embeddings(frames_dir, device='cpu'):
    # Load CLIP model
    model, preprocess = clip.load('ViT-B/32', device=device)
    embeddings = {}
    frame_files = sorted(os.listdir(frames_dir))
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        image = preprocess(Image.open(frame_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image)
            embedding /= embedding.norm(dim=-1, keepdim=True)  # Normalize embedding
            embeddings[frame_file] = embedding.cpu().numpy()
    return embeddings

#meausuring embeddings to effectilvely delete similar frames
def delete_similar_frames(embeddings, frames_dir, similarity_threshold=0.90):
    last_kept_embedding = None
    kept_frames = []
    frame_files = sorted(embeddings.keys())
    for frame_file in frame_files:
        embedding = embeddings[frame_file]
        if last_kept_embedding is None:
            last_kept_embedding = embedding
            kept_frames.append(frame_file)
        else:
            similarity = np.dot(embedding, last_kept_embedding.T).item()
            if similarity < similarity_threshold:
                last_kept_embedding = embedding
                kept_frames.append(frame_file)
            else:
                # Delete the frame
                frame_path = os.path.join(frames_dir, frame_file)
                os.remove(frame_path)
    return kept_frames


def main():
    video_path = 'output2024-10-18 14:44:17.832554.avi'  # Replace with your video file path
    frames_dir = 'yolov11/frames'
    fps = 5
    similarity_threshold = 0.95
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Extracting frames...")
    extract_frames(video_path, frames_dir, fps)
    print("Computing embeddings...")
    embeddings = compute_embeddings(frames_dir, device)
    print("Deleting similar frames...")
    kept_frames = delete_similar_frames(embeddings, frames_dir, similarity_threshold)
    print(f"Kept {len(kept_frames)} frames out of {len(embeddings)}")


if __name__ == '__main__':
    main()