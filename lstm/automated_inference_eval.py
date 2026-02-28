import os
import pandas as pd
import subprocess

def get_expected_label(video_path, csv_path):
    # Extract parent directory name as idd
    idd = os.path.basename(os.path.dirname(video_path))
    df = pd.read_csv(csv_path, sep=';', header=None, names=['id', 'sentence', 'sign_language'])
    row = df[df['id'].astype(str) == str(idd)]
    if not row.empty:
        return row.iloc[0]['sign_language']
    return None

def run_inference(video_path):
    result = subprocess.run([
        'python', 'inference.py', video_path
    ], capture_output=True, text=True)
    output = result.stdout
    # Extract predicted sentence from output
    lines = output.splitlines()
    pred = None
    for i, line in enumerate(lines):
        if '📝 PREDICTED SENTENCE:' in line:
            if i+2 < len(lines):
                pred = lines[i+2].strip()
                break
    return pred, output

def main():
    csv_path = 'drive/sentences_all.csv'
    video_dir = 'drive/Video/Cam2/2/'
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    results = []
    for video in video_files:
        video_path = os.path.join(video_dir, video)
        expected = get_expected_label(video_path, csv_path)  # Pass full path, not just filename
        predicted, raw_output = run_inference(video_path)
        # Case-insensitive comparison
        match = (expected is not None and predicted is not None and 
                 expected.lower().strip() == predicted.lower().strip())
        results.append({
            'video': video,
            'expected': expected,
            'predicted': predicted,
            'match': match,
            'raw_output': raw_output
        })
        print(f"Video: {video}\nExpected: {expected}\nPredicted: {predicted}\nMatch: {match}\n{'-'*40}")
    # Optionally, save results to a CSV
    pd.DataFrame(results).to_csv('inference_eval_results.csv', index=False)

if __name__ == '__main__':
    main()
