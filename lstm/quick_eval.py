"""
Quick Evaluation on Subset
Tests model on a small subset (e.g., 100 videos) for fast feedback
"""
import os
import pandas as pd
import subprocess
import random
from collections import defaultdict

def get_expected_label(video_path, csv_path):
    idd = os.path.basename(os.path.dirname(video_path))
    df = pd.read_csv(csv_path, sep=';', header=None, names=['id', 'sentence', 'sign_language'])
    row = df[df['id'].astype(str) == str(idd)]
    if not row.empty:
        return row.iloc[0]['sign_language']
    return None

def run_inference(video_path):
    result = subprocess.run([
        'python3', 'inference.py', video_path
    ], capture_output=True, text=True, timeout=60)
    output = result.stdout
    lines = output.splitlines()
    pred = None
    for i, line in enumerate(lines):
        if '📝 PREDICTED SENTENCE:' in line:
            if i+2 < len(lines):
                pred = lines[i+2].strip()
                break
    return pred

def main():
    csv_path = 'drive/sentences_all.csv'
    video_base_dir = 'drive/Video/Cam2/'
    
    # Configurable: number of videos to test
    NUM_SAMPLES = 100  # Change this to test more/fewer videos
    NUM_FOLDERS = 50   # Use only the first N folders
    
    print("=" * 70)
    print(f"QUICK EVALUATION ({NUM_SAMPLES} videos)")
    print("=" * 70)
    
    # Find candidate folders (first N, sorted)
    subdirs = [
        entry.path for entry in os.scandir(video_base_dir)
        if entry.is_dir()
    ]
    subdirs = sorted(subdirs, key=lambda p: os.path.basename(p))
    selected_folders = subdirs[:NUM_FOLDERS]

    print(f"📁 Using first {len(selected_folders)} folders out of {len(subdirs)} total")

    # Find all video files from selected folders
    all_videos = []
    for folder in selected_folders:
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith('.mp4'):
                    video_path = os.path.join(root, file)
                    # Only include videos that have labels
                    if get_expected_label(video_path, csv_path):
                        all_videos.append(video_path)
    
    print(f"📊 Total videos with labels: {len(all_videos)}")
    
    # Sample random videos
    sample_videos = random.sample(all_videos, min(NUM_SAMPLES, len(all_videos)))
    print(f"🎲 Testing on {len(sample_videos)} random videos")
    print()
    
    results = []
    for idx, video_path in enumerate(sample_videos, 1):
        video_name = os.path.basename(video_path)
        
        print(f"[{idx}/{len(sample_videos)}] {video_name[:35]}...")
        
        try:
            expected = get_expected_label(video_path, csv_path)
            predicted = run_inference(video_path)
            
            match = (expected is not None and predicted is not None and 
                     expected.lower().strip() == predicted.lower().strip())
            
            results.append({
                'video': video_name,
                'expected': expected,
                'predicted': predicted,
                'match': match
            })
            
            status = "✅" if match else "❌"
            print(f"   Expected:  {expected}")
            print(f"   Predicted: {predicted}")
            print(f"   {status} {'MATCH' if match else 'MISMATCH'}")
            print()
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:50]}")
            print()
    
    # Calculate metrics
    total = len(results)
    correct = sum(1 for r in results if r['match'])
    accuracy = (correct / total * 100) if total > 0 else 0
    
    # Per-class stats
    class_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    for r in results:
        expected = r['expected']
        if expected:
            class_stats[expected]['total'] += 1
            if r['match']:
                class_stats[expected]['correct'] += 1
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"📊 Total: {total} videos")
    print(f"✅ Correct: {correct}")
    print(f"❌ Wrong: {total - correct}")
    print(f"🎯 Accuracy: {accuracy:.2f}%")
    print()
    
    print("📈 Per-Class Accuracy:")
    for label, stats in sorted(class_stats.items(), key=lambda x: x[1]['total'], reverse=True)[:10]:
        acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"   {label:30s}: {acc:6.2f}% ({stats['correct']}/{stats['total']})")
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv('quick_eval_results.csv', index=False)
    print()
    print("✅ Results saved to: quick_eval_results.csv")
    print("=" * 70)

if __name__ == '__main__':
    main()
