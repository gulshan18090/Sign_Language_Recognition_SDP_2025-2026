"""
Full Automated Inference Evaluation
Evaluates model on entire Cam2 dataset and generates accuracy metrics
"""
import os
import pandas as pd
import subprocess
from collections import defaultdict
import json

def get_expected_label(video_path, csv_path):
    """Extract expected label from CSV based on video path"""
    idd = os.path.basename(os.path.dirname(video_path))
    df = pd.read_csv(csv_path, sep=';', header=None, names=['id', 'sentence', 'sign_language'])
    row = df[df['id'].astype(str) == str(idd)]
    if not row.empty:
        return row.iloc[0]['sign_language']
    return None

def run_inference(video_path):
    """Run inference on a video and extract predicted sentence"""
    result = subprocess.run([
        'python3', 'inference.py', video_path
    ], capture_output=True, text=True, timeout=60)
    output = result.stdout
    
    # Extract predicted sentence from output
    lines = output.splitlines()
    pred = None
    for i, line in enumerate(lines):
        if '📝 PREDICTED SENTENCE:' in line:
            if i+2 < len(lines):
                pred = lines[i+2].strip()
                break
    return pred

def calculate_metrics(results):
    """Calculate accuracy metrics from results"""
    total = len(results)
    correct = sum(1 for r in results if r['match'])
    accuracy = (correct / total * 100) if total > 0 else 0
    
    # Per-class accuracy
    class_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    for r in results:
        expected = r['expected']
        if expected:
            class_stats[expected]['total'] += 1
            if r['match']:
                class_stats[expected]['correct'] += 1
    
    # Calculate per-class accuracy
    class_accuracy = {}
    for label, stats in class_stats.items():
        if stats['total'] > 0:
            class_accuracy[label] = (stats['correct'] / stats['total'] * 100)
    
    return {
        'total_videos': total,
        'correct_predictions': correct,
        'overall_accuracy': accuracy,
        'class_accuracy': class_accuracy,
        'class_stats': dict(class_stats)
    }

def main():
    csv_path = 'drive/sentences_all.csv'
    video_base_dir = 'drive/Video/Cam2/'
    
    print("=" * 70)
    print("FULL AUTOMATED INFERENCE EVALUATION")
    print("=" * 70)
    print(f"📂 Video directory: {video_base_dir}")
    print(f"📄 CSV path: {csv_path}")
    print()
    
    # Find all video files in Cam2
    all_videos = []
    for root, dirs, files in os.walk(video_base_dir):
        for file in files:
            if file.endswith('.mp4') or file.endswith('.avi'):
                video_path = os.path.join(root, file)
                all_videos.append(video_path)
    
    print(f"📊 Found {len(all_videos)} videos")
    print(f"⏳ Starting evaluation... (this may take a while)")
    print()
    
    results = []
    processed = 0
    errors = 0
    
    for video_path in all_videos:
        processed += 1
        video_name = os.path.basename(video_path)
        video_id = os.path.basename(os.path.dirname(video_path))
        
        # Progress indicator
        if processed % 10 == 0:
            print(f"Progress: {processed}/{len(all_videos)} ({processed/len(all_videos)*100:.1f}%)")
        
        try:
            expected = get_expected_label(video_path, csv_path)
            if expected is None:
                continue  # Skip videos without labels
            
            predicted = run_inference(video_path)
            
            # Case-insensitive comparison
            match = (expected is not None and predicted is not None and 
                     expected.lower().strip() == predicted.lower().strip())
            
            results.append({
                'video_id': video_id,
                'video_name': video_name,
                'video_path': video_path,
                'expected': expected,
                'predicted': predicted,
                'match': match
            })
            
        except Exception as e:
            errors += 1
            print(f"❌ Error processing {video_name}: {str(e)}")
            continue
    
    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print()
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Print summary
    print("📊 OVERALL METRICS:")
    print(f"   Total videos evaluated: {metrics['total_videos']}")
    print(f"   Correct predictions: {metrics['correct_predictions']}")
    print(f"   Overall accuracy: {metrics['overall_accuracy']:.2f}%")
    print(f"   Errors during processing: {errors}")
    print()
    
    # Print per-class accuracy (top 10 by sample count)
    print("📈 PER-CLASS ACCURACY (Top 10 by sample count):")
    class_stats_sorted = sorted(
        metrics['class_stats'].items(), 
        key=lambda x: x[1]['total'], 
        reverse=True
    )[:10]
    
    for label, stats in class_stats_sorted:
        acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"   {label:30s}: {acc:6.2f}% ({stats['correct']}/{stats['total']})")
    
    print()
    
    # Save detailed results
    df_results = pd.DataFrame(results)
    df_results.to_csv('inference_eval_full_results.csv', index=False)
    print(f"✅ Detailed results saved to: inference_eval_full_results.csv")
    
    # Save metrics as JSON
    with open('inference_eval_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"✅ Metrics saved to: inference_eval_metrics.json")
    
    # Create confusion summary
    print()
    print("🔍 CONFUSION ANALYSIS (Top 5 misclassifications):")
    misclassified = [r for r in results if not r['match']]
    confusion_counts = defaultdict(int)
    for r in misclassified:
        key = f"{r['expected']} → {r['predicted']}"
        confusion_counts[key] += 1
    
    top_confusions = sorted(confusion_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    for confusion, count in top_confusions:
        print(f"   {confusion}: {count} times")
    
    print()
    print("=" * 70)

if __name__ == '__main__':
    main()

