    import os
import glob

logs = sorted(glob.glob('logs/train_*.log'), reverse=True)
if logs:
    latest = logs[0]
    size = os.path.getsize(latest)
    print(f"Latest log: {os.path.basename(latest)}")
    print(f"Size: {size} bytes")
    
    # Read all lines
    with open(latest, 'r', encoding='utf-8', errors='ignore') as f:
        all_lines = f.readlines()
        print(f"Total lines: {len(all_lines)}")
        
        # Show content from line 0 to max 200
        if len(all_lines) > 0:
            print("\n=== Last 100 lines ===")
            for line in all_lines[-100:]:
                print(line.rstrip())
else:
    print("No log files found")
