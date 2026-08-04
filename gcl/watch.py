"""Background watcher: wait for gcl experiment (PID 400) to finish then emit a summary."""
import os, sys, time

def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "/root/gcl/runs/drift_credible/summary.md"
    deadline = time.time() + 7200
    while time.time() < deadline:
        if os.path.exists(out):
            print("SUMMARY_READY")
            return
        if not os.path.exists(out):
            pass
        time.sleep(120)
    print("TIMEOUT")

if __name__ == "__main__":
    main()
