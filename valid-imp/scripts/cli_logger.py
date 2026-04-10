import sys
from datetime import datetime
from pathlib import Path


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)

    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except Exception:
                pass

    def isatty(self):
        return any(getattr(f, 'isatty', lambda: False)() for f in self.files)


def setup_script_logging(log_path: Path):
    """Redirect stdout/stderr to both console and a log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, 'a', encoding='utf-8')
    header = f"\n\n=== {datetime.now().isoformat(sep=' ', timespec='seconds')} {log_path.name} ===\n"
    log_file.write(header)
    log_file.flush()
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
