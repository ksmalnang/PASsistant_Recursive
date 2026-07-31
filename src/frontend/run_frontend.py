"""Helper script to start the Streamlit frontend server."""

import subprocess
import sys
from pathlib import Path

def main() -> None:
    """Run the Streamlit application using subprocess."""
    frontend_dir = Path(__file__).resolve().parent
    app_path = frontend_dir / "app.py"
    port = "8501"
    
    print(f"Starting Streamlit frontend server on port {port}...")
    try:
        subprocess.run(
            [
                "streamlit",
                "run",
                str(app_path),
                "--server.port",
                port,
                "--server.address",
                "0.0.0.0",
            ],
            check=True,
        )
    except KeyboardInterrupt:
        print("\nFrontend server stopped by user.")
    except Exception as exc:
        print(f"Error starting Streamlit server: {exc}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
