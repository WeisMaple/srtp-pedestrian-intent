"""Experimental fixed-camera classroom demo; requires local model weights."""
from modern_pipeline.bootstrap import configure

configure()

if __name__ == '__main__':
    from modern_pipeline.camera_cli import main
    main()
