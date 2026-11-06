#!/usr/bin/env python3
"""CLI facade for the anomaly-detection pipeline runner."""

from anomaly_detection.runner import main, run_anomaly_detection_pipeline

__all__ = ["main", "run_anomaly_detection_pipeline"]


if __name__ == "__main__":
    main()
