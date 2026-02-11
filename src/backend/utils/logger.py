"""
Logging utilities for NeuroTract

Provides structured logging with file and console output.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class NeuroTractLogger:
    """Centralized logger for NeuroTract operations"""

    def __init__(
        self,
        name: str = "neurotract",
        log_dir: str = "logs",
        level: int = logging.INFO,
        console: bool = True,
        file: bool = True
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers = []  # Clear existing handlers

        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        simple_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )

        # Console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(simple_formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if file:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = log_path / f"neurotract_{timestamp}.log"

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(detailed_formatter)
            self.logger.addHandler(file_handler)

            self.logger.info(f"Logging to: {log_file}")

    def get_logger(self) -> logging.Logger:
        """Get the configured logger instance"""
        return self.logger


# Global logger instance
_global_logger: Optional[NeuroTractLogger] = None


def get_logger(name: str = "neurotract") -> logging.Logger:
    """Get or create global logger"""
    global _global_logger
    if _global_logger is None:
        _global_logger = NeuroTractLogger(name)
    return _global_logger.get_logger()


def log_decision(
    decision_id: str,
    component: str,
    decision: str,
    rationale: str,
    parameters: dict,
    output_file: str = "analysis_and_decisions/decision_log.md"
):
    """
    Log a decision to the analysis_and_decisions folder

    Args:
        decision_id: Unique identifier for decision
        component: Component/module name
        decision: What was decided
        rationale: Why this decision was made
        parameters: Dictionary of parameters and values
        output_file: Path to decision log file
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    entry = f"""
### [{decision_id}] Automated Decision
**Timestamp**: {timestamp}
**Component**: {component}
**Decision Maker**: automation
**Status**: implemented

**Decision**: {decision}

**Rationale**: {rationale}

**Parameters & Thresholds**:
"""
    for key, value in parameters.items():
        entry += f"- {key} = {value}\n"

    entry += "\n---\n"

    # Create directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Append to decision log
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(entry)

    logger = get_logger()
    logger.info(f"Decision logged: {decision_id}")
