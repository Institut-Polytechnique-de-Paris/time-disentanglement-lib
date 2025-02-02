# -*-Encoding: utf-8 -*-
"""
Authors: Khalid Oublal, PhD IPP / OneTech (khalid.oublal@poytechnique.edu)
"""

import logging
import sys
import os
from datetime import datetime

class DebugLogger:
    def __init__(self, log_dir=None):
        self.log_dir = log_dir
        self.log_file_path = self._generate_log_file_path()

    def _generate_log_file_path(self):
        if self.log_dir is None:
            return None

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file_name = f"debug_{current_time}.log"
        return os.path.join(self.log_dir, log_file_name)

    def enable_logging(self):
        if self.log_file_path is None:
            return

        # Configure logging
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename=self.log_file_path)

        # Redirect stdout and stderr to logging
        stdout_logger = logging.getLogger('STDOUT')
        stderr_logger = logging.getLogger('STDERR')

        class LogWriter:
            def __init__(self, logger, level):
                self.logger = logger
                self.level = level

            def write(self, message):
                if message.rstrip() != '':
                    self.logger.log(self.level, message.rstrip())

            def flush(self):
                pass

        sys.stdout = LogWriter(stdout_logger, logging.INFO)
        sys.stderr = LogWriter(stderr_logger, logging.ERROR)

    def disable_logging(self):
        if self.log_file_path is None:
            return

        # Restore stdout and stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__