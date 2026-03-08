import logging
import sys
import os

class SafeStreamHandler(logging.StreamHandler):
    """Custom StreamHandler that handles Unicode encoding errors on Windows console."""
    def emit(self, record):
        try:
            msg = self.format(record)
            self.stream.write(msg + self.terminator)
            self.flush()
        except UnicodeEncodeError:
            # Fallback: simple ascii-only version for the console
            msg = self.format(record).encode('ascii', 'replace').decode('ascii')
            self.stream.write(msg + self.terminator)
            self.flush()

def get_logger(name: str):
    """Returns a pre-configured logger with a clean, descriptive format."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Ensure logs directory exists at project root
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Create console handler - Using SafeStreamHandler for stability
        handler = SafeStreamHandler(sys.stdout)
        formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Log to file inside the logs folder - Always FORCED UTF-8
        log_file = os.path.join(log_dir, "system.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8', errors='replace')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger
