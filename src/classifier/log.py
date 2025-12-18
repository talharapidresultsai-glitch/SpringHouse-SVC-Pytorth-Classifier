import logging
import json
import requests
import re

NAMESPACE = 'springhouse.model.classifier'

class LogServiceFormatter(logging.Formatter):
    def __init__(self):
        super().__init__()

    def format(self, record):
        dict_record = {}
        dict_record['namespace'] = NAMESPACE
        dict_record['module'] = record.module
        dict_record['source'] = record.pathname
        dict_record['severity'] = record.levelname
        dict_record['processid'] = record.process
        dict_record['processname'] = record.module
        dict_record['timestamp'] = record.created
        dict_record['relativeCreated'] = record.relativeCreated
        dict_record['loglevel'] = record.levelno
        dict_record['lineno'] = record.lineno
        message = record.getMessage()
        message_len = len(message)
        if message_len > 0:
            message_allowed = re.sub('[^a-zA-Z0-9 \-\?\,\.]', ' ', message)
            dict_record['message'] = message_allowed if message_len < 256 else message_allowed[:256]
        else:
            dict_record['message'] = message
        return json.dumps(dict_record)

class LogServiceHandler(logging.Handler):
    def __init__(self, url, level=logging.INFO):
        super(LogServiceHandler, self).__init__()
        self.url = url
        self.level = level

    def emit(self, record):
        log_entry = self.format(record)
        try:
            requests.post(self.url,
                        json= {"content": log_entry},
                        timeout=2.0)
        except Exception:
            self.handleError(record)
