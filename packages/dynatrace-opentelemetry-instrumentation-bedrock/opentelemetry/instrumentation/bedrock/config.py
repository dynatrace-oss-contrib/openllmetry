from dynatrace_ai_logging import DtAiLogging

class Config:
    enrich_token_usage = False
    exception_logger = None
    event_logger:DtAiLogging = None
    service_name:str = ""
