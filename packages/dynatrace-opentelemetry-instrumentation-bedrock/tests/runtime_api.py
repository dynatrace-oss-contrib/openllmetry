import os
import trace

import boto3
import json
import unittest

from opentelemetry import trace
from opentelemetry.instrumentation.bedrock import BedrockInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def read_secret(secret: str):
    try:
        with open(f"/etc/secrets/{secret}", "r") as f:
            return f.read().rstrip()
    except Exception:
        return os.environ.get(secret.replace('-', '_').upper(), "")

resource = Resource.create(
    {"service.name": "test", "service.version": "0.0.0"}
)
exporter = InMemorySpanExporter()
provider = TracerProvider(resource=resource)
processor = SimpleSpanProcessor(exporter)
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

b = BedrockInstrumentor(
    service_name="test",
    event_logger=None,
)
b.instrument()

class TestBedrockRuntimeAPI(unittest.TestCase):

    def setUp(self):
        key = read_secret("aws-key")
        sec = read_secret("aws-secret")
        self.client = boto3.client(
            "bedrock-runtime",
            region_name="eu-central-1",
            aws_access_key_id=key,
            aws_secret_access_key=sec,
        )
        self.embedding_model = "amazon.titan-embed-text-v1"
        #self.model = "titan-text-lite-v1"
        self.model = "titan-text-express-v1"
        self.prompt = "put the string 1234561666 into a sentence please"
        self.system_prompts = [{"text": "You are an app that creates playlists for a radio station that plays rock and pop music."
                                   "Only return song names and the artist."}]
        self.guardrail = '5zwrmdlsra2e' #read_secret('aws-guardrail') # '859bmlna0an2', '5zwrmdlsra2e'
        self.messages = [{
            "role": "user",
            "content": [
                {"text": "Create a list of 3 pop songs."},
                {"text": "Make sure the songs are by artists from the United Kingdom."},
                {"text": "put the string 1234561666 into a sentence please."},
            ]
        }]

    def test_invoke(self):
        native_request = {
            "inputText": self.prompt,
            "textGenerationConfig": {
                "maxTokenCount": 512,
                "temperature": 0.5,
            },
        }
        request = json.dumps(native_request)
        api_response = self.client.invoke_model(
            modelId=("amazon." + self.model),
            body=request,
            guardrailIdentifier=self.guardrail,
            guardrailVersion="DRAFT",
            trace='ENABLED',
        )
        response = json.loads(api_response["body"].read())
        spans = exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]
        attrs = span.attributes
        self.assertEqual(attrs.get('gen_ai.system'), 'amazon')
        self.assertEqual(attrs.get('gen_ai.request.model'), self.model)
        self.assertEqual(attrs.get('gen_ai.response.model'), self.model)
        self.assertEqual(attrs.get('gen_ai.request.temperature'), 0.5)
        self.assertEqual(attrs.get('gen_ai.request.max_tokens'), 512)
        self.assertEqual(attrs.get('gen_ai.usage.prompt_tokens'), response['inputTextTokenCount'])
        self.assertEqual(attrs.get('gen_ai.usage.completion_tokens'), response['results'][0]['tokenCount'])
        self.assertEqual(attrs.get('gen_ai.prompt.0.user'), self.prompt)
        self.assertNotEqual(attrs.get('gen_ai.completion.0.content'), '')


    def test_converse(self):
        # We need a different model, check the nove one
        # https://docs.aws.amazon.com/nova/latest/userguide/using-converse-api.html
        inference_config = {"temperature": 0.5}
        additional_model_fields = {
            "inferenceConfig": {
                "topK": 20
            }
        }
        guardrail = {
            'guardrailIdentifier': self.guardrail,
            'guardrailVersion': 'DRAFT',
            'trace': 'enabled'
        }
        response = self.client.converse(
            modelId="amazon." + self.model,
            messages=self.messages,
            guardrailConfig=guardrail
            #system=self.system_prompts,
            #inferenceConfig=inference_config,
            #additionalModelRequestFields=additional_model_fields
        )



if __name__ == '__main__':
    unittest.main()