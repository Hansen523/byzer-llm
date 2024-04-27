import time
import json
import uuid
import requests
import asyncio
import traceback
import threading
from typing import List, Dict, Any, Union

class CustomSaasAPI:

    def __init__(self, infer_params: Dict[str, str]) -> None:
        self.api_key = infer_params["saas.api_key"]
        self.model = infer_params.get("saas.model", "azure_tts")
        self.app_id = infer_params.get("saas.app_id", "")
        self.base_url = infer_params.get("saas.base_url", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")
        if self.base_url.endswith("/"):
            self.base_url = self.base_url[:-1]

        self.max_retries = int(infer_params.get("saas.max_retries", 10))

        self.meta = {
            "model_deploy_type": "saas",
            "backend": "saas",
            "support_stream": True,
            "model_name": self.model,
        }

    def get_meta(self) -> List[Dict[str, Any]]:
        return [self.meta]

    async def text_to_speech(self, stream: bool, ins: str, voice: str, chunk_size: int = None, response_format: str = "mp3", **kwargs):
        token_header = {
            'Ocp-Apim-Subscription-Key': self.api_key
        }
        token_response = requests.post(self.base_url, headers=token_header)
        if token_response.status_code != 200:
            raise Exception("Failed to retrieve authentication token")

        access_token = token_response.text
        tts_header = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/ssml+xml',
            'X-Microsoft-OutputFormat': response_format
        }
        ssml = f'''
        <speak version='1.0' xml:lang='en-US'>
            <voice name='{voice}'>
                {ins}
            </voice>
        </speak>
        '''

        response = requests.post(f"{self.base_url}/cognitiveservices/v1", headers=tts_header, data=ssml)
        if response.status_code != 200:
            raise Exception(f"Failed to synthesize speech: {response.text}")

        if stream:
            server = ray.get_actor("BlockBinaryStreamServer")

            def writer():
                request_id = str(uuid.uuid4())
                try:
                    chunk = response.content
                    ray.get(server.add_item.remote(request_id,
                                                   StreamOutputs(outputs=[SingleOutput(text=chunk, metadata=SingleOutputMeta(
                                                       input_tokens_count=0,
                                                       generated_tokens_count=0,
                                                   ))])
                                                   ))
                except:
                    traceback.print_exc()
                ray.get(server.mark_done.remote(request_id))

            threading.Thread(target=writer, daemon=True).start()
            return [("RUNNING", {"metadata": {"request_id": request_id, "stream_server": "BlockBinaryStreamServer"}})]
        else:
            return [(response.content, {"metadata": {}})]

    async def async_stream_chat(self, tokenizer, ins: str, his: List[Dict[str, Any]] = [],
                                max_length: int = 4096,
                                top_p: float = 0.7,
                                temperature: float = 0.9, **kwargs):

        stream = kwargs.get("stream", False)

        messages = [{"role": message["role"], "content": message["content"]} for message in his] + [{"role": "user", "content": ins}]
        last_message = messages[-1]["content"]

        if isinstance(last_message, dict) and "input" in last_message:
            voice = last_message.get("voice", self.voice_type)
            chunk_size = last_message.get("chunk_size", None)
            return await self.text_to_speech(stream=stream,
                                             ins=ins,
                                             voice=voice,
                                             chunk_size=chunk_size)

        raise Exception("Invalid input")