import argparse
import logging
import os

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from kobart_chit_chat import KoBARTConditionalGeneration


class UserMessage(BaseModel):
    userMessage: str


class AiMessage(BaseModel):
    response: str


class MainFastAPI:

    def __init__(self, model: KoBARTConditionalGeneration):

        # setup logging
        # client = google.cloud.logging_v2.Client()
        # client.setup_logging()

        self._url = '0.0.0.0'
        self._logger = logging.getLogger("MainFastAPI")
        self._app = FastAPI(
            title='web service for sentimental chatbot',
            version='0.0.1',
            root_path='/'
        )
        self._app.add_api_route(
            **self._get_ai_message_property,
            endpoint=self.get_ai_message,
            methods=["POST"]
        )
        self._model = model
        self._model.model.eval()

    @property
    def _get_ai_message_property(self):
        return {
            "path": "/"
        }

    def get_ai_message(self, message: UserMessage):
        user_message = message.userMessage
        ai_output = self._model.chat(user_message)
        return {"response": ai_output}

    def start_app(self):
        uvicorn.run(self._app, host=self._url, port=int(os.getenv('PORT') or 8080))
