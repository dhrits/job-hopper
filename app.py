import os
import uuid
from typing import List
from chainlit.types import AskFileResponse
from langchain_core.messages import SystemMessage, ChatMessage, HumanMessage
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.schema.runnable.config import RunnableConfig
import chainlit as cl
from dotenv import load_dotenv

from chain import get_conversation_chain

_ = load_dotenv()

def process_input_file(file: AskFileResponse):
    """Reads the input `file` and returns the resulting documents"""
    # import tempfile
    # with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as temp_file:
    #     temp_file_path = temp_file.name
    # with open(temp_file_path, "wb") as f:
    #     f.write(file.content)
    temp_file_path = file.path
    loader = PyMuPDFLoader(temp_file_path)
    documents = loader.load()
    return documents[0].page_content


@cl.on_chat_start
async def on_chat_start():
    """Begin the chat"""
    files = None

    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload your resume to begin! Once you're done uploading, processing may take a few seconds....",
            accept=["application/pdf"],
            max_size_mb=1,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(
        content=f"Processing `{file.name}`..."
    )
    await msg.send()

    # load the file
    resume = process_input_file(file)
    thread_id = str(uuid.uuid4())
    chain, opener = get_conversation_chain(resume, thread_id)
    
    # Let the user know that the system is ready
    msg.content = opener
    await msg.update()
    thread_id = str(uuid.uuid4())
    cl.user_session.set("chain", chain)
    cl.user_session.set("thread_id", thread_id)


@cl.on_message
async def main(message):
    """Invoked on user message"""
    chain = cl.user_session.get("chain")
    thread_id = cl.user_session.get("thread_id")
    config = {"configurable": {"thread_id": thread_id}}

    msg = cl.Message(content="")
    messages = [HumanMessage(content=message.content)]
    result = chain.astream({'messages': messages}, config, stream_mode="updates")

    async for stream_resp in result:
        if 'assistant' in stream_resp:
            await msg.stream_token(stream_resp['assistant']['messages'][-1].content)

    await msg.send()
