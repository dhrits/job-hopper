import os
from typing import Tuple, List
import torch
from operator import itemgetter
from langchain_core.prompts.chat import SystemMessagePromptTemplate, ChatPromptTemplate, PromptTemplate
from langchain_core.messages import SystemMessage, ChatMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
from langchain_openai.chat_models import ChatOpenAI 
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader, SeleniumURLLoader
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools.render import render_text_description
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.conversation.base import ConversationChain
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.tools import tool
from langchain_community.tools import TavilySearchResults
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.documents.base import Document
from langchain.output_parsers import PydanticOutputParser

from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver

SYS_PROMPT = """
You are a helpful job-coach. Your goal is to help answer any of the user's questions about job-postings, job-requirements,
interviews etc. The user has provided you their resume below. You can make use of any of the tools available to you to best
answer the user's specific questions. 

Please follow the following guidelines:

1. Be as helpful as you can. Don't lie, but be encouraging

2. If the user goes off subject, remind them who you are and what you can help them with. Don't help them with anything not 
related to job search.

3. You know a lot about the user based on their resume and interactions with you. Personalize your interactions
with the user to be most effective in helping them.

4. Please think about all the tools available to you. In particular, if a user provides a URL, you can use tools available
to resolve the URL to a job description. 

5. If the user asks for a rewritten resume or cover-letter, please provide them the resume or cover letter in your response.

Begin with a warm greeting to the user (using their name if available) and tell them about yourself.


Resume:
{resume}
"""

# Tools

def url_resolver(url: str) -> str:
    """Given a `url`, resolves it to the contents of the page"""
    loader = SeleniumURLLoader(urls=[url])
    documents = loader.load()
    contents = documents[0].page_content
    return contents

def resume_writer(resume: str, job_description: str) -> str:
    """Given a `resume` and `job_description`, tailors this resume to the specific job description"""
    llm = ChatOpenAI(model='gpt-4o')
    prompt = """You are a helpful AI Assistant. Given a resume and a job description below, please tailor the resume
    to the specific job. Do not make up any details or add any facts not in the base resume

    Resume:
    {resume}

    Job Description:
    {job_description}
    """
    prompt = ChatPromptTemplate.from_template(prompt)
    chain = prompt | llm
    response = chain.invoke({'resume': resume, 'job_description': job_description}).content
    return response


def cover_letter_writer(resume: str, job_description: str) -> str:
    """Given a `resume` and `job_description`, returns a cover-letter tailored to this specific job description"""
    llm = ChatOpenAI(model='gpt-4o')
    prompt = """You are a helpful AI Assistant. Given a resume and a job description below. Please write a personalized
    cover letter to apply for this job description.

    Resume:
    {resume}

    Job Description:
    {job_description}
    """
    prompt = ChatPromptTemplate.from_template(prompt)
    chain = prompt | llm
    response = chain.invoke({'resume': resume, 'job_description': job_description}).content
    return response


def web_searcher(query: str) -> (str, List[Document]):
    """Given a `query`, searches the web for potential results and returns an answer and relevant context."""
    llm = ChatOpenAI(model='gpt-4o')
    prompt = """
    You are a helpful and kind assistant. Use the context provided below to answer the question.
    
    If you do not know the answer, or are unsure, say you don't know.
    
    Query:
    {question}
    
    Context:
    {context}
    """
    prompt = ChatPromptTemplate.from_template(prompt)
    retriever = TavilySearchAPIRetriever(k=5)
    rag_chain = (
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": prompt | llm, "context": itemgetter("context")}
    )
    results = rag_chain.invoke({'question': query})
    return results['response'], results['context']

def get_conversation_chain(resume, thread_id):
    """Gets the conversation chain based on resume and config"""
    sys_msg = SystemMessagePromptTemplate.from_template(SYS_PROMPT).format(resume=resume)
    tools = [url_resolver, resume_writer, cover_letter_writer, web_searcher]
    llm = ChatOpenAI(model='gpt-4o')
    llm_with_tools = llm.bind_tools(tools)

    def assistant(state: MessagesState):
       """Helper function to invoke assistant"""
       return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}
    
    memory = MemorySaver()
    builder = StateGraph(MessagesState)

    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )

    builder.add_edge("tools", "assistant")
    chain = builder.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": thread_id}}
    opener = chain.invoke(
        {"messages": [HumanMessage("Hello! How are you?")]},
        config)['messages'][-1].content
    return chain, opener
