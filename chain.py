import os
from typing import Tuple, List
import torch
from operator import itemgetter
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams
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
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.documents.base import Document
from langchain.output_parsers import PydanticOutputParser

from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, RemoveMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv; _ = load_dotenv()

MODEL = "gpt-4o"
MODEL_MINI = "gpt-4o-mini"
# MODEL = 'o1-preview'
# MODEL_MINI = 'o1-mini'

EMBED_MODEL_URL = "https://uniui42lc3nrxgsj.us-east-1.aws.endpoints.huggingface.cloud"
COLLECTION_NAME = "indeed_jobs_db_long"

_qclient = QdrantClient(
    url=os.environ.get('QDRANT_DB_BITTER_MAMMAL'), # Name of the qdrant cluster is bitter_mammal
    api_key=os.environ.get('QDRANT_API_KEY_BITTER_MAMMAL'),
)

_embeddings = HuggingFaceEndpointEmbeddings(
    model=EMBED_MODEL_URL,
    task="feature-extraction",
    huggingfacehub_api_token=os.environ["HF_TOKEN"],
)

_vector_store = QdrantVectorStore(
    client=_qclient,
    collection_name=COLLECTION_NAME,
    embedding=_embeddings,
)

_retriever = _vector_store.as_retriever(search_kwargs={"k": 10})

SYS_PROMPT = """
You are a helpful job-coach. Your goal is to help answer any of the user's questions about job-postings, job-requirements,
interviews, contents of resume etc. The user has provided you their resume below. You can make use of any of the tools available to you to best
answer the user's questions. 

Think step-by-step about the user's question and what (if any) tool calls you may need to answer the question.

Additionally please follow the following guidelines in your responses:

1. Be as helpful as you can. Don't lie, but be encouraging.

2. If the user goes off subject, remind them who you are and what you can help them with. Don't help them with anything not 
related to job search.

3. You know a lot about the user based on their resume and interactions with you. Personalize your interactions
with the user to be most effective in helping them.

4. Please think about all the tools available to you. In particular, if a user provides a URL, you can use tools available
to resolve the URL to a job description. 

5. If the user asks you to edit their resume or cover-letter, MAKE SURE you actually share the resulting resume and cover letter in your final message.

6. If the user asks for open roles, MAKE SURE you provide all details about open roles.

Begin with a warm greeting to the user (using their name if available) and tell them about yourself.


Resume:
{resume}
"""

class SummaryState(MessagesState):
    summary: str

# Tools

def url_resolver(url: str) -> str:
    """Given a `url`, resolves it to the contents of the page"""
    loader = SeleniumURLLoader(urls=[url])
    documents = loader.load()
    contents = documents[0].page_content
    return contents

def resume_consultant(resume: str, question: str) -> str:
    """Given a `resume` and a `question`, answers the question using a database of real-time and current job-postings
    using the resume as a reference to find matching job postings"""

    prompt = """
    Given the question, resume and context below, answer the question based on the contents of the resume and the context.
    Provide as much detail from the context as possible.

    Question:
    {question}

    Resume:
    {resume}

    Context:
    {context}
    """
    llm = ChatOpenAI(model=MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_template(prompt)
    rag_chain = (
            {"context": (lambda d: d['question'] + "\n\nResume:\n" + d['resume']) | _retriever, "question": itemgetter("question"), "resume": itemgetter("resume")}
            | RunnablePassthrough.assign(context=itemgetter("context"), resume=itemgetter("resume"))
            | {"response": prompt | llm, "context": itemgetter("context")}
        )

    resp = rag_chain.invoke({"resume": resume, "question": question})
    return resp["response"].content


def jobs_consultant(question: str) -> str:
    """Given a `question` about jobs or job requirements, answers the `question` using a database of real-time and current job-posting"""
    prompt = """
    You are a professional job consultant. Use the context provided below to answer the question.
    
    If you do not know the answer, or are unsure, say you don't know.
    
    Question:
    {question}
    
    Context:
    {context}
    """
    llm = ChatOpenAI(model=MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_template(prompt)
    rag_chain = (
        {"context": itemgetter("question") | _retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": prompt | llm, "context": itemgetter("context")}
    )
    results = rag_chain.invoke({'question': question})
    return results['response'].content


def resume_writer(resume: str, job_description: str) -> str:
    """Given a `resume` and `job_description`, tailors this resume to the specific job description"""
    llm = ChatOpenAI(model=MODEL)
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
    llm = ChatOpenAI(model=MODEL)
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
    llm = ChatOpenAI(model=MODEL)
    prompt = """
    You are a helpful and kind assistant. Use the context provided below to answer the question.
    
    If you do not know the answer, or are unsure, say you don't know.
    
    Query:
    {question}
    
    Context:
    {context}
    """
    prompt = ChatPromptTemplate.from_template(prompt)
    retriever = TavilySearchAPIRetriever(k=20)
    rag_chain = (
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": prompt | llm, "context": itemgetter("context")}
    )
    results = rag_chain.invoke({'question': query})
    return results['response'], results['context']

def summarize(state: SummaryState):
    """
    Summarizes the conversation if the number of messages in state are too many messages.
    It checks if the number of messages is > 60. If so, it summarizes the conversation in
    summary field and deletes old messages.
    """
    if len(state['messages']) < 60:
        # No concern here
        return {'messages': state['messages']}
    else:
        print("[summarize] Too many messages. Pruning and summarizing")
        model = ChatOpenAI(model=MODEL, temperature=0)
        # Check if previous summary exists
        summary = state.get('summary', "")
        if summary:
            # A summary already exists
            summary_message = (
                f"This is summary of the conversation to date: {summary}\n\n"
                "Extend the summary by taking into account the new messages above:"
            )
        else:
            summary_message = "Create a summary of the conversation above:"
        messages = state['messages'] + [HumanMessage(content=summary_message)]
        response = model.invoke(messages)
        # Delete all but last two messages
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
        return {"summary": response.content, "messages": delete_messages}

def get_conversation_chain(resume, thread_id):
    """Gets the conversation chain based on resume and config"""
    sys_msg = SystemMessagePromptTemplate.from_template(SYS_PROMPT).format(resume=resume)
    tools = [url_resolver, resume_consultant, jobs_consultant, resume_writer, cover_letter_writer, web_searcher]
    llm = ChatOpenAI(model=MODEL)
    llm_with_tools = llm.bind_tools(tools)

    def assistant(state: SummaryState):
        summary = state.get('summary', '')
        if summary:
            summary_message = f"Summary of conversation earlier: {summary}"
            messages = [SystemMessage(summary_message)] + state['messages']
        else:
            messages = state['messages']
    
        return {"messages": [llm_with_tools.invoke([sys_msg] + messages)]}
   
    memory = MemorySaver()
    builder = StateGraph(SummaryState)

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
