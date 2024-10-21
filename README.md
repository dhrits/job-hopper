# JobHopper

Hi there! Welcome to JobHopper! Your personal job-coach and assistant. I can help you apply to new jobs, customize your resume or cover letter or give you general advice. To start, just upload a draft resume. Don't worry too much about presentation. To get started, I just need tob know about your professional background. I'll take care of the rest!

To get started, just upload a draft of a resume! It doesn't have to be polished. It just needs to help get the application started.

Note - **This project is meant to be a fun exploration! The developer makes no warranty on the effectiveness of this tool. The code is also meant to be used for educational purposes only.**

## Features

1. Resume analysis
2. Recommendation to job listings based on resume analysis
3. Job-fit analysis based on job description or job URL
4. Resume customization based on job description
5. Personalized cover-letter based on job description

Note: Since this is just an educational project to explore the capabilities of LLMs, *the job descriptions are not current and are based on datasets provided earlier* on [Kaggle](https://www.kaggle.com/datasets/promptcloud/indeed-job-posting-dataset). *This can be changed by implementing a web-based retriever.* 


## App Architecture
![App Architecture](./app_architecture.png)


This application is primarily built on top of [LangChain](https://www.langchain.com/), [LangGraph](https://www.langchain.com/langgraph), [OpenAI GPT-4o](https://openai.com/index/hello-gpt-4o/), and [Qdrant Vector Database](https://qdrant.tech/) using Qdrant Cloud. 

Retrieval from the vectorstore is performed using a fine-tuned version of [Snowflake Arctic Embed Long model](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-long) trained on [indeed jobs dataset](https://www.kaggle.com/datasets/promptcloud/indeed-job-posting-dataset) for best retrieval performance

While these components are customizable, this combination gave the best performance. 

Complex LLM applications can take a lot of prompt-engineering to get right. For this reason, orchestration of the above application is done using **LangGraph**. In particular, a **Assistant/Router** pattern is used along with a team of **Agent Experts** which specialize in one task. 

Additionally, in order to make the application flexible, **the Assistant is stateful** and maintains history of past conversations. Currently, there's no notion of sessions and login, but I plan on persisting this session-memory to a DB and implementing login as a later enhancement.




