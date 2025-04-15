from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate
)
from langchain_core.messages import SystemMessage, HumanMessage

prompt_with_question_suggestion = ChatPromptTemplate([
    MessagesPlaceholder(variable_name="chat_history"),
    SystemMessage("Act as an assistance, using a friendly tone, sounds helpful"),
    SystemMessagePromptTemplate.from_template("Reply to user or if you are questioned then answer any user's questions based on the context only if the context is not empty: {context}"),
    SystemMessage("Then generate a suggestion list of 3 questions the user most likely to ask you"),
    SystemMessage(content="Return in JSON format as follow: \n"),
    SystemMessage(content='{"answer" : "answer goes here", "suggestion": ["question1", "question2", "question3"]}'),
    HumanMessagePromptTemplate.from_template("{input}")
])

prompt = ChatPromptTemplate([
    MessagesPlaceholder(variable_name="chat_history"),
    SystemMessagePromptTemplate.from_template("""
        Act as an assistance, using a friendly and happy tone, using a friendly tone, sounds helpful
        \n - Answer any user's questions below based SOLY on the provided context, do not add any information, if there is no context then answer you don't have any information of it.
        \n - If the user's current question's topic is not related to the last then forget about the last question
    """),
    HumanMessagePromptTemplate.from_template("Question: {input}"),
    SystemMessagePromptTemplate.from_template("\nContext: {context}")
])

question_suggestion_prompt = ChatPromptTemplate([
    HumanMessagePromptTemplate.from_template("Question: {input}"),
    SystemMessage(content="""
        Act as an assistance, using a friendly and happy tone, sounds helpful. 
        \n - Based on the the question asked above, generate a suggestion list of 3 questions the user most likely to ask you, in the language of the question
        \n - If the related field below is equal to False then return general question only about how to use the chatbot
        \n - Return in JSON format as follow: "{"suggestion": ["question1", "question2", "question3"]}"
    """),
    SystemMessagePromptTemplate.from_template("Related: {related}")
])

reformat_question_prompt = ChatPromptTemplate([
    MessagesPlaceholder(variable_name="chat_history"),
    SystemMessage(content="""
        If the current question is not related to the last conversations then FORGET about the last conversations. 
        
        \nIf the user's question still related, then based on the previous conversations, if the input is a question then rewrite the user's question to clarify the actual intent 
        behind the question while optimizing it for similarity search engines to retrieve relevant data
        needed to better answer the question.
        
        \n Answer in the language the user's question is using, in form of a user's question to the AI.
    """),
    HumanMessagePromptTemplate.from_template("Question: {input}")
    # Problems: 
    # If you suddenly switch topic, the llm will try to rewrite the current question to the last topic (solved)
    # The output is in the form of the question to the user
    # The rewrite question will not be optimal and will return varying amount of context leading to varying the length of the answer
])
