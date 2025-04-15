from langchain.chat_models import ChatOpenAI
from langchain_core.runnables import RunnableLambda

from helper import json_parser
from prompt_template import prompt, question_suggestion_prompt, reformat_question_prompt

def combine_result_func(input):
    result = json_parser(input["suggestion"].content)
    result["answer"] = input["query"].content
    return result

llm = ChatOpenAI(
    temperature=0, 
    model="gpt-4o-mini"
)

# combine_result = RunnableLambda(combine_result_func)

# chain = prompt_with_question_suggestion | llm 
prompt_chain = prompt | llm
suggestion_chain = question_suggestion_prompt | llm 
reformat_chain = reformat_question_prompt | llm

"""
    !!! Experimental !!!
"""
# parallel_chain = {
#     "query" : prompt_chain,
#     "suggestion" : suggestion_chain
# } | combine_result
