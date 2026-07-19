from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

search_tool = TavilySearchResults(max_result=5)

llm = ChatMistralAI(model_name="mistral-small-2603")



prompt = ChatPromptTemplate(
   ("""
    You are a helpfull assistant
    
    summarize the following news into clear bullet points
    
    {news}
    """   ) 
)

chain = prompt | llm | StrOutputParser()

while True:
    user_query= input("Enter news topic (or type 'exit' to quit): ").strip()
    
    if user_query.lower() == "exit":
        break
    
    news_result = search_tool.run(user_query)
    
    result = chain.invoke({"news": news_result})
    print(result)
    
    print("\n" + "-" * 40 + "\n")