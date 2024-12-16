import os
from dotenv import load_dotenv
import pymysql

# LangChain imports
from langchain_experimental.sql import SQLDatabaseSequentialChain
from langchain.utilities import SQLDatabase
from langchain.chat_models import ChatGooglePalm
from langchain.prompts import PromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import FewShotPromptTemplate

# Load environment variables
load_dotenv()

# Few-shots examples
from few_shots import few_shots  # Import examples for the few-shot prompt


def get_few_shot_db_chain():
    # Database connection details
    db_user = "root"
    db_password = "root"
    db_host = "localhost"
    db_name = "atliq_tshirts"

    # Create SQL Database instance
    db = SQLDatabase.from_uri(
        f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",
        sample_rows_in_table_info=3
    )

    # Initialize the LLM
    llm = ChatGooglePalm(
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.1
    )

    # Initialize embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    to_vectorize = [" ".join(example.values()) for example in few_shots]
    vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=few_shots)

    # Configure example selector
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=2
    )

    # Define MySQL-specific prompt
    mysql_prompt = """
    You are a MySQL expert. Given an input question, first create a syntactically correct MySQL query to run, 
    then look at the results of the query and return the answer to the input question.
    Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results 
    using the LIMIT clause as per MySQL. You can order the results to return the most informative data in the database.
    Never query for all columns from a table. You must query only the columns that are needed to answer the question. 
    Wrap each column name in backticks (`) to denote them as delimited identifiers.
    Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that 
    do not exist. Also, pay attention to which column is in which table.
    Pay attention to use CURDATE() function to get the current date, if the question involves "today".

    Use the following format:
    Question: Question here
    SQLQuery: Query to run with no pre-amble
    SQLResult: Result of the SQLQuery
    Answer: Final answer here

    No pre-amble.
    """

    # Define the example prompt
    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult", "Answer"],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}"
    )

    # Create FewShotPromptTemplate
    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=mysql_prompt,
        suffix="Please provide your input query.",
        input_variables=["input", "table_info", "top_k"]
    )

    # Initialize SQLDatabaseSequentialChain
    chain = SQLDatabaseSequentialChain.from_llm(
        llm=llm,
        database=db,
        verbose=True,
        prompt=few_shot_prompt
    )

    return chain


# Test the function
if __name__ == "__main__":
    chain = get_few_shot_db_chain()
    question = "What are the top 5 most sold t-shirts in the last month?"
    response = chain.run(input=question)
    print(response)
