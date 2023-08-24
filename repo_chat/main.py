from git import Repo
from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser        
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
# import tiktoken
from langchain.callbacks import get_openai_callback
import os, shutil


class RepoChat():
    def __init__(self, codebase_path: str, model_name="gpt-3.5-turbo-16k", suffixes=[".py", ".ipynb", ".md"]):                
        self.model_name = model_name
        self.cb = None

        if "github.com" in codebase_path: # if it is a github
            codebase_local_path = "./temp_repo/"
            if os.path.isdir(codebase_local_path): shutil.rmtree(codebase_local_path, )
            print(f"Cloning from {codebase_path} to {codebase_local_path}")
            repo = Repo.clone_from(codebase_path, to_path=codebase_local_path)
            codebase_path = codebase_local_path
        else: #if it is local
            print("Loading from local repo...")
        
        # Load
        loader = GenericLoader.from_filesystem(
            codebase_path,
            glob="**/*",
            suffixes=suffixes,
            parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
            show_progress=True,
        )
        documents = loader.load()
        # len(documents)

        python_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, 
                                                                    chunk_size=2000, 
                                                                    chunk_overlap=200,)
        texts = python_splitter.split_documents(documents)

        print(f"Loaded {len(documents)} documents, split into {len(texts)} texts. \nLoading into chroma. This may take a while.")

        with get_openai_callback() as cb:
            # Set up chroma db and retriever
            db = Chroma.from_documents(texts, OpenAIEmbeddings(disallowed_special=()))
            retriever = db.as_retriever(
                search_type="mmr",  # Also test "similarity"
                search_kwargs={"k": 8},
            )

            # Set up LLM with memory
            llm = ChatOpenAI(model_name=self.model_name) 
            memory = ConversationSummaryMemory(llm=llm,memory_key="chat_history",return_messages=True)
            self.qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)
        
        print("Repo loaded into langchain.")
        # self.update_usage(cb)
        # self.print_usage()
        result = self.qa("Give a summary of the repo you have access to based on its readme, main program, or any other useful code you can find.")
        print(result['answer'])


    def update_usage(self, cb):
        if self.cb is None:
            self.cb = cb
            return
        else:
            self.cb.total_tokens += cb.total_tokens
            self.cb.prompt_tokens += cb.prompt_tokens
            self.cb.completion_tokens += cb.completion_tokens
            self.cb.total_cost += cb.total_cost
    
    def print_usage(self):
        print(f'Total tokens: {self.cb.total_tokens}')
        print(f'Prompt tokens: {self.cb.prompt_tokens}')
        print(f'Completion tokens: {self.cb.completion_tokens}')
        print(f'Total cost: {self.cb.total_cost}')

    def chat(self, question):
        with get_openai_callback() as cb:
            result = self.qa(question)

        self.update_usage(cb)

        return result['answer']


                