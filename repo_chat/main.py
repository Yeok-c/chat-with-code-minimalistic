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
from timeit import default_timer as timer
import time
import numpy as np

class RepoChat():
    def __init__(self, 
                 codebase_path: str, 
                 codebase_local_path: str = "./temp_repo/",
                 db_persist_directory: str = "chroma_db",
                 query_tokens: int = 4000,
                 db_creation_mode: str = "local",
                 model_name="gpt-3.5-turbo", 
                 suffixes=[".py", ".ipynb", ".md"]):                
        self.model_name = model_name
        self.cb = None
        self.codebase_local_path = codebase_local_path
        self.db_persist_directory = db_persist_directory
        self.query_tokens = query_tokens
        self.suffixes = suffixes
        
        if db_creation_mode == "local":
            db=self.db_from_local(codebase_path)
        elif db_creation_mode == "github":
            if "github.com" in codebase_path: # if it is a github
                db=self.db_from_github(codebase_path)
        elif db_creation_mode == "persist":
            db=self.db_from_persist()

        retriever = db.as_retriever(
            search_type="mmr",  # Also test "similarity"
            search_kwargs={"k": 8},
        )

        with get_openai_callback() as cb:
            # Set up LLM with memory
            llm = ChatOpenAI(model_name=self.model_name) 
            memory = ConversationSummaryMemory(llm=llm,memory_key="chat_history",return_messages=True)
            self.qa = ConversationalRetrievalChain.from_llm(
                llm, 
                retriever=retriever, 
                memory=memory,
                max_tokens_limit=self.query_tokens,
            )
        self.update_usage(cb)

        print("Repo loaded into langchain.")
        result = self.chat("Based on the readme, main program, or any other overview, describe this repo.")
        print(result)
        self.print_usage()

    def db_from_github(self, codebase_path):
        if os.path.isdir(self.codebase_local_path): shutil.rmtree(self.codebase_local_path)
        print(f"Cloning from {codebase_path} to {self.codebase_local_path}")
        repo = Repo.clone_from(codebase_path, to_path=self.codebase_local_path)
        
        self.from_local(self.codebase_local_path)

    def db_from_local(self, codebase_path):
        print(f"Loading from {codebase_path}")
        
        # Load
        loader = GenericLoader.from_filesystem(
            codebase_path,
            glob="**/*",
            suffixes=self.suffixes,
            parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
            show_progress=True,
        )
        documents = loader.load()
        # len(documents)

        python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, 
            chunk_size=3600, 
            chunk_overlap=200,
            )
        texts = python_splitter.split_documents(documents)

        chunk_size = 300
        n_chunks = np.ceil(len(texts)/chunk_size)

        print(f"Loaded {len(documents)} documents, \
              split into {len(texts)} texts, \
                grouped into {n_chunks} of size {chunk_size} chunks. \
                    \nLoading into chroma. This may take a while.")

        def _split_list(lst, chunk_size):
            for i in range(0, len(lst), chunk_size):
            # for i in range(len(lst)-2, len(lst), chunk_size):
                yield lst[i:i+chunk_size]

        # Set up chroma db and retriever

        # db = Chroma.from_documents(
        #     texts, 
        #     OpenAIEmbeddings(disallowed_special=(),),
        #     persist_directory=persist_directory
        #     )

        db = Chroma(
            persist_directory=self.persist_directory, 
            embedding_function=OpenAIEmbeddings(
                disallowed_special=(),
                )
            )
        
        # Max is 1,000,000 per minute
        for i, chunk in enumerate(_split_list(texts, chunk_size)):
            print(f"Loading documents into chroma_db {i}/{n_chunks}")
            start = timer()
            db.add_documents(documents=chunk)
            # while (timer() - start) < 60: # Time in seconds, e.g. 5.38091952400282
            if (timer() - start) < 60:
                print(f"Waiting, {60 - (timer() - start)}")
                time.sleep(60 + 1 - (timer() - start))
            else:
                # pass
                print(f"Done in {timer() - start}, over one minute, not waiting.")
                # print(chunk)

        return db

    def db_from_persist(self):
        db = Chroma(
                persist_directory=self.db_persist_directory, 
                embedding_function=OpenAIEmbeddings(
                        disallowed_special=(),)
            )
        return db

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


                