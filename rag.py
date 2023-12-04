#import required dependencies
from langchain.prompts import PromptTemplate
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import chainlit as cl
from langchain.chains import RetrievalQA

# шаблон, който ще използваме при генерирането на потребителския промпт
template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

DB_PATH = "vectorstores/db/"


def get_qa_chain(): 
 
 # дефинираме езиковия модел - в случая използваме Orca2 на Microsoft
 llm = Ollama(
 model="orca2",
 verbose=True,
 callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
 )
 
 # дефинираме базата данни за векторите  
 vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=GPT4AllEmbeddings())
 
 # конфигурираме веригата (chain) през която протича заявката на потребителя.
 qa = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    return_source_documents=True,)
 return qa 

# когато chainlit клиента стартира
@cl.on_chat_start
async def start():
 chain=get_qa_chain()
 msg=cl.Message(content="Starting the chatbot...")
 await msg.send()
 msg.content = "Welcome to the Information Systems chat bot. What is your question?"
 await msg.update()
 cl.user_session.set("chain",chain)

# когато изпратим съобщение в chainlit клиента 
@cl.on_message
async def main(message):
 
 chain=cl.user_session.get("chain")
 
 cb = cl.AsyncLangchainCallbackHandler(
    stream_final_answer=True,
    answer_prefix_tokens=["FINAL", "ANSWER"]
 )
 
 cb.answer_reached = True
 
 response = await chain.acall(message.content, callbacks=[cb])
 # отговорът от Orca2
 answer = response["result"]
 #answer = answer.replace(".",".\n")
 
 # ако chroma e намерила източници, те ще бъдат включени в този списък
 sources = response["source_documents"]

 if sources:
  answer+=f"\nSources: "+str(str(sources))
 else:
  answer+=f"\nNo Sources found"

 await cl.Message(content = answer).send() 