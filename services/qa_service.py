from transformers import pipeline
from config import Config

qa_pipeline = None

def init_qa_service():
    global qa_pipeline
    try:
        ###### [ VERSION 1 ] ######
        # qa_pipeline = pipeline("question-answering", model="deepset/tinyroberta-squad2", token=Config.HUGGINGFACE_HUB_TOKEN)

        ###### [ VERSION 2 ] ######
        qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small", token=Config.HUGGINGFACE_HUB_TOKEN)
        print("QA service initialized successfully")
    except Exception as e:
        print(f"Failed to initialize QA service: {str(e)}")
        raise
    return qa_pipeline


def ask_question(context, question):
    try:
        if qa_pipeline is None:
            raise RuntimeError("QA service not initialized")
        

        ###### [ VERSION 1 ] ######
        # result = qa_pipeline(question=question, context=context)
        # return result

        ###### [ VERSION 2 ] ######
        prompt = f"Please answer the following question.\nContext: {context}\nQuestion: {question}"
        result = qa_pipeline(prompt, max_length=100, do_sample=False)
        return result[0]["generated_text"]
    except Exception as e:
        print(f"Error during question answering: {str(e)}")
        return None
