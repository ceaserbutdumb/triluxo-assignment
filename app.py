from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter

app = Flask(__name__)
api = Api(app)

url = "https://brainlox.com/courses/category/technical"
# url = "https://www.coursera.org/courses?query=artificial%20intelligence"
loader = UnstructuredURLLoader(urls=[url])
data = loader.load()


text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(data)


embeddings = HuggingFaceEmbeddings()


vector_store = FAISS.from_documents(documents, embeddings)


llm = HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-beta", temperature=0.5, max_new_tokens=512)


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vector_store.as_retriever(),
    memory=memory
)

class ChatbotResource(Resource):
    def post(self):
        data = request.get_json()
        user_message = data.get('message')
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        response = qa_chain({"question": user_message})
        
        return jsonify({"response": response['answer']})

api.add_resource(ChatbotResource, '/chat')

if __name__ == '__main__':
    app.run(debug=True)