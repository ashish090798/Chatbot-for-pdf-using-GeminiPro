# PDF Answering Machine (using Google-genai)
This particular application lets the user upload any pdf document and ask questions based on that to get answers. I have leveraged Sentence Transformer model for text embedding and Google's genai pro model for retrieval and augmentation (RAG). I am providing a detailed description below regarding all the components used. 

Link to the application : https://vida-pdf-anwering-machine.streamlit.app/

## Loading the PDF :  
- I have used the PdfReader method from the PyPDF2 library to read and then essentially extract all the text. I wanted to use the pdf uploaded by the user directly and not from a stored location which is why I just grabbed all the text from it. 
- If you're planning to use a PDF stored in your local directory, you can use the Langchain's PyPDFLoader to load a stored pdf and then extract the data.

## Creating embedding for the data :
- The first step is to generate embedding on the text extracted from the pdf. nltk tokenizer can be used to tokenize the text and then it is passed to the 'all-MiniLM-L6-v2' Sentence Transformer model provided by Hugging Face.
- Once that is done I created the faiss index for those embedding and added the vectors to this index. This Facebook's library is basically used to perform efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size.

## Searching for top embedding from the data that match the User's query :
- I created a function that takes the faiss index, user's query, sentence data and top k(value of k decides the number of top matching documents that will come back after the search) value as parameters.
- The user query is then encoded similar to how the sentence data was using Sentence Transformer model and search is performed on the faiss index created based on the query to retrieve top k documents.
- Alternative : Instead of using embedding, the document can be split into chunks with some overlap. These splits can then be encoded as chromadb vector and can be directly sent to the LLM. But this is not an efficient technique as the similarity search has not been performed and the results might not be up to the mark for larger PDFs.

## Retrieval-Augmented Generation (RAG) :
- For this purpose I used Google's GenAI Pro model.
- Initially I created a simple template to let the LLM know what I want as an output. Giving a somewhat detailed explanation is the key for the LLM to provide valid answers in return. The prompt template variable takes the input and the template as input parameters.
- I then created an LLM chain that takes the LLM model and prompt template as input. Used SimpleSequentialChain to make sequential calls to the model. 

## Deployment on Streamlit
- Streamlit is an open-source Python framework used to create interactive apps.
- Mentioning text, uploading the PDF and providing the user query input were some of the components that I used in my application.
- The application can be deployed locally using "streamlit run app.py" command.
- To deploy the app on cloud, you can take the help of Streamlit community cloud platform and integrate your application directly from GitHub 
"# Chatbot-for-pdf-using-GeminiPro" 
