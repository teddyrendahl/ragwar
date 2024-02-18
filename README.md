# ragwar
Retrieval Augmented Generation Chatbot for Warhammer40k Rules

https://github.com/teddyrendahl/ragwar/assets/25753048/5862045c-de43-422d-a99c-7f0dda9489ae

## Installation
```bash
$ pip install -r requirements.txt
```

## How to Use
This project requires a valid `OPENAI_API_KEY` to do the embedding and response generation. Simply create
a `.env` file in this directory with your key as shown in the `.env.example`.


## Creating the Database
This project uses Chroma to create a local vector database of the embeddings. It's assumed this is done
before running `app.py`
```bash
$ python create_db.py
```
Afterwards you should see the generated files in a `chroma` folder

## Running the Application
To run the application use `streamlit`:
```bash
$ streamlit run app.py
```


