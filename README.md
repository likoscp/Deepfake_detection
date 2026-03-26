Collaborators:

likoscp | diaszakir

Structure of the project:

    backend
        models
            deepfake_model 
            predict_fake
            saved_model
        utils
            normalize.py
            optimize.py
            resize.py
        features
        main.py
        requirements.txt
        .dockerfile

    frontend

    desktop

    ,gitignore
    README.md

Step 1: Drafts:

    create a model | in progress
    create a backend | in progress
    create a frontend/desktop | not started


 python3 -m venv .diploma
 linux source .diploma/bin/activate
  win .diploma/Scripts/activate
 pip install -r requirements.txt
    source ~/.diploma/diploma/bin/activate (это для моего линукса, у тебя мб будет в другом месте)
uvicorn main:app --reload --host 0.0.0.0 --port 8000 
.\ngrok  http 8000      