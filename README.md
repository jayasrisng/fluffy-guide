# fluffy-guide

real-time multimodal ai copilot for live learning

fluffy-guide is a lightweight research prototype that continuously listens to a lecture and helps you understand it in the moment without interrupting the speaker.

## what it does

- captures lecture audio in real time or demo mode  
- generates a rolling transcript  
- maintains a short evolving summary of recent context  
- lets you ask questions silently using text or voice  
- answers using only what was just said  

## how it works

fluffy-guide uses a two agent system

listener agent  
- captures audio  
- transcribes speech  
- maintains a rolling context buffer  
- updates a short summary  

guide agent  
- takes user questions  
- uses recent transcript and summary  
- returns concise grounded answers  

the lecturer is treated as not the user so interaction remains silent and separate

## setup

clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/fluffy-guide.git
cd fluffy-guide
````

create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

install dependencies

```bash
pip install -r requirements.txt
```

create a .env file and add your api key

```bash
OPENAI_API_KEY=your_api_key_here
```

run the app

```bash
python -m streamlit run app.py
```

## usage

demo mode

* enable demo mode
* start agents
* transcript and summary will simulate a lecture

live mode

* disable demo mode
* select microphone
* play lecture audio near your laptop

## asking questions

you can ask using text or voice input

examples

* summarize the last 30 seconds
* give me another example
* what does that mean simply

## notes

* demo mode is the most reliable for presentations
* live mode depends on microphone quality and environment
* separating lecture audio and user input improves accuracy

## research direction

this project explores

* real time understanding instead of post hoc learning
* continuous multimodal context
* silent human ai interaction
* multi agent system design

can be extended to xr gaze interaction and spatial interfaces

## stack

* python
* streamlit
* faster whisper
* openai api

## author

Jayasri Guthula