# app/

## What Streamlit is

[Streamlit](https://streamlit.io/) turns a plain Python script into an interactive web
app: no HTML, JS, or frontend framework. You write normal Python, call functions like
`st.text_input(...)` or `st.button(...)` to render widgets, and Streamlit reruns your
script top to bottom whenever the user interacts with something. That is why
`streamlit_app.py` reads like a script rather than a web server.

## How to run it

```bash
streamlit run app/streamlit_app.py
```

Opens at `http://localhost:8501`. Stop it with `Ctrl+C`.

## Other demo formats

A Streamlit page suits a model with a small, typeable input. A recorded walkthrough or
a notebook with committed outputs may demonstrate a different kind of project better.
The requirement is that someone can see the thing work without reproducing your
environment first; the format is yours to choose.
