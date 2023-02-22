FROM python 

WORKDIR /proj2

COPY . /proj2

EXPOSE 8502

RUN pip install -r requirements.txt

CMD streamlit run app.py
