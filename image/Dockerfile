FROM public.ecr.aws/lambda/python:3.10

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY src/.env .
COPY src/app.py .
COPY src/chatbot_clio.py .
COPY src/prompt_templates.yaml .
COPY src/supabase_utils.py .

CMD ["app.handler"]