FROM mamounmohamed/super_base_image:v1
WORKDIR /app
COPY app.py app.py
COPY templates templates
COPY Emotion.h5 Emotion.h5
EXPOSE 5000


CMD ["python", "app.py"]
