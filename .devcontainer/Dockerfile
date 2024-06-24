# Verwenden eines Basisimages mit Python
FROM python:3.11-bookworm

# Festlegen des Arbeitsverzeichnisses im Container
WORKDIR /workspace

# Kopieren der Dateien in Container-Volume
COPY .devcontainer/requirements.txt /workspace/requirements.txt
COPY .devcontainer/.env /workspace/.env
COPY .gitignore /workspace/.gitignore
COPY README.md /workspace/README.md
COPY 2048_Project /workspace/2048_Project
COPY .devcontainer /workspace/.devcontainer

# Installieren der notwendigen Libraries
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt-get update && apt-get upgrade -y

# Umgebungsvariablen setzen
ENV MYSQL_HOST=${MYSQL_HOST}
ENV MYSQL_USER=${MYSQL_USER}
ENV MYSQL_PASSWORD=${MYSQL_PASSWORD}
ENV MYSQL_DB=${MYSQL_DB}

# Offene Ports für Flask
EXPOSE 8282
EXPOSE 8283
EXPOSE 3307
EXPOSE 8085

# Startbefehl
CMD ["tail", "-f", "/dev/null"]
# CMD ["python", "/workspace/Project/webgui/app.py"]