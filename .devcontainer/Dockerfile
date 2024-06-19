# Verwenden eines Basisimages mit Python
FROM python:3.11-bookworm

# Festlegen des Arbeitsverzeichnisses im Container
WORKDIR /workspace

# Kopieren der requirements-Datei und Installieren der Abhängigkeiten
COPY requirements.txt /workspace/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Kopieren des Ordners "2048AI-Webgui" in das Arbeitsverzeichnis im Container
#COPY ../2048AI-Webgui /workspace/2048AI-Webgui

# Offene Ports für Flask
EXPOSE 8282
EXPOSE 8283
EXPOSE 3307

# Startbefehl für die App (anpassen, falls notwendig)
#CMD ["python", "/workspace/2048AI-Webgui/webgui/app.py"]