# Verwenden eines Basisimages mit Python
FROM python:3.9-slim

# Festlegen des Arbeitsverzeichnisses im Container
WORKDIR /workspace

# Kopieren der requirements-Datei und Installieren der Abhängigkeiten
COPY requirements.txt /workspace/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Offene Ports für Flask
EXPOSE 8282
EXPOSE 8283
