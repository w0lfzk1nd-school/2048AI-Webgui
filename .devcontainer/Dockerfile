# Verwenden eines Basisimages mit Python
FROM python:3.11-bookworm

# Festlegen des Arbeitsverzeichnisses im Container
WORKDIR /workspace

# Kopieren der requirements-Datei und Installieren der Abhängigkeiten
COPY requirements.txt /workspace/requirements.txt
COPY .env /workspace/.env
#COPY ../* /workspace/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

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
# tail -f /dev/null
CMD ["tail", "-f", "/dev/null"]
# CMD ["python", "/workspace/webgui/app.py"]