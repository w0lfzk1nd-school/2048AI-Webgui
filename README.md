# 2048AI Dev Container

Dies ist eine Entwicklungsumgebung für das `2048AI` Projekt. Diese Umgebung verwendet einen Dev-Container, der in Visual Studio Code (VSCode) ausgeführt wird und alle notwendigen Abhängigkeiten sowie Konfigurationen enthält, um eine reibungslose Entwicklung zu ermöglichen.

## Inhaltsverzeichnis

- [2048AI Dev Container](#2048ai-dev-container)
  - [Inhaltsverzeichnis](#inhaltsverzeichnis)
  - [Voraussetzungen](#voraussetzungen)
  - [Einrichtung](#einrichtung)
  - [Verzeichnisstruktur](#verzeichnisstruktur)
  - [Dev-Container Konfiguration](#dev-container-konfiguration)
  - [Nutzung des Dev-Containers](#nutzung-des-dev-containers)
  - [Ports](#ports)
  - [VSCode Erweiterungen](#vscode-erweiterungen)
  - [Bekannte Probleme](#bekannte-probleme)

## Voraussetzungen

Bevor du beginnst, stelle sicher, dass die folgenden Programme auf deinem Rechner installiert sind:

- [Docker](https://www.docker.com/)
- [Visual Studio Code (VSCode)](https://code.visualstudio.com/)
- [Dev Containers Extension für VSCode](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

## Einrichtung

1. **Clone das Repository:**

    ```bash
    git clone https://github.com/w0lfzk1nd-school/2048AI-Webgui.git
    cd 2048AI-Webgui
    ```

2. **Erstelle das Verzeichnis `.devcontainer` und füge die notwendigen Dateien hinzu:**

    ```bash
    mkdir .devcontainer
    ```

3. **Erstelle die Datei `.devcontainer/devcontainer.json` mit folgendem Inhalt:**

    ```json
    {
        "name": "2048AI Dev Container",
        "image": "mcr.microsoft.com/devcontainers/python:1-3.11-bookworm",
        "workspaceFolder": "/workspace",
        "forwardPorts": [8282, 8283],
        "postCreateCommand": "pip install --upgrade pip && pip install -r /workspace/requirements.txt",
        "customizations": {
            "vscode": {
                "extensions": [
                    "ms-python.python",
                    "yzhang.markdown-all-in-one"
                ],
                "settings": {
                    "python.pythonPath": "/usr/local/bin/python"
                }
            }
        },
        "workspaceMount": "source=${localWorkspaceFolder},target=/workspace,type=bind"
    }
    ```

4. **Erstelle die Datei `requirements.txt` im Hauptverzeichnis des Projekts:**

    ```plaintext
    numpy
    tensorflow
    flask
    ```

## Verzeichnisstruktur

Stelle sicher, dass dein Projektverzeichnis wie folgt aussieht:

2048AI/
├── .devcontainer/
│ ├── devcontainer.json
├── requirements.txt
└── (andere Projektdateien)


## Dev-Container Konfiguration

Die Datei `.devcontainer/devcontainer.json` definiert die Konfiguration des Dev-Containers:

- **Image**: Verwendet ein Python-Image (Python 3.11) basierend auf Debian Bookworm.
- **Arbeitsverzeichnis**: Setzt das Arbeitsverzeichnis im Container auf `/workspace`.
- **Portweiterleitung**: Leitet die Ports 8282 und 8283 vom Container an den Host weiter.
- **Post Create Command**: Installiert die Python-Abhängigkeiten nach dem Erstellen des Containers.
- **VSCode-Erweiterungen**: Installiert die Python- und Markdown-All-in-One-Erweiterungen.
- **Workspace Mount**: Bindet das Host-Verzeichnis `2048AI` in den Container unter `/workspace`.

## Nutzung des Dev-Containers

1. **Öffne VSCode und lade das Projektverzeichnis `2048AI`.**
2. **Klicke auf das grüne Symbol unten links in der Statusleiste (Open a Remote Window).**
3. **Wähle `Reopen in Container`.**

VSCode startet nun den Dev-Container und installiert die notwendigen Abhängigkeiten gemäß der Datei `requirements.txt`.

## Ports

Die folgenden Ports werden für die Entwicklung genutzt und vom Container an den Host weitergeleitet:

- **8282**: Wird auf Port 8282 des Hosts gemappt.
- **8283**: Wird auf Port 8283 des Hosts gemappt.

## VSCode Erweiterungen

Die folgenden Erweiterungen werden im Dev-Container installiert:

- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [Markdown All in One](https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one)

## Bekannte Probleme

Falls der Fehler "workspace does not exist" auftritt, überprüfe die folgenden Punkte:

- **Pfadüberprüfung**: Stelle sicher, dass der Pfad zum Projektverzeichnis korrekt ist. Verwende absolute Pfade, um mögliche Fehler zu vermeiden.
- **Berechtigungen**: Stelle sicher, dass du die notwendigen Berechtigungen hast, um auf das Verzeichnis zuzugreifen und es zu mounten.

Bei weiteren Problemen oder Fragen, kontaktiere bitte den Projektmaintainer.

---

Viel Spaß beim Entwickeln mit deinem 2048AI Dev-Container!
