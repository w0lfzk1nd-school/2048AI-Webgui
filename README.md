# 2048AI Dev Container

**Automatic DevContainer**

[![Open in Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?repo=w0lfzk1nd-school/2048AI-Webgui)
---
Dies ist eine Entwicklungsumgebung für das `2048AI` Projekt. Diese Umgebung verwendet einen Dev-Container, der in Visual **Studio Code (VSCode)** oder **Github Codespaces** ausgeführt wird und alle notwendigen Abhängigkeiten sowie Konfigurationen enthält, um eine reibungslose Entwicklung zu ermöglichen.

## Dokumentation Inhalt Dev-Container:
- DevContainer_README.md --> **vorhanden**
- Automatisierte Installation Abhängikeiten --> **requirements.txt vorhanden**
- Nützliche Extensions --> **prüfen**
- Debugging-Unterstützung --> ?
- Datenbankintegration --> ? (Für highscores bzw runtime data maybe?)
- Produktionsbereite Container --> **vorhanden**
- Sicheres Handling sensibler Daten --> ?
- Demodaten --> ?
- Alternativlösung --> ?
- One-Click Setup --> **Codespace vorhanden**
- Pull Request --> **Eigenes Projekt**


## Inhaltsverzeichnis

- [2048AI Dev Container](#2048ai-dev-container)
  - [](#)
  - [Dokumentation Inhalt Dev-Container:](#dokumentation-inhalt-dev-container)
  - [Inhaltsverzeichnis](#inhaltsverzeichnis)
  - [Voraussetzungen](#voraussetzungen)
  - [Ablauf der Einrichtung](#ablauf-der-einrichtung)
  - [Verzeichnisstruktur](#verzeichnisstruktur)
  - [Dev-Container Konfiguration](#dev-container-konfiguration)
  - [Nutzung des Dev-Containers](#nutzung-des-dev-containers)
  - [Ports](#ports)
  - [VSCode Erweiterungen](#vscode-erweiterungen)
  - [Bekannte Probleme](#bekannte-probleme)
- [**2048 AI Projekt**](#2048-ai-projekt)
  - [**Struktur**](#struktur)
  - [**Benutzung**](#benutzung)
      - [**WebGui**](#webgui)
      - [**Modell-Training und Dataset-Tools**](#modell-training-und-dataset-tools)
      - [**Datasets Struktur**](#datasets-struktur)
    - [**WebGui**](#webgui-1)
    - [**Keras Modell Training**](#keras-modell-training)
- [Zusammenfassung](#zusammenfassung)
  

## Voraussetzungen

Bevor du beginnst, stelle sicher, dass die folgenden Programme auf deinem Rechner installiert sind:

- [Docker](https://www.docker.com/)
- [Visual Studio Code (VSCode)](https://code.visualstudio.com/)
- [Dev Containers Extension für VSCode](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
- (optional) [Git](https://git-scm.com/downloads)

## Ablauf der Einrichtung

*Dies muss in dieser Repo nicht nochmals wiederholt werden*

1. **Clone das Repository:**

    ```bash
    git clone https://github.com/w0lfzk1nd-school/2048AI-Webgui.git
    cd 2048AI-Webgui
    ```

2. **Erstelle das Verzeichnis `.devcontainer`:**

    ```bash
    mkdir .devcontainer
    ```

3. **Erstelle die Datei `.devcontainer/devcontainer.json` mit folgendem Inhalt:**

    ```json
    {
        "name": "2048AI Dev Container",
        "dockerComposeFile": "docker-compose.yml",
        "service": "app",
        "workspaceFolder": "/workspace",
        "forwardPorts": [8282, 8283, 8085],
        "postCreateCommand": "pip install --upgrade pip && pip install -r /workspace/.devcontainer/requirements.txt",
        "customizations": {
            "vscode": {
                "extensions": [
                  "ms-python.python", // Python
                  "yzhang.markdown-all-in-one", //Markdown
                  "ms-toolsai.jupyter-renderers", //Jupyter Notebook
                  "ms-python.black-formatter" // Python Formatter
                ],
                "settings": {
                    "python.pythonPath": "/usr/local/bin/python"
                }
            }
        }
    }

    ```

4. **Erstelle die Datei `requirements.txt` im `.devcontainer`:**

    ```plaintext
    numpy
    tensorflow
    flask
    questionary
    matplotlib
    ```

## Verzeichnisstruktur

Stelle sicher, dass dein Projektverzeichnis wie folgt aussieht:

```
2048AI/
├── .devcontainer/
│  ├── devcontainer.json
│  ├── Dockerfile
│  ├── docker-compose.json
│  ├── .env (Beispiel / Vorlage)
│  ├── env-schema
│  ├── requirements.txt
├── 2048_Project/
│  ├── (Projektdateien)
└── (Repository Dateien)
```


## Dev-Container Konfiguration

Die Datei `.devcontainer/devcontainer.json` definiert die Konfiguration des Dev-Containers:

- **Name**: Alle haben das Recht auf einen Namen.
- **DockerComposeFile**: Bezieht die notwendigen Dockerdaten aus der `Docker-Compose` Datei.
- **Service**: Setzt den Namen für den Service des Devcontainers innerhalb der Compose.
- **Arbeitsverzeichnis**: Setzt das Arbeitsverzeichnis im Container auf `/workspace`.
- **Portweiterleitung**: Leitet die Ports 8282, 8283 und 8085 vom Container an den Host weiter.
- **Post Create Command**: Installiert die Python-Abhängigkeiten nach dem Erstellen des Containers.
- **VSCode-Erweiterungen**: Installiert die Python- und weitere Erweiterungen.
- **Workspace Mount**: Bindet das Host-Verzeichnis `2048AI` in den Container unter `/workspace`.

## Nutzung des Dev-Containers

**In Visual Studio Code**

1. **Öffne VSCode und lade das Projektverzeichnis `2048AI`.**
2. **Klicke auf das grüne Symbol unten links in der Statusleiste (Open a Remote Window).**
3. **Wähle `Reopen in Container`.**

**In Github Codespaces**

1. **Klicke den `Open in Codespaces` Button ganz oben.


VSCode startet nun den Dev-Container und installiert die notwendigen Abhängigkeiten gemäß der Datei `requirements.txt`.

Ob im Browser oder als Anwendung :)

## Ports

Die folgenden Ports werden für die Entwicklung genutzt und vom Container an den Host weitergeleitet:

- **8282**: Wird auf Port 8282 des Hosts gemappt. *(Projekt Hauptport des WebGUI)*
- **8283**: Wird auf Port 8283 des Hosts gemappt. *(Projekt Entwicklungsport des WebGUI)*
- **8085**: Wird auf Port 8085 des Hosts gemappt. *(Datenbankanbindung via pmpMyAdmin)*
  
Die folgenden Ports werden für die Entwicklung genutzt, jedoch nicht an den Host weitergeleitet:

- **3306**: Mysql Docker.

## VSCode Erweiterungen

Die folgenden Erweiterungen werden im Dev-Container installiert:

- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [Markdown All in One](https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one)
- [Jupyter Notebook](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter-renderers)
- [Python Black Formatter](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter)

## Bekannte Probleme

Falls der Fehler "workspace does not exist" auftritt, überprüfe die folgenden Punkte:

- **Pfadüberprüfung**: Stelle sicher, dass der Pfad zum Projektverzeichnis korrekt ist. Verwende absolute Pfade, um mögliche Fehler zu vermeiden.
- **Berechtigungen**: Stelle sicher, dass du die notwendigen Berechtigungen hast, um auf das Verzeichnis zuzugreifen und es zu mounten.

Bei weiteren Problemen oder Fragen, essen sie die Packungsbeilage und sagen sie ihrem Arzt er sei Apotheker.

---

Viel Spaß beim Entwickeln mit deinem 2048AI Dev-Container!

---

# **2048 AI Projekt**

Vor ein paar Jahren habe ich das simple Handyspiel entdeckt und war von der Komplexität begeistert. Nun, ein paar Jahre später, hab ich mich mehr mit AI oder generell neuronalen Netzwerken beschäftigt und beherrschte mehrere Programmiersprachen, darunter Python.

Dieses Projekt umfasst folgendes:
- DevContainer.
- WebGui für Trainingdata erfassung.
- Simples Konsoleninterface um ein Keras Modell mit *Tensorflow* zu trainieren und die gesammelten Daten in ein brauchbares Dataset umformatieren.

**Das Projekt ist noch in der Entwicklungsphase!**

## **Struktur**

- [2048 AI Projekt](#2048ai-ai-projekt)
  - [Struktur](#struktur)
  - [Benutzung](#benutzung)
  - [WebGui Data Collecter](#webgui)
  - [Keras Modell Trainer](#keras-modell-training)
---
## **Benutzung**

#### **WebGui**

Das WebGui startet man mit:
```
python3 webgui/app.py
```

#### **Modell-Training und Dataset-Tools**

Das Konsolenmenü für das Training und vorbereitung/verarbeitung der gesammelten Daten/Datasets startet man mit:
```
python3 main.py
```

#### **Datasets Struktur**

*[ [board, richtung], [board, richtung] ]*

```
  [
    [
      [
        [4, 8, 64, 512],
        [0, 4, 16, 32],
        [0, 0, 2, 8],
        [0, 2, 0, 4] // Board
      ],
      3 // Richtung
    ], // Nächster Zug
  ]
```

- `datasets/dataset_0_*` = 280'000 Spielzüge mit Highscore **2048**.
  - 100'000 menschliche Züge.
  - 70'000 Expert AI Züge.
  - 50'000 ChatGPT / Ollama Züge.
  - 60'000 Zufällige Spielfelder.
- `datasets/dataset_0_280k_augmented` = 1.7 Millionen Spielzüge *wie oben* nur mit Augemntation:
  - *Spiegle vertikal*
  - *Spiegle horizontal*
  - *Drehe 3x um 90°*
- `datasets/dataset_1_*` = Gleiche Spielzüge wie oben mit einem [**MCTS Algorithmus**](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search#:~:text=In%20computer%20science%2C%20Monte%20Carlo,to%20solve%20the%20game%20tree.) analysiert um den bestmöglichen Zug zu ermitteln.

---
### **WebGui**

![WebGui](2048ai_webgui.PNG "WebGui")

Das WebGui wurde entwickelt, um Spielzüge und Spielzustände von echten Menschen zu sammeln, welche das Spiel spielen

**Additional:** Es gibt die Option, welche einen **[Monte Carlo Algorithm](https://de.wikipedia.org/wiki/Monte-Carlo-Simulation)** verwendet um für alle 4 Richtungen ca 10'000 Züge *(400 pro Zug selbst)* schätzt und die gewonnenen Punkte zurückgibt. Anhand dessen lässt sich festlegen welche Richtung für das jetztige Spielfeld ideal ist.

---

### **Keras Modell Training**

Das Training des **[Keras AI Modell](https://de.wikipedia.org/wiki/Keras)** läuft in 4 Schritten ab:

- **Dataset sammeln**
  - Sammle Daten im Internet, lasse Freunde spielen oder schreibe Programme.

- **Dataset vorbereiten**
  - Die gesammelten Daten müssen nun noch vorbereitet werden, dies beinhält:
    - *Duplikate aussortieren* **10min**
    - *Augmentieren (Spiegeln, drehen)* **20min**
    - *Mithilfe des **Monte Carlo Algorithmus** für jedes Board den besten Zug ermitteln lassen.* **6 Stunden**

- **Modell mit Dataset trainieren**
  - Nun kann das Modell mit den gesammelten und vorbereiteten Daten trainiert werden. In diesem Prozess werden alle Spielzustände, mit dem vom Algorithmus errechneten besten Zug, an das Modell in zufälliger Reihenfolge gezeigt.

- **Modell selbst trainieren lassen und verfeinern.**
  - In diesem Schritt, wird das Modell *24 bis 48 Stunden* alleine spielen und bei Problensituationen (wie einem vollen Board) wird dieses mit genaueren Spieldaten zu diesem Problem nachtrainiert.

---

# Zusammenfassung

Im grossen und ganzen wurde das Potential von **AI** im Spiel *2048* bereits ausgeschöpft. Es gibt bereits Modelle, welche das Spiel bis zum letzt möglichen Zug gespielt und somit den bestmöglichen Punktestand erreicht haben.

Jedoch finde ich es für mich selbst eine sehr tolle und interessante Aufgabe und Erfahrung, sich mit solch einem Thema auseinander zu setzen.

Ebenso geht an dieser Stelle ein **riesen** Dankeschön an meine Ookami Chatbot Community, welche innerhalt von 3 Tagen fast 100'000 Spielzüge gespielt haben.

*Dieses Projekt ist noch nicht abgeschlossen und wird daher in der Zukunft geupdatet werden.*