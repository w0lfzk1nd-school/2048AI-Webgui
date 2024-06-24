import mysql.connector
from mysql.connector import Error

# MySQL Verbindungsinformationen
MYSQL_USER = 'master'
MYSQL_PASSWORD = 'Your-DB-Password'
MYSQL_DATABASE = '2048AI'

# Testdaten für die Tabellen
MEMORY_DATA = "INSERT INTO memory (step, board, action, new_board, reward) VALUES (%s, %s, %s, %s, %s)"
TRAININGS_DATA = "INSERT INTO trainings (`key`, start, steps, games, highscore, highblock) VALUES (%s, NOW(), %s, %s, %s, %s)"
WEBGUI_DATA = "INSERT INTO webgui (steps_played, highscore, highscore_txt, highblock, highblock_txt) VALUES (%s, %s, %s, %s, %s)"
WEB_LEADERBOARD_DATA = "INSERT INTO web_leaderboard (time, uname, score, block) VALUES (NOW(), %s, %s, %s)"

# Testdaten für Einfügungen
memory_data_values = (1, '[1,2,3]', 1, '[4,5,6]', 10)
trainings_data_values = ('key1', 100, 5, 200, 15)
webgui_data_values = (500, 250, 'Highscore Text', 20, 'Highblock Text')
web_leaderboard_data_values = ('user1', 300, 25)

# Funktion zum Verbinden zur MySQL-Datenbank
def connect():
    try:
        connection = mysql.connector.connect(
            host='mysql',
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE
        )
        if connection.is_connected():
            print("Connected to MySQL database")
        return connection

    except Error as e:
        print("Error connecting to MySQL:", e)
        return None

# Funktion zum Ausführen von SQL-Befehlen
def execute_query(connection, query, values=None):
    try:
        cursor = connection.cursor()

        if values:
            cursor.execute(query, values)
        else:
            cursor.execute(query)

        # Falls SELECT-Befehl, Ergebnisse abrufen und anzeigen
        if query.lower().startswith("select"):
            result = cursor.fetchall()
            for row in result:
                print(row)

        connection.commit()
        print("Query executed successfully")

    except Error as e:
        print("Error executing query:", e)

    finally:
        if 'cursor' in locals() and cursor is not None:
            cursor.close()

# Hauptfunktion zum Ausführen der Operationen
def main():
    connection = connect()

    if connection:
        try:
            # Testdaten einfügen
            execute_query(connection, MEMORY_DATA, memory_data_values)
            execute_query(connection, TRAININGS_DATA, trainings_data_values)
            execute_query(connection, WEBGUI_DATA, webgui_data_values)
            execute_query(connection, WEB_LEADERBOARD_DATA, web_leaderboard_data_values)

            # Daten auslesen
            print("Memory table:")
            execute_query(connection, "SELECT * FROM memory;")
            print("")

            print("Trainings table:")
            execute_query(connection, "SELECT * FROM trainings;")
            print("")

            print("Webgui table:")
            execute_query(connection, "SELECT * FROM webgui;")
            print("")

            print("Web leaderboard table:")
            execute_query(connection, "SELECT * FROM web_leaderboard;")
            print("")

            # Daten aktualisieren (Beispiel: Update highscore in trainings)
            print("Updating highscore in trainings table...")
            execute_query(connection, "UPDATE trainings SET highscore = 250 WHERE id = 1;")

            # Daten erneut auslesen
            print("Updated trainings table:")
            execute_query(connection, "SELECT * FROM trainings;")
            print("")

            # Daten löschen (Beispiel: Löschen des Eintrags in web_leaderboard)
            print("Deleting entry from web leaderboard table...")
            execute_query(connection, "DELETE FROM web_leaderboard WHERE id = 1;")

            # Daten erneut auslesen
            print("Updated web leaderboard table:")
            execute_query(connection, "SELECT * FROM web_leaderboard;")
            print("")

        finally:
            if connection.is_connected():
                connection.close()
                print("MySQL connection closed")

if __name__ == "__main__":
    main()