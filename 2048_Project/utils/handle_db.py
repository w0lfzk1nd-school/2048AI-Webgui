import mysql.connector
from mysql.connector import Error
import os

class DatabaseHandler:
    def __init__(self):
        self.host = os.getenv('MYSQL_HOST', 'localhost')
        self.user = os.getenv('MYSQL_USER', 'master')
        self.password = os.getenv('MYSQL_PASSWORD', 'yourpassword')
        self.database = os.getenv('MYSQL_DATABASE', '2048AI')
        self.connection = None
        self.connect()

    def connect(self):
        """Stellt eine Verbindung zur MySQL-Datenbank her."""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            if self.connection.is_connected():
                print("Connected to MySQL database")

        except Error as e:
            print("Error connecting to MySQL:", e)
            self.connection = None

    def close_connection(self):
        """Schließt die MySQL-Verbindung."""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("MySQL connection closed")

    def execute_query(self, query, values=None):
        """Führt einen SQL-Befehl aus."""
        try:
            cursor = self.connection.cursor()

            if values:
                cursor.execute(query, values)
            else:
                cursor.execute(query)

            if query.lower().startswith("select"):
                result = cursor.fetchall()
                return result

            self.connection.commit()
            print(f"Query executed successfully: {query}")

        except Error as e:
            print("Error executing query:", e)
            return None

        finally:
            if cursor is not None:
                cursor.close()

    def build_query(self, action, table, data=None, condition=None):
        """Baut einen SQL-Query basierend auf der Aktion."""
        query = ""
        if action == "insert":
            columns = ", ".join(data.keys())
            placeholders = ", ".join(["%s"] * len(data))
            query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
            values = tuple(data.values())
            return query, values

        elif action == "select":
            query = f"SELECT * FROM {table}"
            if condition and "ORDER BY" in condition:
                query += f" {condition}"
            elif condition:
                query += f" WHERE {condition}"
            return query, None

        elif action == "update":
            set_clause = ", ".join([f"{k} = %s" for k in data.keys()])
            query = f"UPDATE {table} SET {set_clause}"
            if condition:
                query += f" WHERE {condition}"
            values = tuple(data.values())
            return query, values

        elif action == "delete":
            query = f"DELETE FROM {table}"
            if condition:
                query += f" WHERE {condition}"
            return query, None

        return None, None

    def handle_db(self, action, table, data=None, condition=None):
        """Handhabt verschiedene Datenbankoperationen."""
        query, values = self.build_query(action, table, data, condition)
        if query:
            result = self.execute_query(query, values)
            return result
        else:
            print("Invalid query")
            return None

# Beispiel zur Nutzung der Klasse
if __name__ == "__main__":
    # MySQL Verbindungsinformationen
    MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
    MYSQL_USER = os.getenv('MYSQL_USER', 'master')
    MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', 'yourpassword')
    MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', '2048AI')

    db_handler = DatabaseHandler(MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE)

    # Beispiel für INSERT in die memory-Tabelle
    memory_data = {
        'step': 1,
        'board': '[1,2,3]',
        'action': 1,
        'new_board': '[4,5,6]',
        'reward': 10
    }
    db_handler.handle_db("insert", "memory", data=memory_data)

    # Beispiel für SELECT aus der memory-Tabelle
    result = db_handler.handle_db("select", "memory", condition="step=1")
    if result:
        for row in result:
            print(row)

    # Beispiel für UPDATE in der memory-Tabelle
    update_data = {'reward': 20}
    db_handler.handle_db("update", "memory", data=update_data, condition="step=1")

    # Beispiel für DELETE aus der memory-Tabelle
    db_handler.handle_db("delete", "memory", condition="step=1")

    # Verbindung schließen
    db_handler.close_connection()
