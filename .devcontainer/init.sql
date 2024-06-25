-- init.sql
CREATE DATABASE IF NOT EXISTS 2048AI;

USE 2048AI;

-- Benutzer 'master' erhält volle Rechte auf '2048AI' DB
GRANT ALL PRIVILEGES ON 2048AI.* TO 'master'@'%';
FLUSH PRIVILEGES;

-- Tabelle 'memory'
CREATE TABLE IF NOT EXISTS memory (
    step INT,
    board JSON,
    action INT,
    new_board JSON,
    reward INT,
    done BOOLEAN
);

-- Insert Defaultdata --
INSERT INTO memory (step, board, action, new_board, reward, done) 
VALUES (0, '[["4", "8", "64", "512"], ["0", "4", "16", "32"], ["0", "0", "2", "8"], ["0", "2", "0", "4"]]', 2, '[["0", "0", "0", "512"], ["0", "8", "64", "32"], ["0", "4", "16", "8"], ["4", "2", "2", "4"]]', 600, 0);

-- Tabelle 'trainings'
CREATE TABLE IF NOT EXISTS trainings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    `key` VARCHAR(255),
    start TIMESTAMP,
    steps INT,
    games INT,
    highscore INT,
    highblock INT
);

-- Insert Defaultdata --
INSERT INTO trainings (`key`, start, steps, games, highscore, highblock) 
VALUES ("ABCDEF", NOW(), 0, 0, 0, 0);

-- Tabelle 'webgui'
CREATE TABLE IF NOT EXISTS webgui (
    steps_played INT,
    highscore INT,
    highscore_txt VARCHAR(100),
    highblock INT,
    highblock_txt VARCHAR(100)
);

-- Insert Defaultdata --
INSERT INTO webgui (steps_played, highscore, highscore_txt, highblock, highblock_txt) 
VALUES (0, 0, "[ 0 ] @00:00:00 01/01/1970", 0, "[ 0 ] @00:00:00 01/01/1970");

-- Tabelle 'web_leaderboard'
CREATE TABLE IF NOT EXISTS web_leaderboard (
    id INT AUTO_INCREMENT PRIMARY KEY,
    time TIMESTAMP,
    uname VARCHAR(15),
    score INT,
    block INT
);

-- Insert Testdata --
INSERT INTO web_leaderboard (time, uname, score, block) 
VALUES (NOW(), "Puppet", 69, 4);
