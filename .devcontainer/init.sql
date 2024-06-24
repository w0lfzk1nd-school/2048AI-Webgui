-- init.sql
CREATE DATABASE IF NOT EXISTS 2048AI;

USE 2048AI;

-- Benutzer 'master' erhält volle Rechte auf '2048AI' DB
GRANT ALL PRIVILEGES ON 2048AI.* TO 'master'@'%';

-- Tabelle 'memory'
CREATE TABLE IF NOT EXISTS memory (
    step INT,
    board JSON,
    action INT,
    new_board JSON,
    reward INT
);

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

-- Tabelle 'webgui'
CREATE TABLE IF NOT EXISTS webgui (
    steps_played INT,
    highscore INT,
    highscore_txt VARCHAR(100),
    highblock INT,
    highblock_txt VARCHAR(100)
);

-- Tabelle 'web_leaderboard'
CREATE TABLE IF NOT EXISTS web_leaderboard (
    id INT AUTO_INCREMENT PRIMARY KEY,
    time TIMESTAMP,
    uname VARCHAR(15),
    score INT,
    block INT
);
