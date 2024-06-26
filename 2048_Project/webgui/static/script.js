let isPredicting = false;
let win_confirm = false;
let game_over = false;
let currentHighscoreSet = false;
let currentUsername = '';

function genPredictionTxt() {
    const texts = ["Writing down pi", "Brewing coffee", "Sorting braincells", "Predicting", "Reading future", "Processing", "Working", "Running", "Simulating", "Doing magic", "Redefining sience", "Doing math", "Adding a lot of numbers", "Shifting tiles", "Calculating moves", "Multiplying tiles", "Crushing numbers",
        "Merging blocks", "Summing digits", "Rewriting equations", "Swapping tiles", "Exploting realitiy", "Abusing reallife bug", "Abusing reallife exploit",
        "Combining blocks", "Solving puzzles", "Number crunching", "Tile shifting", "Creating Black Hole", "Generating next move", "Doing the work", "Let me solve this", "Analyzing situation", "Asking McGyver",
        "Chasing 2048", "Analyzing patterns", "Planning moves", "Strategizing", "Summoning None", "Forging bytes", "Calling Chuck Norris", "Calling FBI", "Questioning existence", "Breaking laws of physic", "Bending lightbeams", "Consuming nuclear waste", "Reading the matrix", "Reading the matrix backwards", "Smashing calculator", "Solving the unsolvable", "Calling the devil", "Playing BlackJack", "Proofing the earths shape is banana", "Calling a lost spirit", "Thinking hard", "Intense thinking", "Building timemaschine", "Using quantom mechanics", "Asking the oracle", "Reviving Alan Turing",
        "Fusing tiles", "Tile calculus", "Powering up", "Asking Einstein", "Redefining Gravity", , "Breaking limits", "Exploring possibilities", "Unleashing potential", "Executing algorithms", "Synthesizing data", "Formulating strategies", "Engaging in logic", "Rearranging integers", "Optimizing moves", "Simulating scenarios", "Assessing options", "Deploying mathematic", "Crunching equations", "Maximizing efficiency", "Harnessing power", "Amplifying results", "Navigating complexity", "Enhancing calculations", "Fine-tuning logic", "Refining algorithms", "Mastering tiles", "Breaking through", "Evolving strategies", "Unlocking secrets", "Mapping outcomes", "Pursuing perfection", "Innovating solutions", "Bridging gaps", "Balancing probabilities", "Analyzing outcomes", "Computing solutions", "Revolutionizing methods", "Forecasting results", "Conquering challenges", "Decoding mysteries", "Firing up enigma", "Bribing reality", "Adding randomness to reality", "Playing the game"]
    const randomIndex = Math.floor(Math.random() * texts.length);
    return texts[randomIndex];
}

function format_num(num) {
    return num.toLocaleString('de-DE').replace(/\./g, "'");
};

function showConfirmation() {
    const userConfirmed = confirm("Are you sure, you want to reset the gamestate?");

    if (userConfirmed) {
        return true;
    } else {
        alert("Reset Game cancelled.");
        return false;
    };
};

function getUserInput() {
    let userInput = prompt("Enter your Username:");
    if (userInput !== null && userInput.length <= 20) {
        document.getElementById("user_input").value = userInput;
        document.getElementById("inputForm").submit();
    }
}

function loadLeaderboard() {
    fetch('/api/get_leaderboard')
        .then(response => response.json())
        .then(data => {
            const leaderboardList = document.getElementById('leaderboard-list');
            leaderboardList.innerHTML = '';
            data.forEach(entry => {
                const listItem = document.createElement('li');
                const formattedTime = new Date(entry[1]).toLocaleString();
                listItem.textContent = `${formattedTime} | ${entry[2]} | ${format_num(parseInt(entry[3]))} | ${format_num(parseInt(entry[4]))}`;
                leaderboardList.appendChild(listItem);
            });
        })
        .catch(error => {
            console.error('Error loading leaderboard:', error);
        });
}


function loadMovesCount() {
    fetch('/api/moves')
        .then(response => response.json())
        .then(data => {
            const movesCount = document.getElementById('moves-count');
            movesCount.textContent = `Total Moves Learned: ${format_num(data)}`;
        });
}

function loadBoard() {
    fetch('/api/board')
        .then(response => response.json())
        .then(data => {
            const boardDiv = document.getElementById('board');
            boardDiv.innerHTML = '';
            let flattenedData = data.flat();
            let cellIndex = 0;

            for (let i = 0; i < 4; i++) {
                const rowDiv = document.createElement('div');
                rowDiv.classList.add('row');

                for (let j = 0; j < 4; j++) {
                    const cellDiv = document.createElement('div');
                    cellDiv.classList.add('cell');
                    cellDiv.textContent = flattenedData[cellIndex] !== 0 ? flattenedData[cellIndex] : '';
                    rowDiv.appendChild(cellDiv);
                    cellIndex++;
                }
                boardDiv.appendChild(rowDiv);
            }
        });
}

function loadScore() {
    fetch('/api/score')
        .then(response => response.json())
        .then(data => {
            const playerScore = document.getElementById('player-score');
            const globalHighscore = document.getElementById('glob-highscore');
            const globalBestblock = document.getElementById('glob-bestblock');
            playerScore.textContent = `Your Player Points: ${format_num(parseInt(data.score))} | Best-Block: [ ${format_num(parseInt(data.block))} ]`;
            globalHighscore.textContent = `Global Highscore: ${data.highscore}`;
            globalBestblock.textContent = `Global BestBlock: ${data.best_block}`;
        });
}

function predictMove() {
    if (isPredicting) {
        console.log("A prediction request is already in progress.");
        return;
    }

    if (game_over) {
        alert("Please reset the game first!")
        return
    }

    isPredicting = true;

    const predictDont = document.getElementById('prediction-result');
    predictDont.textContent = `-- ${genPredictionTxt()}. . . --`;

    fetch('/api/predict')
        .then(response => response.json())
        .then(data => {
            isPredicting = false;

            const scores = data.scores.map(score => parseFloat(score));
            const totalScore = scores.reduce((sum, score) => sum + score, 0);

            const bestActionIndex = data.predicted;
            const bestActionScore = scores[bestActionIndex];
            const confidence = (bestActionScore / totalScore) * 100;

            const predictText = `Move: ${data.predicted_txt} | Scroll down for more informations.`;
            predictDont.textContent = `${predictText}`;

            const confidenceList = document.getElementById('confidence-list');
            confidenceList.innerHTML = '';
            readlabe_actions = ["[Up/Hoch]", "[Left/Links]", "[Down/Runter]", "[Right/Rechts]"]

            data.scores.forEach((score, index) => {
                const confidenceItem = document.createElement('li');
                const confidenceValue = (parseFloat(score) / totalScore) * 100;
                confidenceItem.textContent = `${readlabe_actions[index]}: ${confidenceValue.toFixed(2)}% Confidence | Gained Points: ${format_num(parseInt(score))}`;
                confidenceList.appendChild(confidenceItem);
            });

        })
        .catch(error => {
            isPredicting = false;
            console.error("Prediction request failed:", error);
            predictDont.textContent = `Prediction failed: ${error}`;
        });
}

function getSessionKey() {
    const keyElement = document.getElementById('key-value');
    if (keyElement) {
        return keyElement.textContent.trim();
    } else {
        console.error('Session key element not found');
        return null;
    }
}

function makeMove(direction) {
    const predictDont = document.getElementById('prediction-result');
    if (isPredicting) {
        console.log("A prediction request is in progress.");
        predictDont.textContent = `-- Simulation already running, hold on! --`;
        return;
    }

    predictDont.textContent = `-- Ready --`;
    const confidenceList = document.getElementById('confidence-list');
    confidenceList.textContent = ''

    fetch(`/api/move/${direction}`, {
        method: 'POST'
    })
        .then(response => response.json())
        .then(data => {
            loadScore();
            loadBoard();
            loadMovesCount();
            const autoPredictCheckbox = document.getElementById('auto-predict');
            if (autoPredictCheckbox && autoPredictCheckbox.checked && !data.game_over) {
                setTimeout(() => {
                    predictMove();
                }, 3000);
            } else if (data.game_over) {
                predictDont.textContent = `-- Game Over --`;
            }

            if (data.best_block == 2048 && !win_confirm) {
                alert("Congrats! You won!\n\nYou have reached a block of 2048. Well done! :)");
                win_confirm = true;
            }

            if (data.game_over) {
                game_over = true
                alert("Game Over!");
            } else {
                checkHighscore(data.score, data.block)
            }
        });
}

function checkHighscore(score, block) {
    fetch('/api/get_highscore')
        .then(response => response.json())
        .then(highscoreData => {
            //const uname = prompt("Congratulations! You set a new highscore! Please enter your username:");
            if (score == highscoreData[1]) {
                if (!currentHighscoreSet) {
                    const uname = prompt("Congratulations! You set a new highscore! Please enter your username:");
                    if (uname && uname.length <= 20) {
                        currentUsername = uname;
                        currentHighscoreSet = true;
                        saveHighscore(currentUsername, score, block);
                        alert("Your highscore has been saved!");
                    }
                } else {
                    updateHighscore(currentUsername, score, block);
                }
            }
        });
}

function saveHighscore(uname, score, block) {
    fetch('/api/add_highscore', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({uname: uname, score: score, block: block })
    })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (data.status === 'success') {
                loadLeaderboard();
            } else {
                alert("Failed to save highscore.");
            }
        })
        .catch(error => {
            console.error('There was a problem with the fetch operation:', error);
            alert("There was an error saving your highscore. Please try again.");
        });
}

function updateHighscore(uname, score, block) {
    fetch('/api/update_highscore', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ uname: uname, score: score, block: block })
    })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (data.status === 'success') {
                alert("Your highscore has been updated!");
                loadLeaderboard();
            } else {
                alert("Failed to update highscore.");
            }
        })
        .catch(error => {
            console.error('There was a problem with the fetch operation:', error);
            alert("There was an error updating your highscore. Please try again.");
        });
}

function resetGame() {
    if (showConfirmation()) {
        const playerScore = document.getElementById('player-score');
        playerScore.textContent = `Your Player Points: 0 | Best-Block: [ 0 ]`;

        const predictDont = document.getElementById('prediction-result');
        predictDont.textContent = `-- Ready --`;

        isPredicting = false
        win_confirm = false
        game_over = false

        fetch('/api/reset', {
            method: 'POST'
        })
            .then(() => {
                loadBoard();
                loadMovesCount();
                loadLeaderboard();
            });
    }
}

function getTime() {
    const now = new Date();

    const day = String(now.getDate()).padStart(2, '0');
    const month = String(now.getMonth() + 1).padStart(2, '0');
    const year = String(now.getFullYear()).slice(-2);

    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    const seconds = String(now.getSeconds()).padStart(2, '0');

    const formattedDateTime = `${day}_${month}_${year}__${hours}_${minutes}_${seconds}`;

    return formattedDateTime;
}

function gen_key() {
    const characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    const keyLength = 10;
    let key = `${getTime()}_`;
    for (let i = 0; i < keyLength; i++) {
        key += characters.charAt(Math.floor(Math.random() * characters.length));
    }
    return key;
}


function toggleDarkMode() {
    const body = document.querySelector('body');
    body.classList.toggle('dark-mode');
}


function initializeSession() {
    const key = gen_key();
    document.getElementById('key-value').textContent = key;

    fetch('/api/generate_key', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ key: key })
    })
        .then(response => response.json())
        .then(data => {
            loadBoard();
            loadMovesCount();
        })
        .catch(error => {
            console.error('Error generating or storing key:', error);
        });
}

document.addEventListener('keydown', function (event) {
    switch (event.key) {
        case 'w':
        case 'W':
            makeMove(0);
            break;
        case 'a':
        case 'A':
            makeMove(1);
            break;
        case 's':
        case 'S':
            makeMove(2);
            break;
        case 'd':
        case 'D':
            makeMove(3);
            break;
    }
});