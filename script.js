let isPlaying = false;  // Track play state

function detectEmotion() {
    document.getElementById("videoFeed").style.display = "block";

    fetch("/detect_emotion", { method: "POST" })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
            } else {
                document.getElementById("emotionText").innerText = "Detected Emotion: " + data.emotion;
                document.getElementById("songText").innerText = "🎶 Playing: " + data.song;
                isPlaying = true;
                document.getElementById("playPauseBtn").innerText = "⏸ Pause";
            }
        });
}

function togglePlayPause() {
    if (isPlaying) {
        fetch("/pause");
        document.getElementById("playPauseBtn").innerText = "▶ Play";
    } else {
        fetch("/play");
        document.getElementById("playPauseBtn").innerText = "⏸ Pause";
    }
    isPlaying = !isPlaying;
}

function stopSong() { 
    fetch("/stop"); 
    document.getElementById("playPauseBtn").innerText = "▶ Play";  
    isPlaying = false;
}

function nextSong() { 
    fetch("/next")
        .then(response => response.json())
        .then(data => {
            if (data.song) {
                document.getElementById("songText").innerText = "🎶 Playing: " + data.song;
                document.getElementById("playPauseBtn").innerText = "⏸ Pause";  
                isPlaying = true;
            } else {
                alert(data.error || "No next song available.");
            }
        });
}

function prevSong() { 
    fetch("/previous")
        .then(response => response.json())
        .then(data => {
            if (data.song) {
                document.getElementById("songText").innerText = "🎶 Playing: " + data.song;
                document.getElementById("playPauseBtn").innerText = "⏸ Pause";  
                isPlaying = true;
            } else {
                alert(data.error || "No previous song available.");
            }
        });
}

