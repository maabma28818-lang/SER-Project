 // Well-commented, modular JS for live recording and uploads



// For Live Mic Page

if (document.getElementById('start')) {

    let recorder;

    let chunks = [];

    const startBtn = document.getElementById('start');

    const stopBtn = document.getElementById('stop');

    const analyzeBtn = document.getElementById('analyze');

    const resultDiv = document.getElementById('result');



    startBtn.addEventListener('click', async () => {

        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        recorder = new MediaRecorder(stream);

        recorder.ondataavailable = e => chunks.push(e.data);

        recorder.onstop = () => {

            analyzeBtn.disabled = false;

        };

        recorder.start();

        startBtn.disabled = true;

        stopBtn.disabled = false;

    });



    stopBtn.addEventListener('click', () => {

        recorder.stop();

        stopBtn.disabled = true;

    });



    analyzeBtn.addEventListener('click', () => {

        const blob = new Blob(chunks, { type: 'audio/wav' });

        chunks = [];  // Reset for next recording

        const formData = new FormData();

        formData.append('file', blob, 'recording.wav');



        fetch('/predict', {

            method: 'POST',

            body: formData

        })

        .then(res => res.json())

        .then(data => {

            resultDiv.textContent = `Detected Emotion: ${data.emotion}`;

            resultDiv.classList.add('animate');

            startBtn.disabled = false;  // Reset for new recording

        })

        .catch(err => {

            resultDiv.textContent = 'Error: ' + err;

        });

    });

}



// For Audio Analysis Page

if (document.getElementById('fileInput')) {

    const fileInput = document.getElementById('fileInput');

    const uploadBtn = document.getElementById('upload');

    const resultDiv = document.getElementById('result');

    const fileLabel = document.getElementById('fileLabel');



    fileInput.addEventListener('change', () => {

        if (fileInput.files.length > 0) {

            fileLabel.setAttribute('data-value', fileInput.files[0].name);

        } else {

            fileLabel.setAttribute('data-value', 'No file chosen');

        }

    });



    uploadBtn.addEventListener('click', () => {

        const file = fileInput.files[0];

        if (!file) {

            resultDiv.textContent = 'No file selected.';

            return;

        }

        const formData = new FormData();

        formData.append('file', file);



        fetch('/predict', {

            method: 'POST',

            body: formData

        })

        .then(res => res.json())

        .then(data => {

            resultDiv.textContent = `Detected Emotion: ${data.emotion}`;

            resultDiv.classList.add('animate');

        })

        .catch(err => {

            resultDiv.textContent = 'Error: ' + err;

        });

    });

}